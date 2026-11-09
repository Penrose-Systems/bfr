#include "bfr/core/error.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/pipeline.h"
#include "d3d12/d3d12_device.h"
#include "d3d12/d3d12_pipeline.h"
#include "rhi/acceleration_structure_check.h"
#include "rhi/command_check.h"
#include "rhi/enum_check.h"
#include "rhi/ray_tracing_check.h"

#include <d3d12.h>
#include <dxgiformat.h>
#include <windows.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <utility>

namespace bfr::rhi::d3d12 {

namespace {

// The exact topology draws assemble; the PSO fixes only its class.
D3D_PRIMITIVE_TOPOLOGY topologyFor(Topology topology) {
    switch (topology) {
        case Topology::TriangleStrip:
            return D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
        case Topology::TriangleFan:
            return D3D_PRIMITIVE_TOPOLOGY_TRIANGLEFAN;
        case Topology::LineList:
            return D3D_PRIMITIVE_TOPOLOGY_LINELIST;
        case Topology::LineStrip:
            return D3D_PRIMITIVE_TOPOLOGY_LINESTRIP;
        case Topology::PointList:
            return D3D_PRIMITIVE_TOPOLOGY_POINTLIST;
        case Topology::TriangleList:
        default:
            return D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    }
}

// dispatchIndirect records one dispatch, as Godot's does
// (rendering_device_driver_d3d12.cpp:5685).
constexpr UINT kOneDispatch = 1;

// BFR has no D3D12 mesh draws yet, and Features::meshShader is false until it
// does (owner, 2026-11-08), so checkMeshDraws refuses every one.
core::Result<void> noMeshDraws(const Features& features) {
    return checkMeshDraws(features.meshShader);
}

} // namespace

// The device's recording commands between beginRendering and a submit: the
// state draws and dispatches take, and the draws and dispatches. Each makes
// the usage checks every backend makes, then records one call.

core::Result<void> D3D12Device::setViewport(CommandListHandle commandList, const Viewport& viewport) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    const D3D12_VIEWPORT set{.TopLeftX = viewport.x,
                             .TopLeftY = viewport.y,
                             .Width = viewport.width,
                             .Height = viewport.height,
                             .MinDepth = viewport.minDepth,
                             .MaxDepth = viewport.maxDepth};
    (*list)->list->RSSetViewports(kViewportCount, &set);
    return {};
}

core::Result<void> D3D12Device::setScissor(CommandListHandle commandList, const Rect& scissor) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    // D3D12_RECT holds the edges as LONG.
    const std::uint64_t right = std::uint64_t{scissor.x} + scissor.width;
    const std::uint64_t bottom = std::uint64_t{scissor.y} + scissor.height;
    if (!std::in_range<LONG>(right) || !std::in_range<LONG>(bottom)) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "scissor edges {}, {} are past what D3D12 takes", right,
                             bottom);
    }
    const D3D12_RECT rect{.left = static_cast<LONG>(scissor.x),
                          .top = static_cast<LONG>(scissor.y),
                          .right = static_cast<LONG>(right),
                          .bottom = static_cast<LONG>(bottom)};
    (*list)->list->RSSetScissorRects(kViewportCount, &rect);
    return {};
}

core::Result<void> D3D12Device::setBlendConstants(CommandListHandle commandList, const std::array<float, 4>& color) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    (*list)->list->OMSetBlendFactor(color.data());
    return {};
}

core::Result<void> D3D12Device::setStencilReference(CommandListHandle commandList, std::uint32_t reference) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    (*list)->list->OMSetStencilRef(reference);
    return {};
}

core::Result<void> D3D12Device::bindPipeline(CommandListHandle commandList, PipelineHandle pipeline) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto found = pipelineFor(pipeline);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    (*list)->list->SetPipelineState((*found)->state.Get());
    if ((*found)->compute) {}
    else {
        (*list)->list->IASetPrimitiveTopology(topologyFor((*found)->topology));
    }
    return {};
}

core::Result<void> D3D12Device::setTopology(CommandListHandle commandList, Topology topology) {
    if (auto checked = checkKnown(topology, "topology"); !checked) {
        return checked;
    }
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    (*list)->list->IASetPrimitiveTopology(topologyFor(topology));
    return {};
}

core::Result<void> D3D12Device::pushConstants(CommandListHandle commandList, std::span<const std::byte> data) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if (auto checked = checkPushConstants(data, kPushConstantsSize); !checked) {
        return checked;
    }
    D3D12CommandPool* pool = commandPools_.find((*list)->pool);
    if (pool == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the list's pool is gone");
    }
    // A copy list has no root signature, so nothing would take the bytes.
    const D3D12_COMMAND_LIST_TYPE type = queues_[pool->queue]->listType;
    if (type == D3D12_COMMAND_LIST_TYPE_COPY) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "the list records for copy queue {}, which D3D12 gives no push constants", pool->queue);
    }
    if (auto counted = countPushConstants(pool->pushed, pool->pushConstantUpdates); !counted) {
        return counted;
    }
    // Root constants are per kind of pipeline, so both kinds the queue runs
    // get them; every shader reads the one block.
    const auto count = static_cast<UINT>(data.size() / sizeof(std::uint32_t));
    (*list)->list->SetComputeRoot32BitConstants(kPushConstantsParameter, count, data.data(), 0);
    if (type == D3D12_COMMAND_LIST_TYPE_DIRECT) {
        (*list)->list->SetGraphicsRoot32BitConstants(kPushConstantsParameter, count, data.data(), 0);
    }
    return {};
}

core::Result<void> D3D12Device::bindIndexBuffer(CommandListHandle commandList, BufferHandle buffer,
                                                std::uint64_t offset, IndexType type) {
    if (auto checked = checkKnown(type, "index type"); !checked) {
        return checked;
    }
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto found = bufferFor(buffer);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    // The view runs from offset to the buffer's end, as Vulkan's binding does;
    // D3D12 holds its size as a UINT.
    if (offset > (*found)->size || !std::in_range<UINT>((*found)->size - offset)) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "an index buffer of {} bytes from offset {} is past what D3D12 takes", (*found)->size,
                             offset);
    }
    const D3D12_INDEX_BUFFER_VIEW view{.BufferLocation = (*found)->resource->GetGPUVirtualAddress() + offset,
                                       .SizeInBytes = static_cast<UINT>((*found)->size - offset),
                                       .Format = type == IndexType::Uint16 ? DXGI_FORMAT_R16_UINT
                                                                           : DXGI_FORMAT_R32_UINT};
    (*list)->list->IASetIndexBuffer(&view);
    return {};
}

core::Result<void> D3D12Device::draw(CommandListHandle commandList, std::uint32_t vertexCount,
                                     std::uint32_t instanceCount, std::uint32_t firstVertex,
                                     std::uint32_t firstInstance) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    (*list)->list->DrawInstanced(vertexCount, instanceCount, firstVertex, firstInstance);
    return {};
}

core::Result<void> D3D12Device::drawIndexed(CommandListHandle commandList, std::uint32_t indexCount,
                                            std::uint32_t instanceCount, std::uint32_t firstIndex,
                                            std::int32_t vertexOffset, std::uint32_t firstInstance) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    (*list)->list->DrawIndexedInstanced(indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
    return {};
}

core::Result<void> D3D12Device::dispatch(CommandListHandle commandList, std::uint32_t x, std::uint32_t y,
                                         std::uint32_t z) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    (*list)->list->Dispatch(x, y, z);
    return {};
}

core::Result<void> D3D12Device::executeIndirect(CommandListHandle commandList, IndirectKind kind, std::uint32_t stride,
                                                UINT maxCommandCount, BufferHandle buffer, std::uint64_t offset,
                                                std::optional<BufferHandle> countBuffer, std::uint64_t countOffset) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto found = bufferFor(buffer);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    ID3D12Resource* count = nullptr;
    if (countBuffer) {
        auto counted = bufferFor(*countBuffer);
        if (!counted) {
            return std::unexpected(std::move(counted.error()));
        }
        count = (*counted)->resource.Get();
    }
    ID3D12CommandSignature* signature = dispatchSignature_.Get();
    if (kind != IndirectKind::Dispatch) {
        // A short array the caller sized; the stride comes with each call.
        const auto signatures = std::ranges::find(drawSignatures_, stride, &D3D12DrawSignatures::stride);
        signature = signatures == drawSignatures_.end() ? nullptr
                    : kind == IndirectKind::Draw        ? signatures->draw.Get()
                                                        : signatures->drawIndexed.Get();
        if (signature == nullptr) {
            return core::failure(core::ErrorCode::RhiUnsupported,
                                 "the device has no signature of stride {} for these arguments: the stride is not in "
                                 "DeviceDesc::indirectStrides, or is less than the arguments' size",
                                 stride);
        }
    }
    (*list)->list->ExecuteIndirect(signature, maxCommandCount, (*found)->resource.Get(), offset, count, countOffset);
    return {};
}

core::Result<void> D3D12Device::drawIndirect(CommandListHandle commandList, BufferHandle buffer, std::uint64_t offset,
                                             std::uint32_t drawCount, std::uint32_t stride) {
    if (auto checked = checkDrawCount(drawCount, multiDrawIndirect_); !checked) {
        return checked;
    }
    return executeIndirect(commandList, IndirectKind::Draw, stride, drawCount, buffer, offset, std::nullopt, 0);
}

core::Result<void> D3D12Device::drawIndexedIndirect(CommandListHandle commandList, BufferHandle buffer,
                                                    std::uint64_t offset, std::uint32_t drawCount,
                                                    std::uint32_t stride) {
    if (auto checked = checkDrawCount(drawCount, multiDrawIndirect_); !checked) {
        return checked;
    }
    return executeIndirect(commandList, IndirectKind::DrawIndexed, stride, drawCount, buffer, offset, std::nullopt, 0);
}

core::Result<void> D3D12Device::drawIndirectCount(CommandListHandle commandList, BufferHandle buffer,
                                                  std::uint64_t offset, BufferHandle countBuffer,
                                                  std::uint64_t countOffset, std::uint32_t maxDrawCount,
                                                  std::uint32_t stride) {
    if (auto checked = checkCountDraws(drawIndirectCount_); !checked) {
        return checked;
    }
    return executeIndirect(commandList, IndirectKind::Draw, stride, maxDrawCount, buffer, offset, countBuffer,
                           countOffset);
}

core::Result<void> D3D12Device::drawIndexedIndirectCount(CommandListHandle commandList, BufferHandle buffer,
                                                         std::uint64_t offset, BufferHandle countBuffer,
                                                         std::uint64_t countOffset, std::uint32_t maxDrawCount,
                                                         std::uint32_t stride) {
    if (auto checked = checkCountDraws(drawIndirectCount_); !checked) {
        return checked;
    }
    return executeIndirect(commandList, IndirectKind::DrawIndexed, stride, maxDrawCount, buffer, offset, countBuffer,
                           countOffset);
}

core::Result<void> D3D12Device::dispatchIndirect(CommandListHandle commandList, BufferHandle buffer,
                                                 std::uint64_t offset) {
    return executeIndirect(commandList, IndirectKind::Dispatch, 0, kOneDispatch, buffer, offset, std::nullopt, 0);
}

core::Result<void> D3D12Device::drawMesh(CommandListHandle /*commandList*/, std::uint32_t /*x*/, std::uint32_t /*y*/,
                                         std::uint32_t /*z*/) {
    return noMeshDraws(features_);
}

core::Result<void> D3D12Device::drawMeshIndirect(CommandListHandle /*commandList*/, BufferHandle /*buffer*/,
                                                 std::uint64_t /*offset*/, std::uint32_t /*drawCount*/,
                                                 std::uint32_t /*stride*/) {
    return noMeshDraws(features_);
}

core::Result<void> D3D12Device::drawMeshIndirectCount(CommandListHandle /*commandList*/, BufferHandle /*buffer*/,
                                                      std::uint64_t /*offset*/, BufferHandle /*countBuffer*/,
                                                      std::uint64_t /*countOffset*/, std::uint32_t /*maxDrawCount*/,
                                                      std::uint32_t /*stride*/) {
    return noMeshDraws(features_);
}

// BFR has no D3D12 acceleration structures yet, and
// Features::accelerationStructure is false until it does (owner, 2026-11-08),
// so checkAccelerationStructures refuses every call.

core::Result<AccelerationStructureSizes> D3D12Device::accelerationStructureSizes(
    const AccelerationStructureInputs& /*inputs*/) {
    return std::unexpected(checkAccelerationStructures(features_.accelerationStructure).error());
}

core::Result<AccelerationStructureHandle> D3D12Device::createAccelerationStructure(
    const AccelerationStructureDesc& /*desc*/) {
    return std::unexpected(checkAccelerationStructures(features_.accelerationStructure).error());
}

core::Result<std::uint32_t> D3D12Device::accelerationStructureSlot(AccelerationStructureHandle /*structure*/) {
    return std::unexpected(checkAccelerationStructures(features_.accelerationStructure).error());
}

core::Result<std::uint64_t> D3D12Device::accelerationStructureReference(AccelerationStructureHandle /*structure*/) {
    return std::unexpected(checkAccelerationStructures(features_.accelerationStructure).error());
}

core::Result<void> D3D12Device::destroyAccelerationStructure(AccelerationStructureHandle structure) {
    if (structure.isNull()) {
        return {};
    }
    return checkAccelerationStructures(features_.accelerationStructure);
}

core::Result<void> D3D12Device::writeInstances(std::span<const AccelerationStructureInstance> /*instances*/,
                                               std::span<std::byte> /*out*/) {
    return checkAccelerationStructures(features_.accelerationStructure);
}

core::Result<void> D3D12Device::buildAccelerationStructure(CommandListHandle /*commandList*/,
                                                           const AccelerationStructureBuild& /*build*/) {
    return checkAccelerationStructures(features_.accelerationStructure);
}

core::Result<void> D3D12Device::copyAccelerationStructure(CommandListHandle /*commandList*/,
                                                          AccelerationStructureHandle /*dst*/,
                                                          AccelerationStructureHandle /*src*/) {
    return checkAccelerationStructures(features_.accelerationStructure);
}

// BFR has no D3D12 shader objects or ray tracing pipelines yet, and
// Features::rayTracingPipeline, traceRaysIndirect and bufferAddress are false
// until it does (owner, 2026-11-08).

core::Result<ShaderHandle> D3D12Device::createShader(const ShaderCode& /*code*/) {
    return core::failure(core::ErrorCode::RhiUnsupported, "BFR has no D3D12 shader objects yet");
}

core::Result<void> D3D12Device::destroyShader(ShaderHandle shader) {
    if (shader.isNull()) {
        return {};
    }
    return core::failure(core::ErrorCode::RhiInvalidHandle, "the shader handle is stale");
}

core::Result<PipelineHandle> D3D12Device::createRayTracingPipeline(const RayTracingPipelineDesc& /*desc*/) {
    return std::unexpected(checkRayTracingPipelines(features_.rayTracingPipeline).error());
}

core::Result<void> D3D12Device::shaderIdentifiers(PipelineHandle /*pipeline*/, std::uint32_t /*first*/,
                                                  std::uint32_t /*count*/, std::span<std::byte> /*out*/) {
    return checkRayTracingPipelines(features_.rayTracingPipeline);
}

core::Result<void> D3D12Device::captureReplayIdentifiers(PipelineHandle /*pipeline*/, std::uint32_t /*first*/,
                                                         std::uint32_t /*count*/, std::span<std::byte> /*out*/) {
    return checkCaptureReplay(features_.rayTracingCaptureReplay);
}

core::Result<void> D3D12Device::buildAccelerationStructureIndirect(CommandListHandle /*commandList*/,
                                                                   const AccelerationStructureBuild& /*build*/,
                                                                   BufferHandle /*ranges*/, std::uint64_t /*offset*/,
                                                                   std::uint32_t /*stride*/) {
    return checkIndirectBuilds(features_.accelerationStructureIndirectBuild);
}

core::Result<void> D3D12Device::traceRays(CommandListHandle /*commandList*/, const TraceRaysDesc& /*desc*/) {
    return checkRayTracingPipelines(features_.rayTracingPipeline);
}

core::Result<void> D3D12Device::traceRaysIndirect(CommandListHandle /*commandList*/, BufferHandle /*buffer*/,
                                                  std::uint64_t /*offset*/) {
    return checkTraceRaysIndirect(features_.traceRaysIndirect);
}

core::Result<std::uint64_t> D3D12Device::bufferAddress(BufferHandle /*buffer*/) {
    return std::unexpected(checkBufferAddress(features_.bufferAddress, false).error());
}

core::Result<AccelerationStructurePropertyPoolHandle> D3D12Device::createAccelerationStructurePropertyPool(
    const AccelerationStructurePropertyPoolDesc& /*desc*/) {
    return std::unexpected(checkAccelerationStructures(features_.accelerationStructure).error());
}

core::Result<void> D3D12Device::destroyAccelerationStructurePropertyPool(AccelerationStructurePropertyPoolHandle pool) {
    if (pool.isNull()) {
        return {};
    }
    return checkAccelerationStructures(features_.accelerationStructure);
}

core::Result<void> D3D12Device::resetAccelerationStructureProperties(CommandListHandle /*commandList*/,
                                                                     AccelerationStructurePropertyPoolHandle /*pool*/,
                                                                     std::uint32_t /*first*/, std::uint32_t /*count*/) {
    return checkAccelerationStructures(features_.accelerationStructure);
}

core::Result<void> D3D12Device::writeAccelerationStructureProperties(
    CommandListHandle /*commandList*/, AccelerationStructurePropertyPoolHandle /*pool*/, std::uint32_t /*first*/,
    std::span<const AccelerationStructureHandle> /*structures*/) {
    return checkAccelerationStructures(features_.accelerationStructure);
}

core::Result<void> D3D12Device::resolveAccelerationStructureProperties(CommandListHandle /*commandList*/,
                                                                       AccelerationStructurePropertyPoolHandle /*pool*/,
                                                                       std::uint32_t /*first*/, std::uint32_t /*count*/,
                                                                       BufferHandle /*buffer*/,
                                                                       std::uint64_t /*offset*/) {
    return checkAccelerationStructures(features_.accelerationStructure);
}

core::Result<void> D3D12Device::compactAccelerationStructure(CommandListHandle /*commandList*/,
                                                             AccelerationStructureHandle /*dst*/,
                                                             AccelerationStructureHandle /*src*/) {
    return checkAccelerationStructures(features_.accelerationStructure);
}

core::Result<void> D3D12Device::serializeAccelerationStructure(CommandListHandle /*commandList*/,
                                                               AccelerationStructureHandle /*src*/,
                                                               BufferHandle /*buffer*/, std::uint64_t /*offset*/) {
    return checkSerialization(features_.accelerationStructureSerialization);
}

core::Result<void> D3D12Device::deserializeAccelerationStructure(CommandListHandle /*commandList*/,
                                                                 BufferHandle /*buffer*/, std::uint64_t /*offset*/,
                                                                 AccelerationStructureHandle /*dst*/) {
    return checkSerialization(features_.accelerationStructureSerialization);
}

core::Result<bool> D3D12Device::serializedCompatible(
    std::span<const std::byte, kSerializedIdentifierSize> /*identifier*/) const {
    return std::unexpected(checkSerialization(features_.accelerationStructureSerialization).error());
}

core::Result<std::uint64_t> D3D12Device::rayTracingStackSize(PipelineHandle /*pipeline*/, std::uint32_t /*group*/,
                                                             RayTracingGroupShader /*shader*/) {
    return std::unexpected(checkRayTracingPipelines(features_.rayTracingPipeline).error());
}

} // namespace bfr::rhi::d3d12
