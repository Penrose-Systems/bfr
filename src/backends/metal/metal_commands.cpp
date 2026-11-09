#include "bfr/core/error.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/pipeline.h"
#include "metal/metal_caps.h"
#include "metal/metal_device.h"
#include "metal/metal_pipeline.h"
#include "metal/metal_util.h"
#include "rhi/acceleration_structure_check.h"
#include "rhi/command_check.h"
#include "rhi/enum_check.h"
#include "rhi/ray_tracing_check.h"

#include <Foundation/NSTypes.hpp>
#include <Metal/MTL4ArgumentTable.hpp>
#include <Metal/MTL4CommandBuffer.hpp>
#include <Metal/MTL4ComputeCommandEncoder.hpp>
#include <Metal/MTL4RenderCommandEncoder.hpp>
#include <Metal/MTLArgument.hpp>
#include <Metal/MTLBuffer.hpp>
#include <Metal/MTLComputeCommandEncoder.hpp>
#include <Metal/MTLGPUAddress.hpp>
#include <Metal/MTLRenderCommandEncoder.hpp>
#include <Metal/MTLTypes.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <span>
#include <utility>

namespace bfr::rhi::metal {

namespace {

// Metal reads BFR's indirect arguments as its own, so they are the same.
static_assert(sizeof(DrawIndirectArgs) == sizeof(MTL::DrawPrimitivesIndirectArguments));
static_assert(sizeof(DrawIndexedIndirectArgs) == sizeof(MTL::DrawIndexedPrimitivesIndirectArguments));
static_assert(sizeof(DispatchIndirectArgs) == sizeof(MTL::DispatchThreadgroupsIndirectArguments));

MTL::CullMode cullFor(CullMode cull) {
    switch (cull) {
        case CullMode::Front:
            return MTL::CullModeFront;
        case CullMode::Back:
            return MTL::CullModeBack;
        case CullMode::None:
        default:
            return MTL::CullModeNone;
    }
}

// A compute encoder on buffer for one dispatch of pipeline, through the list's
// arguments; the caller's autorelease pool drains it. Fails with
// RhiDeviceFailed when Metal gives none.
core::Result<MTL4::ComputeCommandEncoder*> computeEncoderFor(MTL4::CommandBuffer& buffer,
                                                             const MTL4::ArgumentTable& arguments,
                                                             const MetalPipeline& pipeline) {
    MTL4::ComputeCommandEncoder* encoder = buffer.computeCommandEncoder();
    if (encoder == nullptr) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, "Metal gave no compute command encoder");
    }
    encoder->setArgumentTable(&arguments);
    encoder->setComputePipelineState(pipeline.compute.get());
    return encoder;
}

// The threads of each of pipeline's threadgroups, which Metal 4 takes at the
// dispatch.
MTL::Size threadsOf(const MetalPipeline& pipeline) {
    const auto& workgroup = pipeline.workgroupSize;
    return MTL::Size::Make(workgroup[0], workgroup[1], workgroup[2]);
}

// BFR has no Metal mesh draws yet, and Features::meshShader is false until it
// does (owner, 2026-11-08), so checkMeshDraws refuses every one.
core::Result<void> noMeshDraws(const Features& features) {
    return checkMeshDraws(features.meshShader);
}

} // namespace

// The device's recording commands between beginRendering and a submit: the
// state draws and dispatches take, and the draws and dispatches. Each makes
// the usage checks every backend makes, then records on the encoder. Metal
// sets pipelines on encoders, so a graphics pipeline bound outside rendering
// is set on the encoder at the first draw after it.

core::Result<const MetalPipeline*> MetalDevice::applyGraphics(MetalCommandList& list) {
    // Looked up every draw: the pipeline may have been destroyed since the last.
    auto found = pipelineFor(list.graphics);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    if (list.encoderGraphics == list.graphics) {
        return *found;
    }
    const MetalPipeline& pipeline = **found;
    MTL4::RenderCommandEncoder& encoder = *list.encoder.get();
    encoder.setRenderPipelineState(pipeline.render.get());
    encoder.setDepthStencilState(pipeline.depthStencil.get());
    encoder.setCullMode(cullFor(pipeline.raster.cull));
    encoder.setFrontFacingWinding(pipeline.raster.frontFace == FrontFace::Clockwise ? MTL::WindingClockwise
                                                                                    : MTL::WindingCounterClockwise);
    encoder.setTriangleFillMode(pipeline.raster.fill == FillMode::Wireframe ? MTL::TriangleFillModeLines
                                                                            : MTL::TriangleFillModeFill);
    encoder.setDepthBias(pipeline.raster.depthBias, pipeline.raster.depthBiasSlope, pipeline.raster.depthBiasClamp);
    encoder.setDepthClipMode(pipeline.raster.depthClamp ? MTL::DepthClipModeClamp : MTL::DepthClipModeClip);
    list.encoderGraphics = list.graphics;
    return *found;
}

core::Result<void> MetalDevice::setViewport(CommandListHandle commandList, const Viewport& viewport) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if (auto checked = checkRendering((*list)->state); !checked) {
        return checked;
    }
    (*list)->encoder->setViewport({.originX = static_cast<double>(viewport.x),
                                   .originY = static_cast<double>(viewport.y),
                                   .width = static_cast<double>(viewport.width),
                                   .height = static_cast<double>(viewport.height),
                                   .znear = static_cast<double>(viewport.minDepth),
                                   .zfar = static_cast<double>(viewport.maxDepth)});
    return {};
}

core::Result<void> MetalDevice::setScissor(CommandListHandle commandList, const Rect& scissor) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if (auto checked = checkRendering((*list)->state); !checked) {
        return checked;
    }
    (*list)->encoder->setScissorRect(
        {.x = scissor.x, .y = scissor.y, .width = scissor.width, .height = scissor.height});
    return {};
}

core::Result<void> MetalDevice::setBlendConstants(CommandListHandle commandList, const std::array<float, 4>& color) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if (auto checked = checkRendering((*list)->state); !checked) {
        return checked;
    }
    const auto [red, green, blue, alpha] = color;
    (*list)->encoder->setBlendColor(red, green, blue, alpha);
    return {};
}

core::Result<void> MetalDevice::setStencilReference(CommandListHandle commandList, std::uint32_t reference) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if (auto checked = checkRendering((*list)->state); !checked) {
        return checked;
    }
    (*list)->encoder->setStencilReferenceValue(reference);
    return {};
}

core::Result<void> MetalDevice::bindPipeline(CommandListHandle commandList, PipelineHandle pipeline) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto found = pipelineFor(pipeline);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    if ((*found)->compute) {
        (*list)->compute = pipeline;
    }
    else {
        (*list)->graphics = pipeline;
        (*list)->primitive = (*found)->primitive;
    }
    return {};
}

core::Result<void> MetalDevice::setTopology(CommandListHandle commandList, Topology topology) {
    if (auto checked = checkKnown(topology, "topology"); !checked) {
        return checked;
    }
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto primitive = primitiveFor(topology);
    if (!primitive) {
        return std::unexpected(std::move(primitive.error()));
    }
    (*list)->primitive = *primitive;
    return {};
}

core::Result<void> MetalDevice::pushConstants(CommandListHandle commandList, std::span<const std::byte> data) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if (auto checked = checkPushConstants(data, kPushConstantsSize); !checked) {
        return checked;
    }
    MetalCommandPool* pool = commandPools_.find((*list)->pool);
    if (pool == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the list's pool is gone");
    }
    const std::uint32_t entry = pool->pushed;
    if (auto counted = countPushConstants(pool->pushed, pool->pushConstantUpdates); !counted) {
        return counted;
    }
    // The next entry of the pool's ring takes the list's last push, then the
    // bytes, and the argument table points at it for the draws and
    // dispatches after this.
    const MetalListSlot& first = lists_[pool->first];
    const std::uint64_t offset = std::uint64_t{entry} * kPushConstantsSize;
    auto* const ring = static_cast<std::byte*>(first.pushRing->contents());
    if ((*list)->pushedAt) {
        std::memcpy(ring + offset, ring + *(*list)->pushedAt, kPushConstantsSize);
    }
    std::memcpy(ring + offset, data.data(), data.size());
    (*list)->pushedAt = offset;
    lists_[(*list)->slot].arguments->setAddress(first.pushRing->gpuAddress() + offset, kPushConstantsBufferIndex);
    return {};
}

core::Result<void> MetalDevice::bindIndexBuffer(CommandListHandle commandList, BufferHandle buffer,
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
    // Metal takes the bytes from offset to the end, which there are none of
    // past it.
    if (offset > (*found)->buffer->length()) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "index offset {} is past the buffer's {} bytes", offset,
                             (*found)->buffer->length());
    }
    (*list)->indexType = type == IndexType::Uint16 ? MTL::IndexTypeUInt16 : MTL::IndexTypeUInt32;
    (*list)->indexBuffer = (*found)->buffer->gpuAddress() + offset;
    (*list)->indexBytes = (*found)->buffer->length() - offset;
    return {};
}

core::Result<void> MetalDevice::draw(CommandListHandle commandList, std::uint32_t vertexCount,
                                     std::uint32_t instanceCount, std::uint32_t firstVertex,
                                     std::uint32_t firstInstance) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if (auto checked = checkRendering((*list)->state); !checked) {
        return checked;
    }
    const auto pipeline = applyGraphics(**list);
    if (!pipeline) {
        return std::unexpected(std::move(pipeline.error()));
    }
    (*list)->encoder->drawPrimitives((*list)->primitive, firstVertex, vertexCount, instanceCount, firstInstance);
    return {};
}

core::Result<void> MetalDevice::drawIndexed(CommandListHandle commandList, std::uint32_t indexCount,
                                            std::uint32_t instanceCount, std::uint32_t firstIndex,
                                            std::int32_t vertexOffset, std::uint32_t firstInstance) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if (auto checked = checkRendering((*list)->state); !checked) {
        return checked;
    }
    // Metal takes the first index as an offset into the buffer.
    const NS::UInteger indexBytes = (*list)->indexType == MTL::IndexTypeUInt16 ? sizeof(std::uint16_t)
                                                                               : sizeof(std::uint32_t);
    const NS::UInteger skipped = NS::UInteger{firstIndex} * indexBytes;
    if (skipped > (*list)->indexBytes) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "first index {} is past the index buffer's end",
                             firstIndex);
    }
    const auto pipeline = applyGraphics(**list);
    if (!pipeline) {
        return std::unexpected(std::move(pipeline.error()));
    }
    (*list)->encoder->drawIndexedPrimitives((*list)->primitive, indexCount, (*list)->indexType,
                                            (*list)->indexBuffer + skipped, (*list)->indexBytes - skipped,
                                            instanceCount, vertexOffset, firstInstance);
    return {};
}

core::Result<void> MetalDevice::dispatch(CommandListHandle commandList, std::uint32_t x, std::uint32_t y,
                                         std::uint32_t z) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto pipeline = pipelineFor((*list)->compute);
    if (!pipeline) {
        return std::unexpected(std::move(pipeline.error()));
    }
    // One encoder per dispatch, which the pool drains the autoreleased
    // reference of.
    const AutoreleasePool pool;
    auto encoder = computeEncoderFor(*(*list)->buffer, *lists_[(*list)->slot].arguments.get(), **pipeline);
    if (!encoder) {
        return std::unexpected(std::move(encoder.error()));
    }
    (*encoder)->dispatchThreadgroups(MTL::Size::Make(x, y, z), threadsOf(**pipeline));
    (*encoder)->endEncoding();
    return {};
}

// Metal 4 draws one indirect draw per call, so drawCount is 0 or 1 past
// checkDrawCount, and stride, the distance between draws, is unused. Its
// encoder reads the arguments at a GPU address; the buffer is resident, as
// every buffer is from creation.

core::Result<void> MetalDevice::drawIndirect(CommandListHandle commandList, BufferHandle buffer, std::uint64_t offset,
                                             std::uint32_t drawCount, std::uint32_t /*stride*/) {
    if (auto checked = checkDrawCount(drawCount, kMultiDrawIndirect); !checked) {
        return checked;
    }
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if (auto checked = checkRendering((*list)->state); !checked) {
        return checked;
    }
    auto found = bufferFor(buffer);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    auto pipeline = applyGraphics(**list);
    if (!pipeline) {
        return std::unexpected(std::move(pipeline.error()));
    }
    // Metal's indirect draw always draws once, so none is recorded for 0.
    if (drawCount != 0) {
        (*list)->encoder->drawPrimitives((*list)->primitive, (*found)->buffer->gpuAddress() + offset);
    }
    return {};
}

core::Result<void> MetalDevice::drawIndexedIndirect(CommandListHandle commandList, BufferHandle buffer,
                                                    std::uint64_t offset, std::uint32_t drawCount,
                                                    std::uint32_t /*stride*/) {
    if (auto checked = checkDrawCount(drawCount, kMultiDrawIndirect); !checked) {
        return checked;
    }
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if (auto checked = checkRendering((*list)->state); !checked) {
        return checked;
    }
    auto found = bufferFor(buffer);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    auto pipeline = applyGraphics(**list);
    if (!pipeline) {
        return std::unexpected(std::move(pipeline.error()));
    }
    // The arguments name the first index, so the whole bound index buffer is
    // given, unlike drawIndexed's.
    if (drawCount != 0) {
        (*list)->encoder->drawIndexedPrimitives((*list)->primitive, (*list)->indexType, (*list)->indexBuffer,
                                                (*list)->indexBytes, (*found)->buffer->gpuAddress() + offset);
    }
    return {};
}

core::Result<void> MetalDevice::drawIndirectCount(CommandListHandle /*commandList*/, BufferHandle /*buffer*/,
                                                  std::uint64_t /*offset*/, BufferHandle /*countBuffer*/,
                                                  std::uint64_t /*countOffset*/, std::uint32_t /*maxDrawCount*/,
                                                  std::uint32_t /*stride*/) {
    return checkCountDraws(kDrawIndirectCount);
}

core::Result<void> MetalDevice::drawIndexedIndirectCount(CommandListHandle /*commandList*/, BufferHandle /*buffer*/,
                                                         std::uint64_t /*offset*/, BufferHandle /*countBuffer*/,
                                                         std::uint64_t /*countOffset*/, std::uint32_t /*maxDrawCount*/,
                                                         std::uint32_t /*stride*/) {
    return checkCountDraws(kDrawIndirectCount);
}

core::Result<void> MetalDevice::dispatchIndirect(CommandListHandle commandList, BufferHandle buffer,
                                                 std::uint64_t offset) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto pipeline = pipelineFor((*list)->compute);
    if (!pipeline) {
        return std::unexpected(std::move(pipeline.error()));
    }
    auto found = bufferFor(buffer);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    const AutoreleasePool pool;
    auto encoder = computeEncoderFor(*(*list)->buffer, *lists_[(*list)->slot].arguments.get(), **pipeline);
    if (!encoder) {
        return std::unexpected(std::move(encoder.error()));
    }
    (*encoder)->dispatchThreadgroups((*found)->buffer->gpuAddress() + offset, threadsOf(**pipeline));
    (*encoder)->endEncoding();
    return {};
}

core::Result<void> MetalDevice::drawMesh(CommandListHandle /*commandList*/, std::uint32_t /*x*/, std::uint32_t /*y*/,
                                         std::uint32_t /*z*/) {
    return noMeshDraws(features_);
}

core::Result<void> MetalDevice::drawMeshIndirect(CommandListHandle /*commandList*/, BufferHandle /*buffer*/,
                                                 std::uint64_t /*offset*/, std::uint32_t /*drawCount*/,
                                                 std::uint32_t /*stride*/) {
    return noMeshDraws(features_);
}

core::Result<void> MetalDevice::drawMeshIndirectCount(CommandListHandle /*commandList*/, BufferHandle /*buffer*/,
                                                      std::uint64_t /*offset*/, BufferHandle /*countBuffer*/,
                                                      std::uint64_t /*countOffset*/, std::uint32_t /*maxDrawCount*/,
                                                      std::uint32_t /*stride*/) {
    return noMeshDraws(features_);
}

// BFR has no Metal acceleration structures yet, and
// Features::accelerationStructure is false until it does (owner, 2026-11-08),
// so checkAccelerationStructures refuses every call.

core::Result<AccelerationStructureSizes> MetalDevice::accelerationStructureSizes(
    const AccelerationStructureInputs& /*inputs*/) {
    return std::unexpected(checkAccelerationStructures(features_.accelerationStructure).error());
}

core::Result<AccelerationStructureHandle> MetalDevice::createAccelerationStructure(
    const AccelerationStructureDesc& /*desc*/) {
    return std::unexpected(checkAccelerationStructures(features_.accelerationStructure).error());
}

core::Result<std::uint32_t> MetalDevice::accelerationStructureSlot(AccelerationStructureHandle /*structure*/) {
    return std::unexpected(checkAccelerationStructures(features_.accelerationStructure).error());
}

core::Result<std::uint64_t> MetalDevice::accelerationStructureReference(AccelerationStructureHandle /*structure*/) {
    return std::unexpected(checkAccelerationStructures(features_.accelerationStructure).error());
}

core::Result<void> MetalDevice::destroyAccelerationStructure(AccelerationStructureHandle structure) {
    if (structure.isNull()) {
        return {};
    }
    return checkAccelerationStructures(features_.accelerationStructure);
}

core::Result<void> MetalDevice::writeInstances(std::span<const AccelerationStructureInstance> /*instances*/,
                                               std::span<std::byte> /*out*/) {
    return checkAccelerationStructures(features_.accelerationStructure);
}

core::Result<void> MetalDevice::buildAccelerationStructure(CommandListHandle /*commandList*/,
                                                           const AccelerationStructureBuild& /*build*/) {
    return checkAccelerationStructures(features_.accelerationStructure);
}

core::Result<void> MetalDevice::copyAccelerationStructure(CommandListHandle /*commandList*/,
                                                          AccelerationStructureHandle /*dst*/,
                                                          AccelerationStructureHandle /*src*/) {
    return checkAccelerationStructures(features_.accelerationStructure);
}

// BFR has no Metal shader objects or ray tracing pipelines yet, and
// Features::rayTracingPipeline, traceRaysIndirect and bufferAddress are false
// until it does (owner, 2026-11-08).

core::Result<ShaderHandle> MetalDevice::createShader(const ShaderCode& /*code*/) {
    return core::failure(core::ErrorCode::RhiUnsupported, "BFR has no Metal shader objects yet");
}

core::Result<void> MetalDevice::destroyShader(ShaderHandle shader) {
    if (shader.isNull()) {
        return {};
    }
    return core::failure(core::ErrorCode::RhiInvalidHandle, "the shader handle is stale");
}

core::Result<PipelineHandle> MetalDevice::createRayTracingPipeline(const RayTracingPipelineDesc& /*desc*/) {
    return std::unexpected(checkRayTracingPipelines(features_.rayTracingPipeline).error());
}

core::Result<void> MetalDevice::shaderIdentifiers(PipelineHandle /*pipeline*/, std::uint32_t /*first*/,
                                                  std::uint32_t /*count*/, std::span<std::byte> /*out*/) {
    return checkRayTracingPipelines(features_.rayTracingPipeline);
}

core::Result<void> MetalDevice::captureReplayIdentifiers(PipelineHandle /*pipeline*/, std::uint32_t /*first*/,
                                                         std::uint32_t /*count*/, std::span<std::byte> /*out*/) {
    return checkCaptureReplay(features_.rayTracingCaptureReplay);
}

core::Result<void> MetalDevice::buildAccelerationStructureIndirect(CommandListHandle /*commandList*/,
                                                                   const AccelerationStructureBuild& /*build*/,
                                                                   BufferHandle /*ranges*/, std::uint64_t /*offset*/,
                                                                   std::uint32_t /*stride*/) {
    return checkIndirectBuilds(features_.accelerationStructureIndirectBuild);
}

core::Result<void> MetalDevice::traceRays(CommandListHandle /*commandList*/, const TraceRaysDesc& /*desc*/) {
    return checkRayTracingPipelines(features_.rayTracingPipeline);
}

core::Result<void> MetalDevice::traceRaysIndirect(CommandListHandle /*commandList*/, BufferHandle /*buffer*/,
                                                  std::uint64_t /*offset*/) {
    return checkTraceRaysIndirect(features_.traceRaysIndirect);
}

core::Result<std::uint64_t> MetalDevice::bufferAddress(BufferHandle /*buffer*/) {
    return std::unexpected(checkBufferAddress(features_.bufferAddress, false).error());
}

core::Result<AccelerationStructurePropertyPoolHandle> MetalDevice::createAccelerationStructurePropertyPool(
    const AccelerationStructurePropertyPoolDesc& /*desc*/) {
    return std::unexpected(checkAccelerationStructures(features_.accelerationStructure).error());
}

core::Result<void> MetalDevice::destroyAccelerationStructurePropertyPool(AccelerationStructurePropertyPoolHandle pool) {
    if (pool.isNull()) {
        return {};
    }
    return checkAccelerationStructures(features_.accelerationStructure);
}

core::Result<void> MetalDevice::resetAccelerationStructureProperties(CommandListHandle /*commandList*/,
                                                                     AccelerationStructurePropertyPoolHandle /*pool*/,
                                                                     std::uint32_t /*first*/, std::uint32_t /*count*/) {
    return checkAccelerationStructures(features_.accelerationStructure);
}

core::Result<void> MetalDevice::writeAccelerationStructureProperties(
    CommandListHandle /*commandList*/, AccelerationStructurePropertyPoolHandle /*pool*/, std::uint32_t /*first*/,
    std::span<const AccelerationStructureHandle> /*structures*/) {
    return checkAccelerationStructures(features_.accelerationStructure);
}

core::Result<void> MetalDevice::resolveAccelerationStructureProperties(CommandListHandle /*commandList*/,
                                                                       AccelerationStructurePropertyPoolHandle /*pool*/,
                                                                       std::uint32_t /*first*/, std::uint32_t /*count*/,
                                                                       BufferHandle /*buffer*/,
                                                                       std::uint64_t /*offset*/) {
    return checkAccelerationStructures(features_.accelerationStructure);
}

core::Result<void> MetalDevice::compactAccelerationStructure(CommandListHandle /*commandList*/,
                                                             AccelerationStructureHandle /*dst*/,
                                                             AccelerationStructureHandle /*src*/) {
    return checkAccelerationStructures(features_.accelerationStructure);
}

core::Result<void> MetalDevice::serializeAccelerationStructure(CommandListHandle /*commandList*/,
                                                               AccelerationStructureHandle /*src*/,
                                                               BufferHandle /*buffer*/, std::uint64_t /*offset*/) {
    return checkSerialization(features_.accelerationStructureSerialization);
}

core::Result<void> MetalDevice::deserializeAccelerationStructure(CommandListHandle /*commandList*/,
                                                                 BufferHandle /*buffer*/, std::uint64_t /*offset*/,
                                                                 AccelerationStructureHandle /*dst*/) {
    return checkSerialization(features_.accelerationStructureSerialization);
}

core::Result<bool> MetalDevice::serializedCompatible(
    AccelerationStructureType /*type*/, std::span<const std::byte, kSerializedIdentifierSize> /*identifier*/) const {
    return std::unexpected(checkSerialization(features_.accelerationStructureSerialization).error());
}

core::Result<std::uint64_t> MetalDevice::rayTracingStackSize(PipelineHandle /*pipeline*/, std::uint32_t /*group*/,
                                                             RayTracingGroupShader /*shader*/) {
    return std::unexpected(checkRayTracingPipelines(features_.rayTracingPipeline).error());
}

} // namespace bfr::rhi::metal
