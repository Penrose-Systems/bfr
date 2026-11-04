#include "bfr/core/error.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/pipeline.h"
#include "d3d12/d3d12_device.h"
#include "d3d12/d3d12_pipeline.h"
#include "rhi/command_check.h"

#include <d3d12.h>
#include <dxgiformat.h>
#include <windows.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <utility>

namespace bfr::rhi::d3d12 {

namespace {

// The list topology of a pipeline's class, which the PSO fixes the class of.
D3D_PRIMITIVE_TOPOLOGY listTopologyFor(TopologyClass topology) {
    switch (topology) {
        case TopologyClass::Point:
            return D3D_PRIMITIVE_TOPOLOGY_POINTLIST;
        case TopologyClass::Line:
            return D3D_PRIMITIVE_TOPOLOGY_LINELIST;
        case TopologyClass::Triangle:
        default:
            return D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    }
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
    if (auto checked = checkRendering((*list)->state); !checked) {
        return checked;
    }
    const D3D12_VIEWPORT set{.TopLeftX = viewport.x,
                             .TopLeftY = viewport.y,
                             .Width = viewport.width,
                             .Height = viewport.height,
                             .MinDepth = viewport.minDepth,
                             .MaxDepth = viewport.maxDepth};
    (*list)->list->RSSetViewports(1, &set);
    return {};
}

core::Result<void> D3D12Device::setScissor(CommandListHandle commandList, const Rect& scissor) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if (auto checked = checkRendering((*list)->state); !checked) {
        return checked;
    }
    const D3D12_RECT rect{.left = static_cast<LONG>(scissor.x),
                          .top = static_cast<LONG>(scissor.y),
                          .right = static_cast<LONG>(scissor.x + scissor.width),
                          .bottom = static_cast<LONG>(scissor.y + scissor.height)};
    (*list)->list->RSSetScissorRects(1, &rect);
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
    if ((*found)->compute) {
        (*list)->state.computeBound = true;
    }
    else {
        (*list)->list->IASetPrimitiveTopology(listTopologyFor((*found)->topology));
        (*list)->state.graphicsBound = true;
    }
    return {};
}

core::Result<void> D3D12Device::pushConstants(CommandListHandle commandList, std::span<const std::byte> data) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if (auto checked = checkPushConstants(data); !checked) {
        return checked;
    }
    D3D12CommandPool* pool = commandPools_.find((*list)->pool);
    if (pool == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the list's pool is gone");
    }
    if (auto counted = countPushConstants(pool->pushed, pool->pushConstantUpdates); !counted) {
        return counted;
    }
    // Root constants are per kind of pipeline, so both kinds the queue runs
    // get them; every shader reads the one block.
    const auto count = static_cast<UINT>(data.size() / sizeof(std::uint32_t));
    const D3D12_COMMAND_LIST_TYPE type = queues_[pool->queue]->listType;
    if (type != D3D12_COMMAND_LIST_TYPE_COPY) {
        (*list)->list->SetComputeRoot32BitConstants(kPushConstantsParameter, count, data.data(), 0);
    }
    if (type == D3D12_COMMAND_LIST_TYPE_DIRECT) {
        (*list)->list->SetGraphicsRoot32BitConstants(kPushConstantsParameter, count, data.data(), 0);
    }
    return {};
}

core::Result<void> D3D12Device::bindIndexBuffer(CommandListHandle commandList, BufferHandle buffer,
                                                std::uint64_t offset, IndexType type) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto found = bufferFor(buffer);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    const D3D12_INDEX_BUFFER_VIEW view{.BufferLocation = (*found)->resource->GetGPUVirtualAddress() + offset,
                                       .SizeInBytes = static_cast<UINT>((*found)->size - offset),
                                       .Format = type == IndexType::Uint16 ? DXGI_FORMAT_R16_UINT
                                                                           : DXGI_FORMAT_R32_UINT};
    (*list)->list->IASetIndexBuffer(&view);
    (*list)->state.indexBound = true;
    return {};
}

core::Result<void> D3D12Device::draw(CommandListHandle commandList, std::uint32_t vertexCount,
                                     std::uint32_t instanceCount, std::uint32_t firstVertex,
                                     std::uint32_t firstInstance) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    if (auto checked = checkDraw((*list)->state, false); !checked) {
        return checked;
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
    if (auto checked = checkDraw((*list)->state, true); !checked) {
        return checked;
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
    if (auto checked = checkDispatch((*list)->state); !checked) {
        return checked;
    }
    (*list)->list->Dispatch(x, y, z);
    return {};
}

} // namespace bfr::rhi::d3d12
