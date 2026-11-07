#include "d3d12/d3d12_sparse.h"

#include "bfr/core/error.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/sparse.h"
#include "d3d12/d3d12_device.h"
#include "d3d12/d3d12_result.h"

#include <d3d12.h>
#include <windows.h>
#include <wrl/client.h>

#include <atomic>
#include <cstdint>
#include <expected>
#include <mutex>
#include <tuple>
#include <utility>

namespace bfr::rhi::d3d12 {

core::Result<void> D3D12TileMappings::add(const D3D12_TILED_RESOURCE_COORDINATE& coordinate,
                                          const D3D12_TILE_REGION_SIZE& size, bool map, std::uint64_t start) {
    if (map && !std::in_range<UINT>(start)) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "memory tile {} is past what D3D12 takes", start);
    }
    std::ignore = coordinates.push(coordinate);
    std::ignore = sizes.push(size);
    std::ignore = flags.push(map ? D3D12_TILE_RANGE_FLAG_NONE : D3D12_TILE_RANGE_FLAG_NULL);
    std::ignore = starts.push(map ? static_cast<UINT>(start) : 0);
    std::ignore = counts.push(size.NumTiles);
    return {};
}

// The device's sparse memory and tile mapping calls. Each passes its values
// to the D3D12 call that does what it says; the texture and buffer regions
// are built beside the resources they map.

core::Result<D3D12SparseMemory*> D3D12Device::sparseMemoryFor(SparseMemoryHandle memory) {
    D3D12SparseMemory* found = sparseMemories_.find(memory);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the sparse memory handle is stale");
    }
    return found;
}

core::Result<SparseMemoryHandle> D3D12Device::createSparseMemory(const SparseMemoryDesc& desc) {
    if (desc.size == 0) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "sparse memory of 0 bytes");
    }
    if (desc.texture.isNull() && desc.buffer.isNull()) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "sparse memory is made for a texture or buffer; neither was given");
    }
    // Cold, as createBuffer is: what the GPU is done with is freed here.
    collectBuffers();
    // On ResourceHeapTier 1 a heap holds one kind of resource, its
    // D3D12_HEAP_FLAG_ALLOW_ONLY_* flags, and on tier 2 any. A multisampled
    // texture's heap takes the multisample placement alignment
    // (D3D12_HEAP_DESC), as Diligent's DeviceMemoryD3D12Impl.cpp places it.
    D3D12_HEAP_FLAGS kind = D3D12_HEAP_FLAG_NONE;
    bool multisampled = false;
    if (!desc.texture.isNull()) {
        auto texture = textureFor(desc.texture);
        if (!texture) {
            return std::unexpected(std::move(texture.error()));
        }
        if (!(*texture)->sparse) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "the texture is not sparse");
        }
        kind = (*texture)->tileHeap;
        multisampled = (*texture)->samples > 1;
    }
    if (!desc.buffer.isNull()) {
        auto buffer = bufferFor(desc.buffer);
        if (!buffer) {
            return std::unexpected(std::move(buffer.error()));
        }
        if (!(*buffer)->sparse) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "the buffer is not sparse");
        }
        if (!desc.texture.isNull() && !sparse_.mixedResources) {
            return core::failure(core::ErrorCode::RhiUnsupported,
                                 "on ResourceHeapTier 1 a heap holds buffers or textures, not both");
        }
        kind = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
    }
    if (desc.tileBytes != D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES) {
        return core::failure(core::ErrorCode::RhiUnsupported, "D3D12's tiles are {} bytes; tileBytes is {}",
                             D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES, desc.tileBytes);
    }
    const D3D12_HEAP_DESC heapDesc{
        .SizeInBytes = desc.size,
        .Properties = {.Type = D3D12_HEAP_TYPE_DEFAULT,
                       .CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                       .MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN,
                       .CreationNodeMask = 0,
                       .VisibleNodeMask = 0},
        .Alignment = multisampled ? D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT
                                  : D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT,
        .Flags = sparse_.mixedResources ? D3D12_HEAP_FLAG_NONE : kind,
    };
    D3D12SparseMemory made{.heap = {}, .flags = heapDesc.Flags};
    if (const HRESULT result = device_->CreateHeap(&heapDesc, IID_PPV_ARGS(&made.heap)); FAILED(result)) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "ID3D12Device::CreateHeap", .result = result});
    }
    const SparseMemoryHandle memory = sparseMemories_.emplace(std::move(made));
    if (memory.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} sparse memories",
                             desc_.maxSparseMemories);
    }
    return memory;
}

core::Result<void> D3D12Device::destroySparseMemory(SparseMemoryHandle memory) {
    if (memory.isNull()) {
        return {};
    }
    if (sparseMemories_.find(memory) == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the sparse memory handle is stale");
    }
    // Retired against every submit and mapping made so far, as a buffer is.
    const std::uint64_t id = submissions_.mark(
        [this](std::uint32_t queue) { return queues_[queue]->submitted.load(std::memory_order_acquire); });
    if (!sparseMemories_.retire(memory, id)) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the sparse memory handle is stale");
    }
    collectBuffers();
    return {};
}

core::Result<Token> D3D12Device::updateTileMappings(const TileMappingDesc& desc) {
    if (auto checked = checkQueue(desc.queue); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    for (const Token wait : desc.waits) {
        if (auto checked = checkToken(wait); !checked) {
            return std::unexpected(std::move(checked.error()));
        }
    }
    if (desc.textureRegions.size() > desc_.maxTileMappings || desc.bufferRegions.size() > desc_.maxTileMappings) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "a mapping takes at most {} regions of each kind",
                             desc_.maxTileMappings);
    }
    // Every kind of queue maps tiles where the device has tiled resources
    // (sparseOf's Queues).
    if (!sparse_.binding) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device has no tiled resources, so queue {} maps none", desc.queue);
    }
    const D3D12SparseMemory* memory = nullptr;
    if (!desc.memory.isNull()) {
        auto found = sparseMemoryFor(desc.memory);
        if (!found) {
            return std::unexpected(std::move(found.error()));
        }
        memory = *found;
    }
    D3D12Queue& queue = *queues_[desc.queue];
    const std::scoped_lock lock{queue.lock};
    // Both resources' regions are built, and so checked, before the queue is
    // sent anything, so a refused mapping changes nothing.
    auto texture = addTextureTiles(queue.textureTiles, desc, memory);
    if (!texture) {
        return std::unexpected(std::move(texture.error()));
    }
    auto buffer = addBufferTiles(queue.bufferTiles, desc, memory);
    if (!buffer) {
        return std::unexpected(std::move(buffer.error()));
    }
    if (auto waited = waitOn(queue, desc.waits); !waited) {
        return std::unexpected(std::move(waited.error()));
    }
    ID3D12Heap* heap = memory != nullptr ? memory->heap.Get() : nullptr;
    const auto update = [&queue, heap](ID3D12Resource* resource, D3D12TileMappings& tiles) {
        if (resource == nullptr || tiles.coordinates.empty()) {
            return;
        }
        // At most maxTileMappings, a UINT.
        const auto count = static_cast<UINT>(tiles.coordinates.size());
        queue.queue->UpdateTileMappings(resource, count, tiles.coordinates.data(), tiles.sizes.data(), heap, count,
                                        tiles.flags.data(), tiles.starts.data(), tiles.counts.data(),
                                        D3D12_TILE_MAPPING_FLAG_NONE);
    };
    update(*texture, queue.textureTiles);
    update(*buffer, queue.bufferTiles);
    const std::uint64_t value = queue.submitted.load(std::memory_order_relaxed) + 1;
    if (auto signalled = signal(queue, value); !signalled) {
        return std::unexpected(std::move(signalled.error()));
    }
    return Token{.queue = desc.queue, .value = value};
}

} // namespace bfr::rhi::d3d12
