#include "d3d12/d3d12_buffer.h"

#include "bfr/core/error.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/sparse.h"
#include "core/slot_chains.h"
#include "d3d12/d3d12_device.h"
#include "d3d12/d3d12_result.h"
#include "d3d12/d3d12_sparse.h"
#include "d3d12/d3d12_texture.h"
#include "rhi/bindless_slots.h"
#include "rhi/buffer_check.h"
#include "rhi/enum_check.h"

#include <d3d12.h>
#include <D3D12MemAlloc.h>
#include <dxgi1_6.h>
#include <dxgiformat.h>
#include <windows.h>
#include <wrl/client.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <span>
#include <utility>

namespace bfr::rhi::d3d12 {

namespace {

using Microsoft::WRL::ComPtr;

// The heap each BufferMemory lives in, and the state D3D12 requires a buffer
// in that heap to be created in.
struct Placement {
    D3D12_HEAP_TYPE heap;
    D3D12_RESOURCE_STATES state;
};

Placement placementFor(BufferMemory memory) {
    switch (memory) {
        case BufferMemory::Device:
            return {.heap = D3D12_HEAP_TYPE_DEFAULT, .state = D3D12_RESOURCE_STATE_COMMON};
        case BufferMemory::Upload:
            return {.heap = D3D12_HEAP_TYPE_UPLOAD, .state = D3D12_RESOURCE_STATE_GENERIC_READ};
        case BufferMemory::Readback:
            return {.heap = D3D12_HEAP_TYPE_READBACK, .state = D3D12_RESOURCE_STATE_COPY_DEST};
    }
    return {.heap = D3D12_HEAP_TYPE_DEFAULT, .state = D3D12_RESOURCE_STATE_COMMON};
}

} // namespace

core::Result<ComPtr<D3D12MA::Allocator>> createAllocator(ID3D12Device* device, IDXGIAdapter* adapter) {
    D3D12MA::ALLOCATOR_DESC desc{};
    desc.pDevice = device;
    desc.pAdapter = adapter;
    ComPtr<D3D12MA::Allocator> allocator;
    if (const HRESULT result = D3D12MA::CreateAllocator(&desc, &allocator); FAILED(result)) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "D3D12MA::CreateAllocator", .result = result});
    }
    return allocator;
}

core::Result<D3D12Buffer> createD3D12Buffer(D3D12MA::Allocator& allocator, ID3D12Device& device, const BufferDesc& desc,
                                            const BufferSlotsHeld& slots) {
    if (desc.sparse && desc.memory != BufferMemory::Device) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "a sparse buffer's memory is the heaps mapped into it, so it is device");
    }
    if (desc.sparse && desc.tileBytes != D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES) {
        return core::failure(core::ErrorCode::RhiUnsupported, "D3D12's tiles are {} bytes; tileBytes is {}",
                             D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES, desc.tileBytes);
    }
    const Placement placement = placementFor(desc.memory);
    const D3D12_RESOURCE_DESC resource{
        .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
        .Alignment = 0,
        .Width = desc.size,
        .Height = 1,
        .DepthOrArraySize = 1,
        .MipLevels = 1,
        .Format = DXGI_FORMAT_UNKNOWN,
        .SampleDesc = {.Count = 1, .Quality = 0},
        .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        .Flags = desc.usage.storage ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE,
    };
    D3D12Buffer buffer{.allocation = {},
                       .resource = {},
                       .memory = desc.memory,
                       .size = desc.size,
                       .mapped = {},
                       .slots = slots,
                       .sparse = desc.sparse};
    if (desc.sparse) {
        // Heaps' tiles are mapped into it, so no memory is placed now.
        // sparseAliased needs no D3D12 flag: any reserved resource's tiles may
        // be mapped to heap tiles others are mapped to.
        if (const HRESULT result = device.CreateReservedResource(&resource, placement.state, nullptr,
                                                                 IID_PPV_ARGS(&buffer.resource));
            FAILED(result)) {
            return failure(core::ErrorCode::RhiDeviceFailed,
                           {.call = "ID3D12Device::CreateReservedResource", .result = result});
        }
        return buffer;
    }
    D3D12MA::ALLOCATION_DESC allocation{};
    allocation.HeapType = placement.heap;
    if (const HRESULT result = allocator.CreateResource(&allocation, &resource, placement.state, nullptr,
                                                        &buffer.allocation, IID_PPV_ARGS(&buffer.resource));
        FAILED(result)) {
        return failure(core::ErrorCode::RhiDeviceFailed,
                       {.call = "D3D12MA::Allocator::CreateResource", .result = result});
    }
    if (desc.memory == BufferMemory::Device) {
        return buffer;
    }
    // The CPU reads none of an upload buffer, and all of a readback one.
    const D3D12_RANGE none{.Begin = 0, .End = 0};
    void* data = nullptr;
    if (const HRESULT result = buffer.resource->Map(0, desc.memory == BufferMemory::Upload ? &none : nullptr, &data);
        FAILED(result)) {
        return failure(core::ErrorCode::RhiDeviceFailed, {.call = "ID3D12Resource::Map", .result = result});
    }
    buffer.mapped = {static_cast<std::byte*>(data), static_cast<std::size_t>(desc.size)};
    return buffer;
}

// The device's buffer calls, beside the buffers they make.

core::Result<D3D12Buffer*> D3D12Device::bufferFor(BufferHandle buffer) {
    D3D12Buffer* found = buffers_.find(buffer);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the buffer handle is stale");
    }
    return found;
}

void D3D12Device::collectRetired() {
    const std::uint64_t completed = submissions_.completed(
        [this](std::uint32_t queue) { return queues_[queue]->fence->GetCompletedValue(); });
    // A collected resource's slots go back to their tables, so no pending work
    // can see a slot reused.
    buffers_.collect(completed, [this](const D3D12Buffer& buffer) { bindless_.giveBuffer(buffer.slots); });
    textures_.collect(completed, [this](const D3D12Texture& texture) { bindless_.giveTexture(texture.slots); });
    samplers_.collect(completed, [this](const D3D12Sampler& sampler) { bindless_.giveSampler(sampler.slot); });
    timestampPools_.collect(completed);
    sparseMemories_.collect(completed);
    pipelines_.collect(completed);
}

core::Result<void> D3D12Device::runGarbageCollection() {
    collectRetired();
    return {};
}

core::Result<BufferHandle> D3D12Device::createBuffer(const BufferDesc& desc) {
    if (auto checked = checkBufferDesc(desc); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // A storage buffer's raw view counts its 32-bit values in a UINT, so bytes
    // past the last whole value, or values past a UINT, are out of its view.
    if (desc.usage.storage && desc.size % sizeof(std::uint32_t) != 0) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "a storage buffer of {} bytes is not whole 32-bit values, which D3D12 views", desc.size);
    }
    if (desc.usage.storage && !std::in_range<UINT>(desc.size / sizeof(std::uint32_t))) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "a storage buffer of {} bytes is more than D3D12 views",
                             desc.size);
    }
    // Cold, so buffers the GPU is done with are freed here, never on the
    // frame path: freeing one frees the allocator's memory.
    collectRetired();
    auto slot = bindless_.takeBuffer(desc);
    if (!slot) {
        return std::unexpected(std::move(slot.error()));
    }
    auto made = createD3D12Buffer(*allocator_.Get(), *device_.Get(), desc, *slot);
    if (!made) {
        bindless_.giveBuffer(*slot);
        return std::unexpected(std::move(made.error()));
    }
    ID3D12Resource* resource = made->resource.Get();
    const BufferHandle buffer = buffers_.emplace(std::move(*made));
    if (buffer.isNull()) {
        bindless_.giveBuffer(*slot);
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} buffers", desc_.maxBuffers);
    }
    if (slot->storage != core::SlotChains::kEnd) {
        // A raw view of the whole buffer, as api/bfr.yaml states under
        // bfr_pipeline: the shader declares a ByteAddressBuffer.
        const D3D12_UNORDERED_ACCESS_VIEW_DESC view{.Format = DXGI_FORMAT_R32_TYPELESS,
                                                    .ViewDimension = D3D12_UAV_DIMENSION_BUFFER,
                                                    .Buffer = {.FirstElement = 0,
                                                               .NumElements = static_cast<UINT>(desc.size /
                                                                                                sizeof(std::uint32_t)),
                                                               .StructureByteStride = 0,
                                                               .CounterOffsetInBytes = 0,
                                                               .Flags = D3D12_BUFFER_UAV_FLAG_RAW}};
        device_->CreateUnorderedAccessView(resource, nullptr, &view,
                                           layout_.resourceAt(layout_.storageBuffers + slot->storage));
    }
    return buffer;
}

core::Result<BufferSlots> D3D12Device::bufferSlots(BufferHandle buffer) {
    auto found = bufferFor(buffer);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    return BindlessSlots::reported((*found)->slots);
}

core::Result<std::span<std::byte>> D3D12Device::mapBuffer(BufferHandle buffer) {
    auto found = bufferFor(buffer);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    if ((*found)->memory == BufferMemory::Device) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "the buffer is in device memory, which the CPU does not map");
    }
    return (*found)->mapped;
}

core::Result<void> D3D12Device::copyBuffer(CommandListHandle commandList, BufferHandle src, BufferHandle dst,
                                           std::span<const BufferCopy> regions) {
    auto list = commandListFor(commandList);
    if (!list) {
        return std::unexpected(std::move(list.error()));
    }
    auto from = bufferFor(src);
    if (!from) {
        return std::unexpected(std::move(from.error()));
    }
    auto to = bufferFor(dst);
    if (!to) {
        return std::unexpected(std::move(to.error()));
    }
    for (const BufferCopy& region : regions) {
        (*list)->list->CopyBufferRegion((*to)->resource.Get(), region.dstOffset, (*from)->resource.Get(),
                                        region.srcOffset, region.size);
    }
    return {};
}

core::Result<void> D3D12Device::destroyBuffer(BufferHandle buffer) {
    if (buffer.isNull()) {
        return {};
    }
    if (buffers_.find(buffer) == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the buffer handle is stale");
    }
    // Retired against every submit made so far, on every queue, so it lives
    // until the GPU has run all of them.
    const std::uint64_t id = submissions_.mark(
        [this](std::uint32_t queue) { return queues_[queue]->submitted.load(std::memory_order_acquire); });
    if (!buffers_.retire(buffer, id)) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the buffer handle is stale");
    }
    collectRetired();
    return {};
}

// A buffer's tiles are D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES (d3d12.h),
// the one size D3D12 has.
core::Result<BufferTiling> D3D12Device::bufferTiling(BufferHandle buffer) {
    auto found = bufferFor(buffer);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    if (!(*found)->sparse) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "the buffer is not sparse");
    }
    return BufferTiling{.tileBytes = D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES};
}

core::Result<ID3D12Resource*> D3D12Device::addBufferTiles(D3D12TileMappings& tiles, const TileMappingDesc& desc,
                                                          const D3D12SparseMemory* memory) {
    tiles.clear();
    if (desc.buffer.isNull()) {
        if (!desc.bufferRegions.empty()) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "buffer regions were given with no buffer");
        }
        return nullptr;
    }
    auto found = bufferFor(desc.buffer);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    if (memory != nullptr && !memory->takes(D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS)) {
        return core::failure(
            core::ErrorCode::RhiUnsupported,
            "on ResourceHeapTier 1 a heap holds one kind of resource, and the memory's is not buffers");
    }
    for (const BufferTileRegion& region : desc.bufferRegions) {
        if (auto checked = checkKnown(region.mapping, "tile mapping"); !checked) {
            return std::unexpected(std::move(checked.error()));
        }
        const bool maps = region.mapping == TileMapping::Map;
        if (maps && memory == nullptr) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "a region maps, but no memory was given");
        }
        if (!std::in_range<UINT>(region.tile) || !std::in_range<UINT>(region.tiles)) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "{} tiles from tile {} are past what D3D12 takes",
                                 region.tiles, region.tile);
        }
        // A buffer is one subresource, its tiles counted along X.
        const D3D12_TILED_RESOURCE_COORDINATE coordinate{.X = static_cast<UINT>(region.tile),
                                                         .Y = 0,
                                                         .Z = 0,
                                                         .Subresource = 0};
        const D3D12_TILE_REGION_SIZE size{
            .NumTiles = static_cast<UINT>(region.tiles), .UseBox = FALSE, .Width = 0, .Height = 0, .Depth = 0};
        if (auto added = tiles.add(coordinate, size, maps, region.memoryTile); !added) {
            return std::unexpected(std::move(added.error()));
        }
    }
    return (*found)->resource.Get();
}

} // namespace bfr::rhi::d3d12
