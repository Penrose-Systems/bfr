#include "metal/metal_buffer.h"

#include "bfr/core/error.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/sparse.h"
#include "core/slot_chains.h"
#include "metal/metal_device.h"
#include "metal/metal_sparse.h"
#include "metal/metal_texture.h"
#include "metal/metal_util.h"
#include "rhi/bindless_slots.h"
#include "rhi/buffer_check.h"
#include "rhi/enum_check.h"

#include <Metal/Metal.hpp>
#include <Metal/MTL4CommandQueue.hpp>
#include <Metal/MTL4ComputeCommandEncoder.hpp>
#include <Metal/MTLAllocation.hpp>
#include <Metal/MTLBuffer.hpp>
#include <Metal/MTLGPUAddress.hpp>
#include <Metal/MTLResidencySet.hpp>
#include <Metal/MTLResource.hpp>
#include <Metal/MTLResourceStateCommandEncoder.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <mutex>
#include <optional>
#include <span>
#include <tuple>
#include <utility>

namespace bfr::rhi::metal {

namespace {

// Device memory is the GPU's alone; the CPU shares upload and readback memory,
// write-combined for upload, which the CPU only writes, and cached for
// readback, which it reads.
MTL::ResourceOptions optionsFor(BufferMemory memory) {
    switch (memory) {
        case BufferMemory::Device:
            return MTL::ResourceStorageModePrivate;
        case BufferMemory::Upload:
            return MTL::ResourceStorageModeShared | MTL::ResourceCPUCacheModeWriteCombined;
        case BufferMemory::Readback:
            return MTL::ResourceStorageModeShared | MTL::ResourceCPUCacheModeDefaultCache;
    }
    return MTL::ResourceStorageModeShared;
}

} // namespace

Resident::Resident(Ref<MTL::ResidencySet> set, const MTL::Allocation* allocation)
    : set_(std::move(set)), allocation_(allocation) {
    set_->addAllocation(allocation_);
}

Resident::~Resident() {
    if (allocation_ != nullptr) {
        set_->removeAllocation(allocation_);
    }
}

Resident::Resident(Resident&& other) noexcept
    : set_(std::move(other.set_)), allocation_(std::exchange(other.allocation_, nullptr)) {}

core::Result<MetalBuffer> createMetalBuffer(MTL::Device& device, const BufferDesc& desc,
                                            Ref<MTL::ResidencySet> residency) {
    std::optional<MTL::SparsePageSize> page;
    std::uint64_t tileBytes = 0;
    if (desc.sparse) {
        if (desc.memory != BufferMemory::Device) {
            return core::failure(core::ErrorCode::RhiInvalidCall,
                                 "a sparse buffer's memory is the sparse memory mapped into it, so it is device");
        }
        // MTLBuffer.hpp and MTLDevice.hpp:408 have no aliased sparse buffer.
        if (desc.sparseAliased) {
            return core::failure(core::ErrorCode::RhiUnsupported, "Metal has no aliased sparse buffer");
        }
        auto found = pageSizeFor(device, desc.tileBytes);
        if (!found) {
            return std::unexpected(std::move(found.error()));
        }
        page = *found;
        tileBytes = device.sparseTileSizeInBytes(*page);
    }
    // A sparse buffer is made at its page size, with no memory of its own.
    Ref<MTL::Buffer> buffer = adopt(page ? device.newBuffer(desc.size, optionsFor(desc.memory), *page)
                                         : device.newBuffer(desc.size, optionsFor(desc.memory)));
    if (!buffer) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, "Metal gave no buffer of {} bytes", desc.size);
    }
    std::span<std::byte> mapped;
    if (desc.memory != BufferMemory::Device) {
        mapped = {static_cast<std::byte*>(buffer->contents()), static_cast<std::size_t>(desc.size)};
    }
    Resident resident{std::move(residency), buffer.get()};
    return MetalBuffer{.buffer = std::move(buffer),
                       .memory = desc.memory,
                       .mapped = mapped,
                       .slots = {},
                       .page = page,
                       .tileBytes = tileBytes,
                       .resident = std::move(resident)};
}

// The device's buffer calls, beside the buffers they make.

core::Result<MetalBuffer*> MetalDevice::bufferFor(BufferHandle buffer) {
    MetalBuffer* found = buffers_.find(buffer);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the buffer handle is stale");
    }
    return found;
}

void MetalDevice::collectRetired() {
    // The buffers collected leave the residency set, which is not thread-safe.
    const std::scoped_lock lock{residencyLock_};
    const std::uint64_t completed = submissions_.completed(
        [this](std::uint32_t queue) { return queues_[queue]->event->signaledValue(); });
    // A collected resource's slots go back to their tables, so no pending work
    // can see a slot reused.
    buffers_.collect(completed, [this](const MetalBuffer& buffer) { bindless_.giveBuffer(buffer.slots); });
    textures_.collect(completed, [this](const MetalTexture& texture) { bindless_.giveTexture(texture.slots); });
    samplers_.collect(completed, [this](const MetalSampler& sampler) { bindless_.giveSampler(sampler.slot); });
    timestampPools_.collect(completed);
    sparseMemories_.collect(completed);
    pipelines_.collect(completed);
    residency_->commit();
}

core::Result<void> MetalDevice::runGarbageCollection() {
    collectRetired();
    return {};
}

core::Result<BufferHandle> MetalDevice::createBuffer(const BufferDesc& desc) {
    if (auto checked = checkBufferDesc(desc); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // Cold, so buffers the GPU is done with are freed here, never on the
    // frame path: freeing one frees the allocator's memory.
    collectRetired();
    // Held until made is gone, which takes it out of the set if it is not kept.
    const std::scoped_lock lock{residencyLock_};
    auto slot = bindless_.takeBuffer(desc);
    if (!slot) {
        return std::unexpected(std::move(slot.error()));
    }
    auto made = createMetalBuffer(*device_.get(), desc, residency_);
    if (!made) {
        bindless_.giveBuffer(*slot);
        return std::unexpected(std::move(made.error()));
    }
    made->slots = *slot;
    residency_->commit();
    const MTL::GPUAddress address = made->buffer->gpuAddress();
    const BufferHandle buffer = buffers_.emplace(std::move(*made));
    if (buffer.isNull()) {
        bindless_.giveBuffer(*slot);
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} buffers", desc_.maxBuffers);
    }
    if (slot->storage != core::SlotChains::kEnd) {
        tables().writeStorageBuffer(slot->storage, address);
    }
    return buffer;
}

core::Result<BufferSlots> MetalDevice::bufferSlots(BufferHandle buffer) {
    auto found = bufferFor(buffer);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    return BindlessSlots::reported((*found)->slots);
}

core::Result<void> MetalDevice::copyBuffer(CommandListHandle commandList, BufferHandle src, BufferHandle dst,
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
    // Metal 4 copies on a compute encoder. The pool drains the autoreleased
    // encoder reference.
    const AutoreleasePool pool;
    MTL4::ComputeCommandEncoder* encoder = (*list)->buffer->computeCommandEncoder();
    if (encoder == nullptr) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, "Metal gave no compute command encoder");
    }
    for (const BufferCopy& region : regions) {
        encoder->copyFromBuffer((*from)->buffer.get(), region.srcOffset, (*to)->buffer.get(), region.dstOffset,
                                region.size);
    }
    encoder->endEncoding();
    return {};
}

core::Result<std::span<std::byte>> MetalDevice::mapBuffer(BufferHandle buffer) {
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

core::Result<void> MetalDevice::destroyBuffer(BufferHandle buffer) {
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

core::Result<BufferTiling> MetalDevice::bufferTiling(BufferHandle buffer) {
    auto found = bufferFor(buffer);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    if (!(*found)->page) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "the buffer is not sparse");
    }
    return BufferTiling{.tileBytes = (*found)->tileBytes};
}

// Ranges and heap offsets in bytes: tiles times the buffer's tile bytes. The
// header (MTL4CommandQueue.hpp:82-87) names no unit, so bytes are unverified.
core::Result<const MTL::Buffer*> MetalDevice::addBufferMappings(MetalQueue& queue, const TileMappingDesc& desc,
                                                                const MetalSparseMemory* memory) {
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
    const MetalBuffer& buffer = **found;
    if (memory != nullptr) {
        if (auto fits = checkPageFits(*buffer.page, memory->page); !fits) {
            return std::unexpected(std::move(fits.error()));
        }
    }
    for (const BufferTileRegion& region : desc.bufferRegions) {
        if (auto checked = checkKnown(region.mapping, "tile mapping"); !checked) {
            return std::unexpected(std::move(checked.error()));
        }
        const bool maps = region.mapping == TileMapping::Map;
        if (maps && memory == nullptr) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "a region maps, but no memory was given");
        }
        // The buffer's regions are at most maxTileMappings, which the array holds.
        std::ignore = queue.bufferMappings.push(MTL4::UpdateSparseBufferMappingOperation{
            .mode = maps ? MTL::SparseTextureMappingModeMap : MTL::SparseTextureMappingModeUnmap,
            .bufferRange = NS::Range::Make(region.tile * buffer.tileBytes, region.tiles * buffer.tileBytes),
            .heapOffset = maps ? region.memoryTile * buffer.tileBytes : 0});
    }
    return buffer.buffer.get();
}

} // namespace bfr::rhi::metal
