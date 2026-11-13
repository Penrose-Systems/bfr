#include "metal/metal_sparse.h"

#include "bfr/core/error.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/sparse.h"
#include "metal/metal_buffer.h"
#include "metal/metal_device.h"
#include "metal/metal_texture.h"
#include "metal/metal_util.h"

#include <Metal/MTL4CommandQueue.hpp>
#include <Metal/MTLDevice.hpp>
#include <Metal/MTLEvent.hpp>
#include <Metal/MTLHeap.hpp>
#include <Metal/MTLResidencySet.hpp>
#include <Metal/MTLResource.hpp>

#include <atomic>
#include <cstdint>
#include <expected>
#include <mutex>
#include <utility>

namespace bfr::rhi::metal {

core::Result<MTL::SparsePageSize> pageSizeFor(MTL::Device& device, std::uint32_t tileBytes) {
    for (const MTL::SparsePageSize page : kSparsePageSizes) {
        if (tileBytes != 0 && device.sparseTileSizeInBytes(page) == tileBytes) {
            return page;
        }
    }
    return core::failure(core::ErrorCode::RhiUnsupported, "Metal has no sparse page of {} bytes", tileBytes);
}

core::Result<void> checkPageFits(MTL::SparsePageSize page, MTL::SparsePageSize heapPage) {
    if (page > heapPage) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the resource's sparse page is larger than any the memory's heap takes");
    }
    return {};
}

// The device's sparse memory and tile mapping calls. Each passes its values
// to the Metal call that does what it says; the texture and buffer ops are
// built beside the resources they map.

core::Result<MetalSparseMemory*> MetalDevice::sparseMemoryFor(SparseMemoryHandle memory) {
    MetalSparseMemory* found = sparseMemories_.find(memory);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the sparse memory handle is stale");
    }
    return found;
}

core::Result<SparseMemoryHandle> MetalDevice::createSparseMemory(const SparseMemoryDesc& desc) {
    if (desc.size == 0) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "sparse memory of 0 bytes");
    }
    if (desc.texture.isNull() && desc.buffer.isNull()) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "sparse memory names no texture or buffer");
    }
    auto page = pageSizeFor(*device_.get(), desc.tileBytes);
    if (!page) {
        return std::unexpected(std::move(page.error()));
    }
    // Cold, as createBuffer is: what the GPU is done with is freed here.
    collectRetired();
    if (!desc.texture.isNull()) {
        auto texture = textureFor(desc.texture);
        if (!texture) {
            return std::unexpected(std::move(texture.error()));
        }
        if (auto fits = checkPageFits(*(*texture)->page, *page); !fits) {
            return std::unexpected(std::move(fits.error()));
        }
    }
    if (!desc.buffer.isNull()) {
        auto buffer = bufferFor(desc.buffer);
        if (!buffer) {
            return std::unexpected(std::move(buffer.error()));
        }
        if (auto fits = checkPageFits(*(*buffer)->page, *page); !fits) {
            return std::unexpected(std::move(fits.error()));
        }
    }
    // A placement heap whose tiles sparse resources of the page size map
    // (MTLHeap.hpp:40-81). Which heap type Metal 4's mapping ops take is not
    // stated locally: Placement is unverified on hardware.
    const Ref<MTL::HeapDescriptor> descriptor = adopt(MTL::HeapDescriptor::alloc()->init());
    descriptor->setType(MTL::HeapTypePlacement);
    descriptor->setSize(desc.size);
    descriptor->setStorageMode(MTL::StorageModePrivate);
    descriptor->setMaxCompatiblePlacementSparsePageSize(*page);
    // Held until made is gone, which takes it out of the set if it is not kept.
    const std::scoped_lock lock{residencyLock_};
    Ref<MTL::Heap> heap = adopt(device_->newHeap(descriptor.get()));
    if (!heap) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, "Metal gave no heap of {} bytes", desc.size);
    }
    Resident resident{residency_, heap.get()};
    residency_->commit();
    const SparseMemoryHandle memory = sparseMemories_.emplace(
        MetalSparseMemory{.heap = std::move(heap), .page = *page, .resident = std::move(resident)});
    if (memory.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} sparse memories",
                             desc_.maxSparseMemories);
    }
    return memory;
}

core::Result<void> MetalDevice::destroySparseMemory(SparseMemoryHandle memory) {
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
    collectRetired();
    return {};
}

// Metal 4 queues the waits as they are made, so every check is done and every
// op built before the first.
core::Result<Token> MetalDevice::updateTileMappings(const TileMappingDesc& desc) {
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
    // Null when every region unmaps: metal-cpp's const Heap*
    // (MTL4CommandQueue.hpp:144-146) drops Metal's nullability, so whether
    // Metal takes no heap to unmap is unverified.
    const MetalSparseMemory* memory = nullptr;
    if (!desc.memory.isNull()) {
        auto found = sparseMemoryFor(desc.memory);
        if (!found) {
            return std::unexpected(std::move(found.error()));
        }
        memory = *found;
    }
    const MTL::Heap* heap = memory != nullptr ? memory->heap.get() : nullptr;
    MetalQueue& queue = *queues_[desc.queue];
    const std::scoped_lock lock{queue.lock};
    queue.textureMappings.clear();
    queue.bufferMappings.clear();
    auto texture = addTextureMappings(queue, desc, memory);
    if (!texture) {
        return std::unexpected(std::move(texture.error()));
    }
    auto buffer = addBufferMappings(queue, desc, memory);
    if (!buffer) {
        return std::unexpected(std::move(buffer.error()));
    }
    // As submit waits.
    for (const Token wait : desc.waits) {
        queue.queue->wait(queues_[wait.queue]->event.get(), wait.value);
    }
    if (!queue.textureMappings.empty()) {
        queue.queue->updateTextureMappings(*texture, heap, queue.textureMappings.data(), queue.textureMappings.size());
    }
    if (!queue.bufferMappings.empty()) {
        queue.queue->updateBufferMappings(*buffer, heap, queue.bufferMappings.data(), queue.bufferMappings.size());
    }
    const std::uint64_t value = queue.submitted.load(std::memory_order_relaxed) + 1;
    queue.queue->signalEvent(queue.event.get(), value);
    queue.submitted.store(value, std::memory_order_release);
    return Token{.queue = desc.queue, .value = value};
}

} // namespace bfr::rhi::metal
