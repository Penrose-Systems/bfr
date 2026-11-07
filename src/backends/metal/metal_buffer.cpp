#include "metal/metal_buffer.h"

#include "bfr/core/error.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/command_list.h"
#include "core/slot_chains.h"
#include "metal/metal_device.h"
#include "metal/metal_texture.h"
#include "metal/metal_util.h"
#include "rhi/bindless_slots.h"
#include "rhi/buffer_check.h"

#include <Metal/Metal.hpp>
#include <Metal/MTL4ComputeCommandEncoder.hpp>
#include <Metal/MTLAllocation.hpp>
#include <Metal/MTLBuffer.hpp>
#include <Metal/MTLGPUAddress.hpp>
#include <Metal/MTLResidencySet.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <mutex>
#include <span>
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
    Ref<MTL::Buffer> buffer = adopt(device.newBuffer(desc.size, optionsFor(desc.memory)));
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
                       .slot = core::SlotChains::kEnd,
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

void MetalDevice::collectBuffers() {
    // The buffers collected leave the residency set, which is not thread-safe.
    const std::scoped_lock lock{residencyLock_};
    const std::uint64_t completed = submissions_.completed(
        [this](std::uint32_t queue) { return queues_[queue]->event->signaledValue(); });
    // A collected resource's slots go back to their tables, so no pending work
    // can see a slot reused.
    buffers_.collect(completed, [this](const MetalBuffer& buffer) { bindless_.giveBuffer(buffer.slot); });
    textures_.collect(completed, [this](const MetalTexture& texture) { bindless_.giveTexture(texture.slots); });
    samplers_.collect(completed, [this](const MetalSampler& sampler) { bindless_.giveSampler(sampler.slot); });
    timestampPools_.collect(completed);
    pipelines_.collect(completed);
    residency_->commit();
}

core::Result<BufferHandle> MetalDevice::createBuffer(const BufferDesc& desc) {
    if (auto checked = checkBufferDesc(desc); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // Cold, so buffers the GPU is done with are freed here, never on the
    // frame path: freeing one frees the allocator's memory.
    collectBuffers();
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
    made->slot = *slot;
    residency_->commit();
    const MTL::GPUAddress address = made->buffer->gpuAddress();
    const BufferHandle buffer = buffers_.emplace(std::move(*made));
    if (buffer.isNull()) {
        bindless_.giveBuffer(*slot);
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} buffers", kBufferLimit);
    }
    if (*slot != core::SlotChains::kEnd) {
        tables().writeStorageBuffer(*slot, address);
    }
    return buffer;
}

core::Result<std::uint32_t> MetalDevice::bufferSlot(BufferHandle buffer) {
    auto found = bufferFor(buffer);
    if (!found) {
        return std::unexpected(std::move(found.error()));
    }
    return BindlessSlots::reported((*found)->slot);
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
    collectBuffers();
    return {};
}

} // namespace bfr::rhi::metal
