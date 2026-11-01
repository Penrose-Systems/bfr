#include "metal/metal_buffer.h"

#include "bfr/core/error.h"
#include "bfr/rhi/buffer.h"
#include "metal/metal_device.h"
#include "metal/metal_util.h"
#include "rhi/buffer_check.h"

#include <Metal/Metal.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <span>
#include <utility>

namespace bfr::rhi::metal {

namespace {

// Device memory is the GPU's alone; the CPU shares upload and readback memory.
MTL::ResourceOptions optionsFor(BufferMemory memory) {
    return memory == BufferMemory::Device ? MTL::ResourceStorageModePrivate : MTL::ResourceStorageModeShared;
}

} // namespace

core::Result<MetalBuffer> createMetalBuffer(MTL::Device& device, const BufferDesc& desc) {
    MetalBuffer made{.buffer = adopt(device.newBuffer(desc.size, optionsFor(desc.memory))),
                     .memory = desc.memory,
                     .mapped = {}};
    if (!made.buffer) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, "Metal gave no buffer of {} bytes", desc.size);
    }
    if (desc.memory != BufferMemory::Device) {
        made.mapped = {static_cast<std::byte*>(made.buffer->contents()), static_cast<std::size_t>(desc.size)};
    }
    return made;
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
    buffers_.collect(
        submissions_.completed([this](std::uint32_t queue) { return queues_[queue]->event->signaledValue(); }));
}

core::Result<BufferHandle> MetalDevice::createBuffer(const BufferDesc& desc) {
    if (auto checked = checkBufferDesc(desc); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // Cold, so buffers the GPU is done with are freed here, never on the
    // frame path: freeing one frees the allocator's memory.
    collectBuffers();
    auto made = createMetalBuffer(*device_.get(), desc);
    if (!made) {
        return std::unexpected(std::move(made.error()));
    }
    const BufferHandle buffer = buffers_.emplace(std::move(*made));
    if (buffer.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached, "the device holds at most {} buffers", kBufferLimit);
    }
    return buffer;
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
