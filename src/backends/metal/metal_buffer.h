#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/buffer.h"
#include "metal/metal_util.h"

#include <Metal/Metal.hpp>

#include <cstddef>
#include <span>

namespace bfr::rhi::metal {

/// A Metal buffer, released when the last Ref to it goes. mapped is where the
/// CPU reaches upload and readback memory, empty for device memory.
struct MetalBuffer {
    Ref<MTL::Buffer> buffer;
    BufferMemory memory = BufferMemory::Device;
    std::span<std::byte> mapped;
};

/// A buffer on device as desc says. Fails with RhiDeviceFailed when Metal
/// gives no buffer.
[[nodiscard]] core::Result<MetalBuffer> createMetalBuffer(MTL::Device& device, const BufferDesc& desc);

} // namespace bfr::rhi::metal
