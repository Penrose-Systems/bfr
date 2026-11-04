#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/texture.h"
#include "metal/metal_buffer.h"
#include "metal/metal_util.h"
#include "rhi/bindless_slots.h"

#include <Metal/Metal.hpp>

namespace bfr::rhi::metal {

/// A Metal texture, released when the last Ref to it goes, and resident while
/// it lives, as a buffer is.
struct MetalTexture {
    Ref<MTL::Texture> texture;
    Format format = Format::R8Unorm;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    /// Its places in the tables, which the device gives back when the texture
    /// is collected.
    TextureSlotsHeld slots;
    // Declared after texture, so it leaves the set before it is released.
    Resident resident;
};

/// A texture on device as desc says, in private storage, added to residency,
/// which the caller commits under its lock. Fails with RhiDeviceFailed when
/// Metal gives no texture.
[[nodiscard]] core::Result<MetalTexture> createMetalTexture(MTL::Device& device, const TextureDesc& desc,
                                                            Ref<MTL::ResidencySet> residency);

} // namespace bfr::rhi::metal
