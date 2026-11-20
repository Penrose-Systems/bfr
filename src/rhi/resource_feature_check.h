#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/texture.h"

#include <algorithm>
#include <array>
#include <cstdint>

namespace bfr::rhi {

/// Fails with RhiUnsupported for a sparse buffer the device's Sparse caps
/// lack: one at all (binding and buffers), or an aliased one (aliased).
/// Vulkan's rules (VUID-VkBufferCreateInfo-flags-00915, -00916, -00917),
/// which those caps name on every backend. Each backend's createBuffer calls
/// it after checkBufferDesc.
[[nodiscard]] inline core::Result<void> checkSparseBuffer(const BufferDesc& desc, const Sparse& sparse) {
    if (!desc.sparse) {
        return {};
    }
    if (!sparse.binding || !sparse.buffers) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device has no sparse buffers (Sparse::binding, Sparse::buffers)");
    }
    if (desc.sparseAliased && !sparse.aliased) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device aliases no sparse resource (Sparse::aliased)");
    }
    return {};
}

/// Whether sparse has sparse textures of samples: true for one sample, else
/// the samplesN cap of its count.
[[nodiscard]] inline bool hasSparseSamples(std::uint32_t samples, const Sparse& sparse) {
    constexpr std::uint32_t kTwo = 2;
    constexpr std::uint32_t kFour = 4;
    constexpr std::uint32_t kEight = 8;
    constexpr std::uint32_t kSixteen = 16;
    switch (samples) {
        case kTwo:
            return sparse.samples2;
        case kFour:
            return sparse.samples4;
        case kEight:
            return sparse.samples8;
        case kSixteen:
            return sparse.samples16;
        default:
            return true;
    }
}

/// The ASTC HDR formats, which Features::compressionAstcHdr gates.
inline constexpr std::array kAstcHdrFormats{
    Format::Astc4x4Float,   Format::Astc5x4Float,   Format::Astc5x5Float,  Format::Astc6x5Float,
    Format::Astc6x6Float,   Format::Astc8x5Float,   Format::Astc8x6Float,  Format::Astc8x8Float,
    Format::Astc10x5Float,  Format::Astc10x6Float,  Format::Astc10x8Float, Format::Astc10x10Float,
    Format::Astc12x10Float, Format::Astc12x12Float,
};

/// Fails with RhiUnsupported for a texture the device's features lack: one
/// of format Bgra4Unorm (Features::formatBgra4Unorm) or an ASTC HDR format
/// (Features::compressionAstcHdr), whose VkFormat values below Vulkan 1.3
/// are an extension's, valid only with it enabled
/// (VUID-VkImageCreateInfo-format-parameter); a cube texture of more than one cube, whose views are cube arrays
/// (Features::imageCubeArray, VUID-VkImageViewCreateInfo-viewType-01004); or
/// a sparse texture where Sparse lacks binding, its dimension (textures2D for
/// a 2D or cube texture, textures3D for 3D, none for 1D), its sample count,
/// or aliasing (VUID-VkImageCreateInfo-flags-00969, -01924; -imageType-00970
/// to -00976). Those caps name Vulkan's rules on every backend. Each
/// backend's createTexture calls it after checkEnums.
[[nodiscard]] inline core::Result<void> checkTextureFeatures(const TextureDesc& desc, const Features& features,
                                                             const Sparse& sparse) {
    if (desc.format == Format::Bgra4Unorm && !features.formatBgra4Unorm) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device has no bgra4_unorm textures (Features::formatBgra4Unorm)");
    }
    if (std::ranges::contains(kAstcHdrFormats, desc.format) && !features.compressionAstcHdr) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device has no ASTC HDR textures (Features::compressionAstcHdr)");
    }
    if (desc.dimension == TextureDimension::Cube && desc.layers > kCubeFaces && !features.imageCubeArray) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device has no cube arrays, and {} layers are more than one cube "
                             "(Features::imageCubeArray)",
                             desc.layers);
    }
    if (!desc.sparse) {
        return {};
    }
    const bool dimension = desc.dimension == TextureDimension::D3
                               ? sparse.textures3D
                               : desc.dimension != TextureDimension::D1 && sparse.textures2D;
    if (!sparse.binding || !dimension || !hasSparseSamples(desc.samples, sparse)) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device has no sparse texture of this dimension with {} samples (Sparse)",
                             desc.samples);
    }
    if (desc.sparseAliased && !sparse.aliased) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device aliases no sparse resource (Sparse::aliased)");
    }
    return {};
}

} // namespace bfr::rhi
