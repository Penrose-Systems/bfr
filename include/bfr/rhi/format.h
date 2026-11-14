#pragma once

#include "bfr/bfr_enums.hpp" // IWYU pragma: export

#include <array>
#include <cstdint>

namespace bfr::rhi {

/// Every format but Undefined, in the order Format declares them.
inline constexpr std::array kFormats{
    Format::R8Unorm,        Format::R8Snorm,        Format::R8Uint,         Format::R8Sint,
    Format::Rg8Unorm,       Format::Rg8Snorm,       Format::Rg8Uint,        Format::Rg8Sint,
    Format::Rgba8Unorm,     Format::Rgba8Srgb,      Format::Rgba8Snorm,     Format::Rgba8Uint,
    Format::Rgba8Sint,      Format::Bgra8Unorm,     Format::Bgra8Srgb,      Format::R16Unorm,
    Format::R16Snorm,       Format::R16Uint,        Format::R16Sint,        Format::R16Float,
    Format::Rg16Unorm,      Format::Rg16Snorm,      Format::Rg16Uint,       Format::Rg16Sint,
    Format::Rg16Float,      Format::Rgba16Unorm,    Format::Rgba16Snorm,    Format::Rgba16Uint,
    Format::Rgba16Sint,     Format::Rgba16Float,    Format::R32Uint,        Format::R32Sint,
    Format::R32Float,       Format::Rg32Uint,       Format::Rg32Sint,       Format::Rg32Float,
    Format::Rgb32Uint,      Format::Rgb32Sint,      Format::Rgb32Float,     Format::Rgba32Uint,
    Format::Rgba32Sint,     Format::Rgba32Float,    Format::Rgb10a2Unorm,   Format::Rgb10a2Uint,
    Format::Bgr10a2Unorm,   Format::Rg11b10Float,   Format::Rgb9e5Float,    Format::B5g6r5Unorm,
    Format::Bgr5a1Unorm,    Format::Bgra4Unorm,     Format::R64Uint,        Format::R64Sint,
    Format::R64Float,       Format::Rg64Uint,       Format::Rg64Sint,       Format::Rg64Float,
    Format::Rgb64Uint,      Format::Rgb64Sint,      Format::Rgb64Float,     Format::Rgba64Uint,
    Format::Rgba64Sint,     Format::Rgba64Float,    Format::D16Unorm,       Format::D24UnormS8Uint,
    Format::D32Float,       Format::D32FloatS8Uint, Format::S8Uint,         Format::Bc1Unorm,
    Format::Bc1Srgb,        Format::Bc2Unorm,       Format::Bc2Srgb,        Format::Bc3Unorm,
    Format::Bc3Srgb,        Format::Bc4Unorm,       Format::Bc4Snorm,       Format::Bc5Unorm,
    Format::Bc5Snorm,       Format::Bc6hUfloat,     Format::Bc6hSfloat,     Format::Bc7Unorm,
    Format::Bc7Srgb,        Format::Etc2Rgb8Unorm,  Format::Etc2Rgb8Srgb,   Format::Etc2Rgb8a1Unorm,
    Format::Etc2Rgb8a1Srgb, Format::Etc2Rgba8Unorm, Format::Etc2Rgba8Srgb,  Format::EacR11Unorm,
    Format::EacR11Snorm,    Format::EacRg11Unorm,   Format::EacRg11Snorm,   Format::Astc4x4Unorm,
    Format::Astc4x4Srgb,    Format::Astc4x4Float,   Format::Astc5x4Unorm,   Format::Astc5x4Srgb,
    Format::Astc5x4Float,   Format::Astc5x5Unorm,   Format::Astc5x5Srgb,    Format::Astc5x5Float,
    Format::Astc6x5Unorm,   Format::Astc6x5Srgb,    Format::Astc6x5Float,   Format::Astc6x6Unorm,
    Format::Astc6x6Srgb,    Format::Astc6x6Float,   Format::Astc8x5Unorm,   Format::Astc8x5Srgb,
    Format::Astc8x5Float,   Format::Astc8x6Unorm,   Format::Astc8x6Srgb,    Format::Astc8x6Float,
    Format::Astc8x8Unorm,   Format::Astc8x8Srgb,    Format::Astc8x8Float,   Format::Astc10x5Unorm,
    Format::Astc10x5Srgb,   Format::Astc10x5Float,  Format::Astc10x6Unorm,  Format::Astc10x6Srgb,
    Format::Astc10x6Float,  Format::Astc10x8Unorm,  Format::Astc10x8Srgb,   Format::Astc10x8Float,
    Format::Astc10x10Unorm, Format::Astc10x10Srgb,  Format::Astc10x10Float, Format::Astc12x10Unorm,
    Format::Astc12x10Srgb,  Format::Astc12x10Float, Format::Astc12x12Unorm, Format::Astc12x12Srgb,
    Format::Astc12x12Float,
};

} // namespace bfr::rhi
