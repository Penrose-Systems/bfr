#include "metal/metal_formats.h"

#include "bfr/rhi/caps.h"
#include "bfr/rhi/format.h"

#include <CoreGraphics/CGColorSpace.h>
#include <Metal/MTLPixelFormat.hpp>

#include <algorithm>
#include <array>
#include <span>

namespace bfr::rhi::metal {

namespace {

struct MetalFormat {
    Format format;
    MTL::PixelFormat metal;
};

// Each format's Metal pixel format, by name; Invalid where Metal has none
// (no three-channel 32-bit formats, and none named as Bgra4Unorm is).
constexpr std::array kMetalFormats{
    MetalFormat{.format = Format::R8Unorm, .metal = MTL::PixelFormatR8Unorm},
    MetalFormat{.format = Format::R8Snorm, .metal = MTL::PixelFormatR8Snorm},
    MetalFormat{.format = Format::R8Uint, .metal = MTL::PixelFormatR8Uint},
    MetalFormat{.format = Format::R8Sint, .metal = MTL::PixelFormatR8Sint},
    MetalFormat{.format = Format::Rg8Unorm, .metal = MTL::PixelFormatRG8Unorm},
    MetalFormat{.format = Format::Rg8Snorm, .metal = MTL::PixelFormatRG8Snorm},
    MetalFormat{.format = Format::Rg8Uint, .metal = MTL::PixelFormatRG8Uint},
    MetalFormat{.format = Format::Rg8Sint, .metal = MTL::PixelFormatRG8Sint},
    MetalFormat{.format = Format::Rgba8Unorm, .metal = MTL::PixelFormatRGBA8Unorm},
    MetalFormat{.format = Format::Rgba8Srgb, .metal = MTL::PixelFormatRGBA8Unorm_sRGB},
    MetalFormat{.format = Format::Rgba8Snorm, .metal = MTL::PixelFormatRGBA8Snorm},
    MetalFormat{.format = Format::Rgba8Uint, .metal = MTL::PixelFormatRGBA8Uint},
    MetalFormat{.format = Format::Rgba8Sint, .metal = MTL::PixelFormatRGBA8Sint},
    MetalFormat{.format = Format::Bgra8Unorm, .metal = MTL::PixelFormatBGRA8Unorm},
    MetalFormat{.format = Format::Bgra8Srgb, .metal = MTL::PixelFormatBGRA8Unorm_sRGB},
    MetalFormat{.format = Format::R16Unorm, .metal = MTL::PixelFormatR16Unorm},
    MetalFormat{.format = Format::R16Snorm, .metal = MTL::PixelFormatR16Snorm},
    MetalFormat{.format = Format::R16Uint, .metal = MTL::PixelFormatR16Uint},
    MetalFormat{.format = Format::R16Sint, .metal = MTL::PixelFormatR16Sint},
    MetalFormat{.format = Format::R16Float, .metal = MTL::PixelFormatR16Float},
    MetalFormat{.format = Format::Rg16Unorm, .metal = MTL::PixelFormatRG16Unorm},
    MetalFormat{.format = Format::Rg16Snorm, .metal = MTL::PixelFormatRG16Snorm},
    MetalFormat{.format = Format::Rg16Uint, .metal = MTL::PixelFormatRG16Uint},
    MetalFormat{.format = Format::Rg16Sint, .metal = MTL::PixelFormatRG16Sint},
    MetalFormat{.format = Format::Rg16Float, .metal = MTL::PixelFormatRG16Float},
    MetalFormat{.format = Format::Rgba16Unorm, .metal = MTL::PixelFormatRGBA16Unorm},
    MetalFormat{.format = Format::Rgba16Snorm, .metal = MTL::PixelFormatRGBA16Snorm},
    MetalFormat{.format = Format::Rgba16Uint, .metal = MTL::PixelFormatRGBA16Uint},
    MetalFormat{.format = Format::Rgba16Sint, .metal = MTL::PixelFormatRGBA16Sint},
    MetalFormat{.format = Format::Rgba16Float, .metal = MTL::PixelFormatRGBA16Float},
    MetalFormat{.format = Format::R32Uint, .metal = MTL::PixelFormatR32Uint},
    MetalFormat{.format = Format::R32Sint, .metal = MTL::PixelFormatR32Sint},
    MetalFormat{.format = Format::R32Float, .metal = MTL::PixelFormatR32Float},
    MetalFormat{.format = Format::Rg32Uint, .metal = MTL::PixelFormatRG32Uint},
    MetalFormat{.format = Format::Rg32Sint, .metal = MTL::PixelFormatRG32Sint},
    MetalFormat{.format = Format::Rg32Float, .metal = MTL::PixelFormatRG32Float},
    MetalFormat{.format = Format::Rgb32Uint, .metal = MTL::PixelFormatInvalid},
    MetalFormat{.format = Format::Rgb32Sint, .metal = MTL::PixelFormatInvalid},
    MetalFormat{.format = Format::Rgb32Float, .metal = MTL::PixelFormatInvalid},
    MetalFormat{.format = Format::Rgba32Uint, .metal = MTL::PixelFormatRGBA32Uint},
    MetalFormat{.format = Format::Rgba32Sint, .metal = MTL::PixelFormatRGBA32Sint},
    MetalFormat{.format = Format::Rgba32Float, .metal = MTL::PixelFormatRGBA32Float},
    MetalFormat{.format = Format::Rgb10a2Unorm, .metal = MTL::PixelFormatRGB10A2Unorm},
    MetalFormat{.format = Format::Rgb10a2Uint, .metal = MTL::PixelFormatRGB10A2Uint},
    MetalFormat{.format = Format::Bgr10a2Unorm, .metal = MTL::PixelFormatBGR10A2Unorm},
    MetalFormat{.format = Format::Rg11b10Float, .metal = MTL::PixelFormatRG11B10Float},
    MetalFormat{.format = Format::Rgb9e5Float, .metal = MTL::PixelFormatRGB9E5Float},
    MetalFormat{.format = Format::B5g6r5Unorm, .metal = MTL::PixelFormatB5G6R5Unorm},
    MetalFormat{.format = Format::Bgr5a1Unorm, .metal = MTL::PixelFormatBGR5A1Unorm},
    MetalFormat{.format = Format::Bgra4Unorm, .metal = MTL::PixelFormatInvalid},
    MetalFormat{.format = Format::D16Unorm, .metal = MTL::PixelFormatDepth16Unorm},
    MetalFormat{.format = Format::D24UnormS8Uint, .metal = MTL::PixelFormatDepth24Unorm_Stencil8},
    MetalFormat{.format = Format::D32Float, .metal = MTL::PixelFormatDepth32Float},
    MetalFormat{.format = Format::D32FloatS8Uint, .metal = MTL::PixelFormatDepth32Float_Stencil8},
    MetalFormat{.format = Format::S8Uint, .metal = MTL::PixelFormatStencil8},
    MetalFormat{.format = Format::Bc1Unorm, .metal = MTL::PixelFormatBC1_RGBA},
    MetalFormat{.format = Format::Bc1Srgb, .metal = MTL::PixelFormatBC1_RGBA_sRGB},
    MetalFormat{.format = Format::Bc2Unorm, .metal = MTL::PixelFormatBC2_RGBA},
    MetalFormat{.format = Format::Bc2Srgb, .metal = MTL::PixelFormatBC2_RGBA_sRGB},
    MetalFormat{.format = Format::Bc3Unorm, .metal = MTL::PixelFormatBC3_RGBA},
    MetalFormat{.format = Format::Bc3Srgb, .metal = MTL::PixelFormatBC3_RGBA_sRGB},
    MetalFormat{.format = Format::Bc4Unorm, .metal = MTL::PixelFormatBC4_RUnorm},
    MetalFormat{.format = Format::Bc4Snorm, .metal = MTL::PixelFormatBC4_RSnorm},
    MetalFormat{.format = Format::Bc5Unorm, .metal = MTL::PixelFormatBC5_RGUnorm},
    MetalFormat{.format = Format::Bc5Snorm, .metal = MTL::PixelFormatBC5_RGSnorm},
    MetalFormat{.format = Format::Bc6hUfloat, .metal = MTL::PixelFormatBC6H_RGBUfloat},
    MetalFormat{.format = Format::Bc6hSfloat, .metal = MTL::PixelFormatBC6H_RGBFloat},
    MetalFormat{.format = Format::Bc7Unorm, .metal = MTL::PixelFormatBC7_RGBAUnorm},
    MetalFormat{.format = Format::Bc7Srgb, .metal = MTL::PixelFormatBC7_RGBAUnorm_sRGB},
    MetalFormat{.format = Format::Etc2Rgb8Unorm, .metal = MTL::PixelFormatETC2_RGB8},
    MetalFormat{.format = Format::Etc2Rgb8Srgb, .metal = MTL::PixelFormatETC2_RGB8_sRGB},
    MetalFormat{.format = Format::Etc2Rgb8a1Unorm, .metal = MTL::PixelFormatETC2_RGB8A1},
    MetalFormat{.format = Format::Etc2Rgb8a1Srgb, .metal = MTL::PixelFormatETC2_RGB8A1_sRGB},
    MetalFormat{.format = Format::Etc2Rgba8Unorm, .metal = MTL::PixelFormatEAC_RGBA8},
    MetalFormat{.format = Format::Etc2Rgba8Srgb, .metal = MTL::PixelFormatEAC_RGBA8_sRGB},
    MetalFormat{.format = Format::EacR11Unorm, .metal = MTL::PixelFormatEAC_R11Unorm},
    MetalFormat{.format = Format::EacR11Snorm, .metal = MTL::PixelFormatEAC_R11Snorm},
    MetalFormat{.format = Format::EacRg11Unorm, .metal = MTL::PixelFormatEAC_RG11Unorm},
    MetalFormat{.format = Format::EacRg11Snorm, .metal = MTL::PixelFormatEAC_RG11Snorm},
    MetalFormat{.format = Format::Astc4x4Unorm, .metal = MTL::PixelFormatASTC_4x4_LDR},
    MetalFormat{.format = Format::Astc4x4Srgb, .metal = MTL::PixelFormatASTC_4x4_sRGB},
    MetalFormat{.format = Format::Astc4x4Float, .metal = MTL::PixelFormatASTC_4x4_HDR},
    MetalFormat{.format = Format::Astc5x4Unorm, .metal = MTL::PixelFormatASTC_5x4_LDR},
    MetalFormat{.format = Format::Astc5x4Srgb, .metal = MTL::PixelFormatASTC_5x4_sRGB},
    MetalFormat{.format = Format::Astc5x4Float, .metal = MTL::PixelFormatASTC_5x4_HDR},
    MetalFormat{.format = Format::Astc5x5Unorm, .metal = MTL::PixelFormatASTC_5x5_LDR},
    MetalFormat{.format = Format::Astc5x5Srgb, .metal = MTL::PixelFormatASTC_5x5_sRGB},
    MetalFormat{.format = Format::Astc5x5Float, .metal = MTL::PixelFormatASTC_5x5_HDR},
    MetalFormat{.format = Format::Astc6x5Unorm, .metal = MTL::PixelFormatASTC_6x5_LDR},
    MetalFormat{.format = Format::Astc6x5Srgb, .metal = MTL::PixelFormatASTC_6x5_sRGB},
    MetalFormat{.format = Format::Astc6x5Float, .metal = MTL::PixelFormatASTC_6x5_HDR},
    MetalFormat{.format = Format::Astc6x6Unorm, .metal = MTL::PixelFormatASTC_6x6_LDR},
    MetalFormat{.format = Format::Astc6x6Srgb, .metal = MTL::PixelFormatASTC_6x6_sRGB},
    MetalFormat{.format = Format::Astc6x6Float, .metal = MTL::PixelFormatASTC_6x6_HDR},
    MetalFormat{.format = Format::Astc8x5Unorm, .metal = MTL::PixelFormatASTC_8x5_LDR},
    MetalFormat{.format = Format::Astc8x5Srgb, .metal = MTL::PixelFormatASTC_8x5_sRGB},
    MetalFormat{.format = Format::Astc8x5Float, .metal = MTL::PixelFormatASTC_8x5_HDR},
    MetalFormat{.format = Format::Astc8x6Unorm, .metal = MTL::PixelFormatASTC_8x6_LDR},
    MetalFormat{.format = Format::Astc8x6Srgb, .metal = MTL::PixelFormatASTC_8x6_sRGB},
    MetalFormat{.format = Format::Astc8x6Float, .metal = MTL::PixelFormatASTC_8x6_HDR},
    MetalFormat{.format = Format::Astc8x8Unorm, .metal = MTL::PixelFormatASTC_8x8_LDR},
    MetalFormat{.format = Format::Astc8x8Srgb, .metal = MTL::PixelFormatASTC_8x8_sRGB},
    MetalFormat{.format = Format::Astc8x8Float, .metal = MTL::PixelFormatASTC_8x8_HDR},
    MetalFormat{.format = Format::Astc10x5Unorm, .metal = MTL::PixelFormatASTC_10x5_LDR},
    MetalFormat{.format = Format::Astc10x5Srgb, .metal = MTL::PixelFormatASTC_10x5_sRGB},
    MetalFormat{.format = Format::Astc10x5Float, .metal = MTL::PixelFormatASTC_10x5_HDR},
    MetalFormat{.format = Format::Astc10x6Unorm, .metal = MTL::PixelFormatASTC_10x6_LDR},
    MetalFormat{.format = Format::Astc10x6Srgb, .metal = MTL::PixelFormatASTC_10x6_sRGB},
    MetalFormat{.format = Format::Astc10x6Float, .metal = MTL::PixelFormatASTC_10x6_HDR},
    MetalFormat{.format = Format::Astc10x8Unorm, .metal = MTL::PixelFormatASTC_10x8_LDR},
    MetalFormat{.format = Format::Astc10x8Srgb, .metal = MTL::PixelFormatASTC_10x8_sRGB},
    MetalFormat{.format = Format::Astc10x8Float, .metal = MTL::PixelFormatASTC_10x8_HDR},
    MetalFormat{.format = Format::Astc10x10Unorm, .metal = MTL::PixelFormatASTC_10x10_LDR},
    MetalFormat{.format = Format::Astc10x10Srgb, .metal = MTL::PixelFormatASTC_10x10_sRGB},
    MetalFormat{.format = Format::Astc10x10Float, .metal = MTL::PixelFormatASTC_10x10_HDR},
    MetalFormat{.format = Format::Astc12x10Unorm, .metal = MTL::PixelFormatASTC_12x10_LDR},
    MetalFormat{.format = Format::Astc12x10Srgb, .metal = MTL::PixelFormatASTC_12x10_sRGB},
    MetalFormat{.format = Format::Astc12x10Float, .metal = MTL::PixelFormatASTC_12x10_HDR},
    MetalFormat{.format = Format::Astc12x12Unorm, .metal = MTL::PixelFormatASTC_12x12_LDR},
    MetalFormat{.format = Format::Astc12x12Srgb, .metal = MTL::PixelFormatASTC_12x12_sRGB},
    MetalFormat{.format = Format::Astc12x12Float, .metal = MTL::PixelFormatASTC_12x12_HDR},
};

static_assert(kMetalFormats.size() == kFormats.size());

} // namespace

MTL::PixelFormat pixelFormatFor(Format format) {
    return std::ranges::find(kMetalFormats, format, &MetalFormat::format)->metal;
}

SwapchainCaps surfaceCapsFor() {
    return {};
}

std::span<const SwapchainFormat> swapchainFormats() {
    return {};
}

LayerFormat layerFormatFor(Format format, ColorSpace colorSpace) {
    if (colorSpace == ColorSpace::Srgb && format == Format::Bgra8Unorm) {
        return {.pixelFormat = MTL::PixelFormatBGRA8Unorm, .colorSpaceName = kCGColorSpaceSRGB, .extendedRange = false};
    }
    if (colorSpace == ColorSpace::SrgbExtendedLinear && format == Format::Rgba16Float) {
        return {.pixelFormat = MTL::PixelFormatRGBA16Float,
                .colorSpaceName = kCGColorSpaceExtendedLinearSRGB,
                .extendedRange = true};
    }
    return {};
}

} // namespace bfr::rhi::metal
