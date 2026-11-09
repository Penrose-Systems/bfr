#include "metal/metal_formats.h"

#include "bfr/rhi/caps.h"
#include "bfr/rhi/format.h"
#include "metal/metal_caps.h"

#include <CoreGraphics/CGColorSpace.h>
#include <Metal/MTLPixelFormat.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>

namespace bfr::rhi::metal {

namespace {

// What a pixel format may do, as Godot's Metal driver records Apple's Metal
// feature set tables (drivers/metal/pixel_formats.h:74-112). Metal has no call
// for it.
enum FormatBits : std::uint16_t {
    kNone = 0,
    kRead = 1U << 0U,
    kFilter = 1U << 1U,
    kWrite = 1U << 2U,
    kAtomic = 1U << 3U,
    kColor = 1U << 4U,
    kDepth = 1U << 5U,
    kBlend = 1U << 6U,
    kMsaa = 1U << 7U,
    kResolve = 1U << 8U,
};

constexpr std::uint16_t kRf = kRead | kFilter;
constexpr std::uint16_t kRwc = kRead | kWrite | kColor;
constexpr std::uint16_t kRwcm = kRwc | kMsaa;
constexpr std::uint16_t kRfcmrb = kRf | kColor | kMsaa | kResolve | kBlend;
constexpr std::uint16_t kRfwcmb = kRf | kWrite | kColor | kMsaa | kBlend;
constexpr std::uint16_t kAll = kRfwcmb | kResolve;
constexpr std::uint16_t kDrm = kDepth | kRead | kMsaa;
constexpr std::uint16_t kDrmr = kDrm | kResolve;
constexpr std::uint16_t kDrfmr = kDrm | kFilter | kResolve;
constexpr std::uint16_t kDrf = kDepth | kRf;

// A format rendered to without multisampling renders at one sample, where the
// device says it takes one.
constexpr std::uint32_t kOneSample = 1;

struct MetalFormat {
    Format format;
    MTL::PixelFormat metal;
    /// What it may do on a Mac, Godot's base table
    /// (pixel_formats.cpp:609-780); formatCapsFor adjusts it per device.
    std::uint16_t caps;
};

// Each format's Metal pixel format, by name; Invalid where Metal has none
// (no three-channel 32-bit formats, no 64-bit ones, and none named as
// Bgra4Unorm is). Every
// compressed format is read and filtered. Godot's base table gives
// Depth24Unorm_Stencil8 nothing; formatCapsFor asks the device for it.
constexpr std::array kMetalFormats{
    MetalFormat{.format = Format::R8Unorm, .metal = MTL::PixelFormatR8Unorm, .caps = kAll},
    MetalFormat{.format = Format::R8Snorm, .metal = MTL::PixelFormatR8Snorm, .caps = kAll},
    MetalFormat{.format = Format::R8Uint, .metal = MTL::PixelFormatR8Uint, .caps = kRwcm},
    MetalFormat{.format = Format::R8Sint, .metal = MTL::PixelFormatR8Sint, .caps = kRwcm},
    MetalFormat{.format = Format::Rg8Unorm, .metal = MTL::PixelFormatRG8Unorm, .caps = kAll},
    MetalFormat{.format = Format::Rg8Snorm, .metal = MTL::PixelFormatRG8Snorm, .caps = kAll},
    MetalFormat{.format = Format::Rg8Uint, .metal = MTL::PixelFormatRG8Uint, .caps = kRwcm},
    MetalFormat{.format = Format::Rg8Sint, .metal = MTL::PixelFormatRG8Sint, .caps = kRwcm},
    MetalFormat{.format = Format::Rgba8Unorm, .metal = MTL::PixelFormatRGBA8Unorm, .caps = kAll},
    MetalFormat{.format = Format::Rgba8Srgb, .metal = MTL::PixelFormatRGBA8Unorm_sRGB, .caps = kAll},
    MetalFormat{.format = Format::Rgba8Snorm, .metal = MTL::PixelFormatRGBA8Snorm, .caps = kAll},
    MetalFormat{.format = Format::Rgba8Uint, .metal = MTL::PixelFormatRGBA8Uint, .caps = kRwcm},
    MetalFormat{.format = Format::Rgba8Sint, .metal = MTL::PixelFormatRGBA8Sint, .caps = kRwcm},
    MetalFormat{.format = Format::Bgra8Unorm, .metal = MTL::PixelFormatBGRA8Unorm, .caps = kAll},
    MetalFormat{.format = Format::Bgra8Srgb, .metal = MTL::PixelFormatBGRA8Unorm_sRGB, .caps = kAll},
    MetalFormat{.format = Format::R16Unorm, .metal = MTL::PixelFormatR16Unorm, .caps = kRfwcmb},
    MetalFormat{.format = Format::R16Snorm, .metal = MTL::PixelFormatR16Snorm, .caps = kRfwcmb},
    MetalFormat{.format = Format::R16Uint, .metal = MTL::PixelFormatR16Uint, .caps = kRwcm},
    MetalFormat{.format = Format::R16Sint, .metal = MTL::PixelFormatR16Sint, .caps = kRwcm},
    MetalFormat{.format = Format::R16Float, .metal = MTL::PixelFormatR16Float, .caps = kAll},
    MetalFormat{.format = Format::Rg16Unorm, .metal = MTL::PixelFormatRG16Unorm, .caps = kRfwcmb},
    MetalFormat{.format = Format::Rg16Snorm, .metal = MTL::PixelFormatRG16Snorm, .caps = kRfwcmb},
    MetalFormat{.format = Format::Rg16Uint, .metal = MTL::PixelFormatRG16Uint, .caps = kRwcm},
    MetalFormat{.format = Format::Rg16Sint, .metal = MTL::PixelFormatRG16Sint, .caps = kRwcm},
    MetalFormat{.format = Format::Rg16Float, .metal = MTL::PixelFormatRG16Float, .caps = kAll},
    MetalFormat{.format = Format::Rgba16Unorm, .metal = MTL::PixelFormatRGBA16Unorm, .caps = kRfwcmb},
    MetalFormat{.format = Format::Rgba16Snorm, .metal = MTL::PixelFormatRGBA16Snorm, .caps = kRfwcmb},
    MetalFormat{.format = Format::Rgba16Uint, .metal = MTL::PixelFormatRGBA16Uint, .caps = kRwcm},
    MetalFormat{.format = Format::Rgba16Sint, .metal = MTL::PixelFormatRGBA16Sint, .caps = kRwcm},
    MetalFormat{.format = Format::Rgba16Float, .metal = MTL::PixelFormatRGBA16Float, .caps = kAll},
    MetalFormat{.format = Format::R32Uint, .metal = MTL::PixelFormatR32Uint, .caps = kRwc},
    MetalFormat{.format = Format::R32Sint, .metal = MTL::PixelFormatR32Sint, .caps = kRwc},
    MetalFormat{.format = Format::R32Float, .metal = MTL::PixelFormatR32Float, .caps = kAll},
    MetalFormat{.format = Format::Rg32Uint, .metal = MTL::PixelFormatRG32Uint, .caps = kRwcm},
    MetalFormat{.format = Format::Rg32Sint, .metal = MTL::PixelFormatRG32Sint, .caps = kRwcm},
    MetalFormat{.format = Format::Rg32Float, .metal = MTL::PixelFormatRG32Float, .caps = kAll},
    MetalFormat{.format = Format::Rgb32Uint, .metal = MTL::PixelFormatInvalid, .caps = kNone},
    MetalFormat{.format = Format::Rgb32Sint, .metal = MTL::PixelFormatInvalid, .caps = kNone},
    MetalFormat{.format = Format::Rgb32Float, .metal = MTL::PixelFormatInvalid, .caps = kNone},
    MetalFormat{.format = Format::Rgba32Uint, .metal = MTL::PixelFormatRGBA32Uint, .caps = kRwc},
    MetalFormat{.format = Format::Rgba32Sint, .metal = MTL::PixelFormatRGBA32Sint, .caps = kRwc},
    MetalFormat{.format = Format::Rgba32Float, .metal = MTL::PixelFormatRGBA32Float, .caps = kAll},
    MetalFormat{.format = Format::Rgb10a2Unorm, .metal = MTL::PixelFormatRGB10A2Unorm, .caps = kAll},
    MetalFormat{.format = Format::Rgb10a2Uint, .metal = MTL::PixelFormatRGB10A2Uint, .caps = kRwcm},
    MetalFormat{.format = Format::Bgr10a2Unorm, .metal = MTL::PixelFormatBGR10A2Unorm, .caps = kAll},
    MetalFormat{.format = Format::Rg11b10Float, .metal = MTL::PixelFormatRG11B10Float, .caps = kAll},
    MetalFormat{.format = Format::Rgb9e5Float, .metal = MTL::PixelFormatRGB9E5Float, .caps = kAll},
    MetalFormat{.format = Format::B5g6r5Unorm, .metal = MTL::PixelFormatB5G6R5Unorm, .caps = kRfcmrb},
    MetalFormat{.format = Format::Bgr5a1Unorm, .metal = MTL::PixelFormatBGR5A1Unorm, .caps = kRfcmrb},
    MetalFormat{.format = Format::Bgra4Unorm, .metal = MTL::PixelFormatInvalid, .caps = kNone},
    MetalFormat{.format = Format::R64Uint, .metal = MTL::PixelFormatInvalid, .caps = kNone},
    MetalFormat{.format = Format::R64Sint, .metal = MTL::PixelFormatInvalid, .caps = kNone},
    MetalFormat{.format = Format::R64Float, .metal = MTL::PixelFormatInvalid, .caps = kNone},
    MetalFormat{.format = Format::Rg64Uint, .metal = MTL::PixelFormatInvalid, .caps = kNone},
    MetalFormat{.format = Format::Rg64Sint, .metal = MTL::PixelFormatInvalid, .caps = kNone},
    MetalFormat{.format = Format::Rg64Float, .metal = MTL::PixelFormatInvalid, .caps = kNone},
    MetalFormat{.format = Format::Rgb64Uint, .metal = MTL::PixelFormatInvalid, .caps = kNone},
    MetalFormat{.format = Format::Rgb64Sint, .metal = MTL::PixelFormatInvalid, .caps = kNone},
    MetalFormat{.format = Format::Rgb64Float, .metal = MTL::PixelFormatInvalid, .caps = kNone},
    MetalFormat{.format = Format::Rgba64Uint, .metal = MTL::PixelFormatInvalid, .caps = kNone},
    MetalFormat{.format = Format::Rgba64Sint, .metal = MTL::PixelFormatInvalid, .caps = kNone},
    MetalFormat{.format = Format::Rgba64Float, .metal = MTL::PixelFormatInvalid, .caps = kNone},
    MetalFormat{.format = Format::D16Unorm, .metal = MTL::PixelFormatDepth16Unorm, .caps = kDrfmr},
    MetalFormat{.format = Format::D24UnormS8Uint, .metal = MTL::PixelFormatDepth24Unorm_Stencil8, .caps = kNone},
    MetalFormat{.format = Format::D32Float, .metal = MTL::PixelFormatDepth32Float, .caps = kDrmr},
    MetalFormat{.format = Format::D32FloatS8Uint, .metal = MTL::PixelFormatDepth32Float_Stencil8, .caps = kDrmr},
    MetalFormat{.format = Format::S8Uint, .metal = MTL::PixelFormatStencil8, .caps = kDrm},
    MetalFormat{.format = Format::Bc1Unorm, .metal = MTL::PixelFormatBC1_RGBA, .caps = kRf},
    MetalFormat{.format = Format::Bc1Srgb, .metal = MTL::PixelFormatBC1_RGBA_sRGB, .caps = kRf},
    MetalFormat{.format = Format::Bc2Unorm, .metal = MTL::PixelFormatBC2_RGBA, .caps = kRf},
    MetalFormat{.format = Format::Bc2Srgb, .metal = MTL::PixelFormatBC2_RGBA_sRGB, .caps = kRf},
    MetalFormat{.format = Format::Bc3Unorm, .metal = MTL::PixelFormatBC3_RGBA, .caps = kRf},
    MetalFormat{.format = Format::Bc3Srgb, .metal = MTL::PixelFormatBC3_RGBA_sRGB, .caps = kRf},
    MetalFormat{.format = Format::Bc4Unorm, .metal = MTL::PixelFormatBC4_RUnorm, .caps = kRf},
    MetalFormat{.format = Format::Bc4Snorm, .metal = MTL::PixelFormatBC4_RSnorm, .caps = kRf},
    MetalFormat{.format = Format::Bc5Unorm, .metal = MTL::PixelFormatBC5_RGUnorm, .caps = kRf},
    MetalFormat{.format = Format::Bc5Snorm, .metal = MTL::PixelFormatBC5_RGSnorm, .caps = kRf},
    MetalFormat{.format = Format::Bc6hUfloat, .metal = MTL::PixelFormatBC6H_RGBUfloat, .caps = kRf},
    MetalFormat{.format = Format::Bc6hSfloat, .metal = MTL::PixelFormatBC6H_RGBFloat, .caps = kRf},
    MetalFormat{.format = Format::Bc7Unorm, .metal = MTL::PixelFormatBC7_RGBAUnorm, .caps = kRf},
    MetalFormat{.format = Format::Bc7Srgb, .metal = MTL::PixelFormatBC7_RGBAUnorm_sRGB, .caps = kRf},
    MetalFormat{.format = Format::Etc2Rgb8Unorm, .metal = MTL::PixelFormatETC2_RGB8, .caps = kRf},
    MetalFormat{.format = Format::Etc2Rgb8Srgb, .metal = MTL::PixelFormatETC2_RGB8_sRGB, .caps = kRf},
    MetalFormat{.format = Format::Etc2Rgb8a1Unorm, .metal = MTL::PixelFormatETC2_RGB8A1, .caps = kRf},
    MetalFormat{.format = Format::Etc2Rgb8a1Srgb, .metal = MTL::PixelFormatETC2_RGB8A1_sRGB, .caps = kRf},
    MetalFormat{.format = Format::Etc2Rgba8Unorm, .metal = MTL::PixelFormatEAC_RGBA8, .caps = kRf},
    MetalFormat{.format = Format::Etc2Rgba8Srgb, .metal = MTL::PixelFormatEAC_RGBA8_sRGB, .caps = kRf},
    MetalFormat{.format = Format::EacR11Unorm, .metal = MTL::PixelFormatEAC_R11Unorm, .caps = kRf},
    MetalFormat{.format = Format::EacR11Snorm, .metal = MTL::PixelFormatEAC_R11Snorm, .caps = kRf},
    MetalFormat{.format = Format::EacRg11Unorm, .metal = MTL::PixelFormatEAC_RG11Unorm, .caps = kRf},
    MetalFormat{.format = Format::EacRg11Snorm, .metal = MTL::PixelFormatEAC_RG11Snorm, .caps = kRf},
    MetalFormat{.format = Format::Astc4x4Unorm, .metal = MTL::PixelFormatASTC_4x4_LDR, .caps = kRf},
    MetalFormat{.format = Format::Astc4x4Srgb, .metal = MTL::PixelFormatASTC_4x4_sRGB, .caps = kRf},
    MetalFormat{.format = Format::Astc4x4Float, .metal = MTL::PixelFormatASTC_4x4_HDR, .caps = kRf},
    MetalFormat{.format = Format::Astc5x4Unorm, .metal = MTL::PixelFormatASTC_5x4_LDR, .caps = kRf},
    MetalFormat{.format = Format::Astc5x4Srgb, .metal = MTL::PixelFormatASTC_5x4_sRGB, .caps = kRf},
    MetalFormat{.format = Format::Astc5x4Float, .metal = MTL::PixelFormatASTC_5x4_HDR, .caps = kRf},
    MetalFormat{.format = Format::Astc5x5Unorm, .metal = MTL::PixelFormatASTC_5x5_LDR, .caps = kRf},
    MetalFormat{.format = Format::Astc5x5Srgb, .metal = MTL::PixelFormatASTC_5x5_sRGB, .caps = kRf},
    MetalFormat{.format = Format::Astc5x5Float, .metal = MTL::PixelFormatASTC_5x5_HDR, .caps = kRf},
    MetalFormat{.format = Format::Astc6x5Unorm, .metal = MTL::PixelFormatASTC_6x5_LDR, .caps = kRf},
    MetalFormat{.format = Format::Astc6x5Srgb, .metal = MTL::PixelFormatASTC_6x5_sRGB, .caps = kRf},
    MetalFormat{.format = Format::Astc6x5Float, .metal = MTL::PixelFormatASTC_6x5_HDR, .caps = kRf},
    MetalFormat{.format = Format::Astc6x6Unorm, .metal = MTL::PixelFormatASTC_6x6_LDR, .caps = kRf},
    MetalFormat{.format = Format::Astc6x6Srgb, .metal = MTL::PixelFormatASTC_6x6_sRGB, .caps = kRf},
    MetalFormat{.format = Format::Astc6x6Float, .metal = MTL::PixelFormatASTC_6x6_HDR, .caps = kRf},
    MetalFormat{.format = Format::Astc8x5Unorm, .metal = MTL::PixelFormatASTC_8x5_LDR, .caps = kRf},
    MetalFormat{.format = Format::Astc8x5Srgb, .metal = MTL::PixelFormatASTC_8x5_sRGB, .caps = kRf},
    MetalFormat{.format = Format::Astc8x5Float, .metal = MTL::PixelFormatASTC_8x5_HDR, .caps = kRf},
    MetalFormat{.format = Format::Astc8x6Unorm, .metal = MTL::PixelFormatASTC_8x6_LDR, .caps = kRf},
    MetalFormat{.format = Format::Astc8x6Srgb, .metal = MTL::PixelFormatASTC_8x6_sRGB, .caps = kRf},
    MetalFormat{.format = Format::Astc8x6Float, .metal = MTL::PixelFormatASTC_8x6_HDR, .caps = kRf},
    MetalFormat{.format = Format::Astc8x8Unorm, .metal = MTL::PixelFormatASTC_8x8_LDR, .caps = kRf},
    MetalFormat{.format = Format::Astc8x8Srgb, .metal = MTL::PixelFormatASTC_8x8_sRGB, .caps = kRf},
    MetalFormat{.format = Format::Astc8x8Float, .metal = MTL::PixelFormatASTC_8x8_HDR, .caps = kRf},
    MetalFormat{.format = Format::Astc10x5Unorm, .metal = MTL::PixelFormatASTC_10x5_LDR, .caps = kRf},
    MetalFormat{.format = Format::Astc10x5Srgb, .metal = MTL::PixelFormatASTC_10x5_sRGB, .caps = kRf},
    MetalFormat{.format = Format::Astc10x5Float, .metal = MTL::PixelFormatASTC_10x5_HDR, .caps = kRf},
    MetalFormat{.format = Format::Astc10x6Unorm, .metal = MTL::PixelFormatASTC_10x6_LDR, .caps = kRf},
    MetalFormat{.format = Format::Astc10x6Srgb, .metal = MTL::PixelFormatASTC_10x6_sRGB, .caps = kRf},
    MetalFormat{.format = Format::Astc10x6Float, .metal = MTL::PixelFormatASTC_10x6_HDR, .caps = kRf},
    MetalFormat{.format = Format::Astc10x8Unorm, .metal = MTL::PixelFormatASTC_10x8_LDR, .caps = kRf},
    MetalFormat{.format = Format::Astc10x8Srgb, .metal = MTL::PixelFormatASTC_10x8_sRGB, .caps = kRf},
    MetalFormat{.format = Format::Astc10x8Float, .metal = MTL::PixelFormatASTC_10x8_HDR, .caps = kRf},
    MetalFormat{.format = Format::Astc10x10Unorm, .metal = MTL::PixelFormatASTC_10x10_LDR, .caps = kRf},
    MetalFormat{.format = Format::Astc10x10Srgb, .metal = MTL::PixelFormatASTC_10x10_sRGB, .caps = kRf},
    MetalFormat{.format = Format::Astc10x10Float, .metal = MTL::PixelFormatASTC_10x10_HDR, .caps = kRf},
    MetalFormat{.format = Format::Astc12x10Unorm, .metal = MTL::PixelFormatASTC_12x10_LDR, .caps = kRf},
    MetalFormat{.format = Format::Astc12x10Srgb, .metal = MTL::PixelFormatASTC_12x10_sRGB, .caps = kRf},
    MetalFormat{.format = Format::Astc12x10Float, .metal = MTL::PixelFormatASTC_12x10_HDR, .caps = kRf},
    MetalFormat{.format = Format::Astc12x12Unorm, .metal = MTL::PixelFormatASTC_12x12_LDR, .caps = kRf},
    MetalFormat{.format = Format::Astc12x12Srgb, .metal = MTL::PixelFormatASTC_12x12_sRGB, .caps = kRf},
    MetalFormat{.format = Format::Astc12x12Float, .metal = MTL::PixelFormatASTC_12x12_HDR, .caps = kRf},
};

static_assert(kMetalFormats.size() == kFormats.size());

// Each format and color space a layer is set up to present in.
constexpr std::array kSwapchainFormats{
    SwapchainFormat{.format = Format::Bgra8Unorm, .colorSpace = ColorSpace::Srgb},
    SwapchainFormat{.format = Format::Rgba16Float, .colorSpace = ColorSpace::SrgbExtendedLinear},
};

} // namespace

MTL::PixelFormat pixelFormatFor(Format format) {
    if (format == Format::Undefined) {
        return MTL::PixelFormatInvalid;
    }
    return std::ranges::find(kMetalFormats, format, &MetalFormat::format)->metal;
}

Format formatFor(MTL::PixelFormat pixelFormat) {
    if (pixelFormat == MTL::PixelFormatInvalid) {
        return Format::Undefined;
    }
    const auto* found = std::ranges::find(kMetalFormats, pixelFormat, &MetalFormat::metal);
    return found != kMetalFormats.end() ? found->format : Format::Undefined;
}

FormatCaps formatCapsFor(MTL::Device* device, Format format) {
    const MetalFormat& metal = *std::ranges::find(kMetalFormats, format, &MetalFormat::format);
    std::uint16_t caps = metal.caps;
    if (metal.metal == MTL::PixelFormatInvalid) {
        caps = kNone;
    }
    // Godot's changes for one device (pixel_formats.cpp:897-1000), in its
    // order: whole rows, then bits added, then bits taken. Its changes for
    // families below Apple7 never apply, as Metal 4 runs on Apple7 and later
    // (belowMinimum); nor does the one it makes only for Vulkan's sake (:931).
    const bool iosBelowApple9 = TARGET_OS_OSX == 0 && !device->supportsFamily(MTL::GPUFamilyApple9);
    if (iosBelowApple9 && (format == Format::R32Float || format == Format::Rg32Float)) {
        caps = kRwcm | kBlend;
    }
    if (iosBelowApple9 && format == Format::Rgba32Float) {
        caps = kRwcm;
    }
    const bool bc = format >= Format::Bc1Unorm && format <= Format::Bc7Srgb;
    if (bc && !device->supportsBCTextureCompression()) {
        caps = kNone;
    }
    const bool int32 = format == Format::R32Uint || format == Format::R32Sint || format == Format::Rg32Uint ||
                       format == Format::Rg32Sint || format == Format::Rgba32Uint || format == Format::Rgba32Sint;
    const bool float32 = format == Format::R32Float || format == Format::Rg32Float || format == Format::Rgba32Float;
    if ((format == Format::R32Uint || format == Format::R32Sint) && device->supportsFamily(MTL::GPUFamilyApple6)) {
        caps |= kAtomic;
    }
    if ((format == Format::Rg32Uint || format == Format::Rg32Sint) && has64BitAtomics(device)) {
        caps |= kAtomic;
    }
    if (int32 && device->supports32BitMSAA()) {
        caps |= kMsaa;
    }
    if (float32 && device->supports32BitMSAA()) {
        caps |= kResolve;
    }
    if (float32 && device->supports32BitFloatFiltering()) {
        caps |= kFilter;
    }
    if (format == Format::Rgba32Float && device->supports32BitFloatFiltering()) {
        caps |= kBlend;
    }
    if (format == Format::Rgb9e5Float && TARGET_OS_SIMULATOR != 0) {
        caps &= static_cast<std::uint16_t>(~kColor);
    }

    // Godot never gives Depth24Unorm_Stencil8 anything (:779, :1012). The Forge
    // samples it linearly and renders to it where the device says it has it
    // (Common_3/Graphics/Metal/MetalCapBuilder.h:440-443).
    if (format == Format::D24UnormS8Uint && device->isDepth24Stencil8PixelFormatSupported()) {
        caps = kDrf;
    }

    const auto has = [caps](std::uint16_t bit) {
        return (caps & bit) != 0;
    };
    const bool renders = has(kColor) || has(kDepth);
    // Godot leaves copies on for every format the device has
    // (rendering_device_driver_metal.cpp:694-709). Its table has no texel
    // buffer or min and max filter columns.
    return FormatCaps{
        .format = format,
        .sampled = has(kRead),
        .filterLinear = has(kFilter),
        .filterMinmax = false,
        .storage = has(kWrite),
        .storageAtomic = has(kAtomic),
        .colorAttachment = has(kColor),
        .blend = has(kBlend),
        .depthStencil = has(kDepth),
        .copySrc = caps != kNone,
        .copyDst = caps != kNone,
        .texelBuffer = false,
        .storageTexelBuffer = false,
        .storageTexelBufferAtomic = false,
        .sampleCounts = has(kMsaa) ? sampleCountsOf(device) : (renders ? sampleCountsOf(device) & kOneSample : 0U),
        // Godot's table gives depth formats resolve on every Metal 4 device
        // but gates depth resolve itself apart, as Caps::resolve does.
        .resolve = has(kResolve) && (!has(kDepth) || resolveCapsOf(device).depthModes != 0),
    };
}

std::span<const SwapchainFormat> swapchainFormats() {
    return kSwapchainFormats;
}

LayerFormat layerFormatFor(Format format, ColorSpace colorSpace) {
    const bool listed = std::ranges::any_of(kSwapchainFormats, [format, colorSpace](const SwapchainFormat& entry) {
        return entry.format == format && entry.colorSpace == colorSpace;
    });
    if (!listed) {
        return {};
    }
    // The color space names the layer's CGColorSpace; an extended range one
    // also wants extended dynamic range content.
    switch (colorSpace) {
        case ColorSpace::Srgb:
            return {.pixelFormat = pixelFormatFor(format), .colorSpaceName = kCGColorSpaceSRGB, .extendedRange = false};
        case ColorSpace::SrgbExtendedLinear:
            return {.pixelFormat = pixelFormatFor(format),
                    .colorSpaceName = kCGColorSpaceExtendedLinearSRGB,
                    .extendedRange = true};
        default:
            return {};
    }
}

} // namespace bfr::rhi::metal
