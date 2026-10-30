#include "metal/metal_formats.h"

#include "bfr/rhi/caps.h"
#include "bfr/rhi/format.h"
#include "rhi/swapchain_build.h"

#include <CoreGraphics/CGColorSpace.h>
#include <Metal/MTLDevice.hpp>
#include <Metal/MTLPixelFormat.hpp>
#include <TargetConditionals.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>

namespace bfr::rhi::metal {

namespace {

// What Metal lets a format do. The table below is Godot's, for Apple GPUs:
// drivers/metal/pixel_formats.cpp, initMTLPixelFormatCapabilities.
enum Bit : std::uint16_t {
    Read = 1U << 0,
    Filter = 1U << 1,
    Write = 1U << 2,
    Atomic = 1U << 3,
    ColorAttachment = 1U << 4,
    DepthStencilAttachment = 1U << 5,
    Blend = 1U << 6,
    Multisample = 1U << 7,
    Resolve = 1U << 8,
};

using Bits = std::uint16_t;

constexpr Bits kRf = Read | Filter;
constexpr Bits kRwc = Read | Write | ColorAttachment;
constexpr Bits kRwcm = kRwc | Multisample;
constexpr Bits kRwcmb = kRwcm | Blend;
constexpr Bits kRfwcmb = kRwcm | Filter | Blend;
constexpr Bits kRfcmrb = Read | Filter | ColorAttachment | Multisample | Resolve | Blend;
constexpr Bits kAll = kRfwcmb | Resolve;
constexpr Bits kDrm = DepthStencilAttachment | Read | Multisample;
constexpr Bits kDrmr = kDrm | Resolve;
constexpr Bits kDrfmr = kDrmr | Filter;

struct Entry {
    constexpr Entry(Format entryFormat, MTL::PixelFormat entryPixelFormat, Bits entryBits)
        : format(entryFormat), pixelFormat(entryPixelFormat), bits(entryBits) {}

    Format format;
    MTL::PixelFormat pixelFormat;
    Bits bits;
};

constexpr std::array kTable{
    Entry{Format::R8Unorm, MTL::PixelFormatR8Unorm, kAll},
    Entry{Format::R8Snorm, MTL::PixelFormatR8Snorm, kAll},
    Entry{Format::R8Uint, MTL::PixelFormatR8Uint, kRwcm},
    Entry{Format::R8Sint, MTL::PixelFormatR8Sint, kRwcm},
    Entry{Format::Rg8Unorm, MTL::PixelFormatRG8Unorm, kAll},
    Entry{Format::Rg8Snorm, MTL::PixelFormatRG8Snorm, kAll},
    Entry{Format::Rg8Uint, MTL::PixelFormatRG8Uint, kRwcm},
    Entry{Format::Rg8Sint, MTL::PixelFormatRG8Sint, kRwcm},
    Entry{Format::Rgba8Unorm, MTL::PixelFormatRGBA8Unorm, kAll},
    Entry{Format::Rgba8Srgb, MTL::PixelFormatRGBA8Unorm_sRGB, kAll},
    Entry{Format::Rgba8Snorm, MTL::PixelFormatRGBA8Snorm, kAll},
    Entry{Format::Rgba8Uint, MTL::PixelFormatRGBA8Uint, kRwcm},
    Entry{Format::Rgba8Sint, MTL::PixelFormatRGBA8Sint, kRwcm},
    Entry{Format::Bgra8Unorm, MTL::PixelFormatBGRA8Unorm, kAll},
    Entry{Format::Bgra8Srgb, MTL::PixelFormatBGRA8Unorm_sRGB, kAll},
    Entry{Format::R16Unorm, MTL::PixelFormatR16Unorm, kRfwcmb},
    Entry{Format::R16Snorm, MTL::PixelFormatR16Snorm, kRfwcmb},
    Entry{Format::R16Uint, MTL::PixelFormatR16Uint, kRwcm},
    Entry{Format::R16Sint, MTL::PixelFormatR16Sint, kRwcm},
    Entry{Format::R16Float, MTL::PixelFormatR16Float, kAll},
    Entry{Format::Rg16Unorm, MTL::PixelFormatRG16Unorm, kRfwcmb},
    Entry{Format::Rg16Snorm, MTL::PixelFormatRG16Snorm, kRfwcmb},
    Entry{Format::Rg16Uint, MTL::PixelFormatRG16Uint, kRwcm},
    Entry{Format::Rg16Sint, MTL::PixelFormatRG16Sint, kRwcm},
    Entry{Format::Rg16Float, MTL::PixelFormatRG16Float, kAll},
    Entry{Format::Rgba16Unorm, MTL::PixelFormatRGBA16Unorm, kRfwcmb},
    Entry{Format::Rgba16Snorm, MTL::PixelFormatRGBA16Snorm, kRfwcmb},
    Entry{Format::Rgba16Uint, MTL::PixelFormatRGBA16Uint, kRwcm},
    Entry{Format::Rgba16Sint, MTL::PixelFormatRGBA16Sint, kRwcm},
    Entry{Format::Rgba16Float, MTL::PixelFormatRGBA16Float, kAll},
    Entry{Format::R32Uint, MTL::PixelFormatR32Uint, kRwc},
    Entry{Format::R32Sint, MTL::PixelFormatR32Sint, kRwc},
    Entry{Format::R32Float, MTL::PixelFormatR32Float, kAll},
    Entry{Format::Rg32Uint, MTL::PixelFormatRG32Uint, kRwcm},
    Entry{Format::Rg32Sint, MTL::PixelFormatRG32Sint, kRwcm},
    Entry{Format::Rg32Float, MTL::PixelFormatRG32Float, kAll},
    Entry{Format::Rgba32Uint, MTL::PixelFormatRGBA32Uint, kRwc},
    Entry{Format::Rgba32Sint, MTL::PixelFormatRGBA32Sint, kRwc},
    Entry{Format::Rgba32Float, MTL::PixelFormatRGBA32Float, kAll},
    Entry{Format::Rgb10a2Unorm, MTL::PixelFormatRGB10A2Unorm, kAll},
    Entry{Format::Rgb10a2Uint, MTL::PixelFormatRGB10A2Uint, kRwcm},
    Entry{Format::Bgr10a2Unorm, MTL::PixelFormatBGR10A2Unorm, kAll},
    Entry{Format::Rg11b10Float, MTL::PixelFormatRG11B10Float, kAll},
    Entry{Format::Rgb9e5Float, MTL::PixelFormatRGB9E5Float, kAll},
    Entry{Format::B5g6r5Unorm, MTL::PixelFormatB5G6R5Unorm, kRfcmrb},
    Entry{Format::Bgr5a1Unorm, MTL::PixelFormatBGR5A1Unorm, kRfcmrb},
    Entry{Format::D16Unorm, MTL::PixelFormatDepth16Unorm, kDrfmr},
    // Godot gives Apple GPUs none of this format's capabilities.
    Entry{Format::D24UnormS8Uint, MTL::PixelFormatDepth24Unorm_Stencil8, 0},
    Entry{Format::D32Float, MTL::PixelFormatDepth32Float, kDrmr},
    Entry{Format::D32FloatS8Uint, MTL::PixelFormatDepth32Float_Stencil8, kDrmr},
    Entry{Format::S8Uint, MTL::PixelFormatStencil8, kDrm},
    Entry{Format::Bc1Unorm, MTL::PixelFormatBC1_RGBA, kRf},
    Entry{Format::Bc1Srgb, MTL::PixelFormatBC1_RGBA_sRGB, kRf},
    Entry{Format::Bc2Unorm, MTL::PixelFormatBC2_RGBA, kRf},
    Entry{Format::Bc2Srgb, MTL::PixelFormatBC2_RGBA_sRGB, kRf},
    Entry{Format::Bc3Unorm, MTL::PixelFormatBC3_RGBA, kRf},
    Entry{Format::Bc3Srgb, MTL::PixelFormatBC3_RGBA_sRGB, kRf},
    Entry{Format::Bc4Unorm, MTL::PixelFormatBC4_RUnorm, kRf},
    Entry{Format::Bc4Snorm, MTL::PixelFormatBC4_RSnorm, kRf},
    Entry{Format::Bc5Unorm, MTL::PixelFormatBC5_RGUnorm, kRf},
    Entry{Format::Bc5Snorm, MTL::PixelFormatBC5_RGSnorm, kRf},
    Entry{Format::Bc6hUfloat, MTL::PixelFormatBC6H_RGBUfloat, kRf},
    Entry{Format::Bc6hSfloat, MTL::PixelFormatBC6H_RGBFloat, kRf},
    Entry{Format::Bc7Unorm, MTL::PixelFormatBC7_RGBAUnorm, kRf},
    Entry{Format::Bc7Srgb, MTL::PixelFormatBC7_RGBAUnorm_sRGB, kRf},
    Entry{Format::Etc2Rgb8Unorm, MTL::PixelFormatETC2_RGB8, kRf},
    Entry{Format::Etc2Rgb8Srgb, MTL::PixelFormatETC2_RGB8_sRGB, kRf},
    Entry{Format::Etc2Rgb8a1Unorm, MTL::PixelFormatETC2_RGB8A1, kRf},
    Entry{Format::Etc2Rgb8a1Srgb, MTL::PixelFormatETC2_RGB8A1_sRGB, kRf},
    Entry{Format::Etc2Rgba8Unorm, MTL::PixelFormatEAC_RGBA8, kRf},
    Entry{Format::Etc2Rgba8Srgb, MTL::PixelFormatEAC_RGBA8_sRGB, kRf},
    Entry{Format::EacR11Unorm, MTL::PixelFormatEAC_R11Unorm, kRf},
    Entry{Format::EacR11Snorm, MTL::PixelFormatEAC_R11Snorm, kRf},
    Entry{Format::EacRg11Unorm, MTL::PixelFormatEAC_RG11Unorm, kRf},
    Entry{Format::EacRg11Snorm, MTL::PixelFormatEAC_RG11Snorm, kRf},
    Entry{Format::Astc4x4Unorm, MTL::PixelFormatASTC_4x4_LDR, kRf},
    Entry{Format::Astc4x4Srgb, MTL::PixelFormatASTC_4x4_sRGB, kRf},
    Entry{Format::Astc4x4Float, MTL::PixelFormatASTC_4x4_HDR, kRf},
    Entry{Format::Astc5x4Unorm, MTL::PixelFormatASTC_5x4_LDR, kRf},
    Entry{Format::Astc5x4Srgb, MTL::PixelFormatASTC_5x4_sRGB, kRf},
    Entry{Format::Astc5x4Float, MTL::PixelFormatASTC_5x4_HDR, kRf},
    Entry{Format::Astc5x5Unorm, MTL::PixelFormatASTC_5x5_LDR, kRf},
    Entry{Format::Astc5x5Srgb, MTL::PixelFormatASTC_5x5_sRGB, kRf},
    Entry{Format::Astc5x5Float, MTL::PixelFormatASTC_5x5_HDR, kRf},
    Entry{Format::Astc6x5Unorm, MTL::PixelFormatASTC_6x5_LDR, kRf},
    Entry{Format::Astc6x5Srgb, MTL::PixelFormatASTC_6x5_sRGB, kRf},
    Entry{Format::Astc6x5Float, MTL::PixelFormatASTC_6x5_HDR, kRf},
    Entry{Format::Astc6x6Unorm, MTL::PixelFormatASTC_6x6_LDR, kRf},
    Entry{Format::Astc6x6Srgb, MTL::PixelFormatASTC_6x6_sRGB, kRf},
    Entry{Format::Astc6x6Float, MTL::PixelFormatASTC_6x6_HDR, kRf},
    Entry{Format::Astc8x5Unorm, MTL::PixelFormatASTC_8x5_LDR, kRf},
    Entry{Format::Astc8x5Srgb, MTL::PixelFormatASTC_8x5_sRGB, kRf},
    Entry{Format::Astc8x5Float, MTL::PixelFormatASTC_8x5_HDR, kRf},
    Entry{Format::Astc8x6Unorm, MTL::PixelFormatASTC_8x6_LDR, kRf},
    Entry{Format::Astc8x6Srgb, MTL::PixelFormatASTC_8x6_sRGB, kRf},
    Entry{Format::Astc8x6Float, MTL::PixelFormatASTC_8x6_HDR, kRf},
    Entry{Format::Astc8x8Unorm, MTL::PixelFormatASTC_8x8_LDR, kRf},
    Entry{Format::Astc8x8Srgb, MTL::PixelFormatASTC_8x8_sRGB, kRf},
    Entry{Format::Astc8x8Float, MTL::PixelFormatASTC_8x8_HDR, kRf},
    Entry{Format::Astc10x5Unorm, MTL::PixelFormatASTC_10x5_LDR, kRf},
    Entry{Format::Astc10x5Srgb, MTL::PixelFormatASTC_10x5_sRGB, kRf},
    Entry{Format::Astc10x5Float, MTL::PixelFormatASTC_10x5_HDR, kRf},
    Entry{Format::Astc10x6Unorm, MTL::PixelFormatASTC_10x6_LDR, kRf},
    Entry{Format::Astc10x6Srgb, MTL::PixelFormatASTC_10x6_sRGB, kRf},
    Entry{Format::Astc10x6Float, MTL::PixelFormatASTC_10x6_HDR, kRf},
    Entry{Format::Astc10x8Unorm, MTL::PixelFormatASTC_10x8_LDR, kRf},
    Entry{Format::Astc10x8Srgb, MTL::PixelFormatASTC_10x8_sRGB, kRf},
    Entry{Format::Astc10x8Float, MTL::PixelFormatASTC_10x8_HDR, kRf},
    Entry{Format::Astc10x10Unorm, MTL::PixelFormatASTC_10x10_LDR, kRf},
    Entry{Format::Astc10x10Srgb, MTL::PixelFormatASTC_10x10_sRGB, kRf},
    Entry{Format::Astc10x10Float, MTL::PixelFormatASTC_10x10_HDR, kRf},
    Entry{Format::Astc12x10Unorm, MTL::PixelFormatASTC_12x10_LDR, kRf},
    Entry{Format::Astc12x10Srgb, MTL::PixelFormatASTC_12x10_sRGB, kRf},
    Entry{Format::Astc12x10Float, MTL::PixelFormatASTC_12x10_HDR, kRf},
    Entry{Format::Astc12x12Unorm, MTL::PixelFormatASTC_12x12_LDR, kRf},
    Entry{Format::Astc12x12Srgb, MTL::PixelFormatASTC_12x12_sRGB, kRf},
    Entry{Format::Astc12x12Float, MTL::PixelFormatASTC_12x12_HDR, kRf},
};

const Entry* find(Format format) {
    const auto* const found = std::ranges::find(kTable, format, &Entry::format);
    return found == kTable.end() ? nullptr : &*found;
}

bool isBc(Format format) {
    const auto value = static_cast<std::uint16_t>(format);
    return value >= static_cast<std::uint16_t>(Format::Bc1Unorm) &&
           value < static_cast<std::uint16_t>(Format::Etc2Rgb8Unorm);
}

bool isAstcFloat(Format format) {
    switch (format) {
        case Format::Astc4x4Float:
        case Format::Astc5x4Float:
        case Format::Astc5x5Float:
        case Format::Astc6x5Float:
        case Format::Astc6x6Float:
        case Format::Astc8x5Float:
        case Format::Astc8x6Float:
        case Format::Astc8x8Float:
        case Format::Astc10x5Float:
        case Format::Astc10x6Float:
        case Format::Astc10x8Float:
        case Format::Astc10x10Float:
        case Format::Astc12x10Float:
        case Format::Astc12x12Float:
            return true;
        default:
            return false;
    }
}

// What the device changes in the table, as Godot's modifyMTLFormatCapabilities
// does for Apple GPU family 7 and later.
Bits adjusted(MTL::Device* device, Format format, Bits bits) {
    const bool onMac = TARGET_OS_OSX != 0;
    const bool iosOnly8 = !onMac && !device->supportsFamily(MTL::GPUFamilyApple9);
    const bool msaa32 = device->supports32BitMSAA();
    const bool filter32 = device->supports32BitFloatFiltering();
    const auto set = [&bits](bool condition, Bits flags) {
        if (condition) {
            bits = static_cast<Bits>(bits | flags);
        }
    };
    const auto clear = [&bits](bool condition, Bits flags) {
        if (condition) {
            bits = static_cast<Bits>(bits & ~flags);
        }
    };
    switch (format) {
        case Format::R32Uint:
        case Format::R32Sint:
            set(device->supportsFamily(MTL::GPUFamilyApple6), Atomic);
            set(msaa32, Multisample);
            break;
        case Format::Rg32Uint:
        case Format::Rg32Sint:
        case Format::Rgba32Uint:
        case Format::Rgba32Sint:
            set(msaa32, Multisample);
            break;
        case Format::R32Float:
        case Format::Rg32Float:
        case Format::Rgba32Float:
            // Off a Mac, before Apple GPU family 9, 32-bit float is rendered to
            // without filtering or resolving, unless the device says it can.
            if (iosOnly8) {
                bits = format == Format::Rgba32Float ? kRwcm : kRwcmb;
            }
            set(filter32, Filter);
            set(msaa32, Resolve);
            set(filter32 && format == Format::Rgba32Float, Blend);
            break;
        case Format::Rgb9e5Float:
            // A Mac cannot enable channels one by one while it blends.
            clear(onMac, Blend);
            break;
        default:
            break;
    }
    // A format the device lacks has no capability at all.
    if ((isBc(format) && !device->supportsBCTextureCompression()) ||
        (isAstcFloat(format) && !device->supportsFamily(MTL::GPUFamilyApple6))) {
        return 0;
    }
    return bits;
}

// The sample counts a format renders at: every count the device takes for a
// format that multisamples, 1 for one that only renders, none otherwise.
std::uint32_t sampleCountsFor(MTL::Device* device, Bits bits) {
    if ((bits & (ColorAttachment | DepthStencilAttachment)) == 0) {
        return 0;
    }
    std::uint32_t counts = 1;
    if ((bits & Multisample) != 0) {
        for (const std::uint32_t count : {2U, 4U, 8U}) {
            if (device->supportsTextureSampleCount(count)) {
                counts |= count;
            }
        }
    }
    return counts;
}

constexpr std::array kSurfaceFormats{
    SurfaceFormat{.format = Format::Bgra8Unorm, .colorSpace = ColorSpace::Srgb},
    SurfaceFormat{.format = Format::Rgba16Float, .colorSpace = ColorSpace::SrgbExtendedLinear},
};

} // namespace

MTL::PixelFormat pixelFormatFor(Format format) {
    const Entry* entry = find(format);
    return entry != nullptr ? entry->pixelFormat : MTL::PixelFormatInvalid;
}

FormatCaps formatCapsFor(MTL::Device* device, Format format) {
    FormatCaps caps{.format = format};
    const Entry* entry = find(format);
    if (entry == nullptr) {
        return caps;
    }
    const Bits bits = adjusted(device, format, entry->bits);
    if (bits == 0) {
        return caps;
    }
    caps.sampled = (bits & Read) != 0;
    caps.filterLinear = (bits & Filter) != 0;
    caps.storage = (bits & Write) != 0;
    caps.storageAtomic = (bits & Atomic) != 0;
    caps.colorAttachment = (bits & ColorAttachment) != 0;
    caps.blend = (bits & Blend) != 0;
    caps.depthStencil = (bits & DepthStencilAttachment) != 0;
    caps.sampleCounts = sampleCountsFor(device, bits);
    return caps;
}

SurfaceCaps surfaceCapsFor(std::uint32_t maxTexture2d) {
    // A layer holds at most 3 drawables; 2 is the fewest BFR asks of it.
    return {.minImageCount = 2,
            .maxImageCount = 3,
            .width = kSizeFromSwapchain,
            .height = kSizeFromSwapchain,
            .minWidth = 1,
            .minHeight = 1,
            .maxWidth = maxTexture2d,
            .maxHeight = maxTexture2d,
            .rotation = SurfaceRotation::None,
            .fifo = true,
            .fifoRelaxed = false,
            .mailbox = false,
            // Display sync is a Mac layer's switch; other layers always wait.
            .immediate = TARGET_OS_OSX != 0,
            .alphaOpaque = true,
            .alphaPremultiplied = false,
            .alphaPostmultiplied = false,
            .alphaInherit = false};
}

std::span<const SurfaceFormat> surfaceFormats() {
    return kSurfaceFormats;
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
