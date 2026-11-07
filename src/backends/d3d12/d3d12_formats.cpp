#include "d3d12/d3d12_formats.h"

#include "bfr/core/error.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/format.h"
#include "d3d12/d3d12_result.h"

#include <d3d12.h>
#include <dxgicommon.h>
#include <dxgiformat.h>
#include <windows.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

namespace bfr::rhi::d3d12 {

namespace {

// The DXGI formats of each BFR format DXGI has one for. Bgr10a2Unorm, the
// 64-bit formats, S8Uint, ETC2, EAC and ASTC are left out: DXGI has no format
// for them.
struct Entry {
    Format format;
    DxgiFormats formats;

    constexpr Entry(Format format, DXGI_FORMAT texture, DXGI_FORMAT depth = DXGI_FORMAT_UNKNOWN,
                    DXGI_FORMAT typeless = DXGI_FORMAT_UNKNOWN)
        : format(format), formats{.texture = texture, .depth = depth, .typeless = typeless} {}
};

constexpr std::array kEntries{
    Entry{Format::R8Unorm, DXGI_FORMAT_R8_UNORM},
    Entry{Format::R8Snorm, DXGI_FORMAT_R8_SNORM},
    Entry{Format::R8Uint, DXGI_FORMAT_R8_UINT},
    Entry{Format::R8Sint, DXGI_FORMAT_R8_SINT},
    Entry{Format::Rg8Unorm, DXGI_FORMAT_R8G8_UNORM},
    Entry{Format::Rg8Snorm, DXGI_FORMAT_R8G8_SNORM},
    Entry{Format::Rg8Uint, DXGI_FORMAT_R8G8_UINT},
    Entry{Format::Rg8Sint, DXGI_FORMAT_R8G8_SINT},
    Entry{Format::Rgba8Unorm, DXGI_FORMAT_R8G8B8A8_UNORM},
    Entry{Format::Rgba8Srgb, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB},
    Entry{Format::Rgba8Snorm, DXGI_FORMAT_R8G8B8A8_SNORM},
    Entry{Format::Rgba8Uint, DXGI_FORMAT_R8G8B8A8_UINT},
    Entry{Format::Rgba8Sint, DXGI_FORMAT_R8G8B8A8_SINT},
    Entry{Format::Bgra8Unorm, DXGI_FORMAT_B8G8R8A8_UNORM},
    Entry{Format::Bgra8Srgb, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB},
    Entry{Format::R16Unorm, DXGI_FORMAT_R16_UNORM},
    Entry{Format::R16Snorm, DXGI_FORMAT_R16_SNORM},
    Entry{Format::R16Uint, DXGI_FORMAT_R16_UINT},
    Entry{Format::R16Sint, DXGI_FORMAT_R16_SINT},
    Entry{Format::R16Float, DXGI_FORMAT_R16_FLOAT},
    Entry{Format::Rg16Unorm, DXGI_FORMAT_R16G16_UNORM},
    Entry{Format::Rg16Snorm, DXGI_FORMAT_R16G16_SNORM},
    Entry{Format::Rg16Uint, DXGI_FORMAT_R16G16_UINT},
    Entry{Format::Rg16Sint, DXGI_FORMAT_R16G16_SINT},
    Entry{Format::Rg16Float, DXGI_FORMAT_R16G16_FLOAT},
    Entry{Format::Rgba16Unorm, DXGI_FORMAT_R16G16B16A16_UNORM},
    Entry{Format::Rgba16Snorm, DXGI_FORMAT_R16G16B16A16_SNORM},
    Entry{Format::Rgba16Uint, DXGI_FORMAT_R16G16B16A16_UINT},
    Entry{Format::Rgba16Sint, DXGI_FORMAT_R16G16B16A16_SINT},
    Entry{Format::Rgba16Float, DXGI_FORMAT_R16G16B16A16_FLOAT},
    Entry{Format::R32Uint, DXGI_FORMAT_R32_UINT},
    Entry{Format::R32Sint, DXGI_FORMAT_R32_SINT},
    Entry{Format::R32Float, DXGI_FORMAT_R32_FLOAT},
    Entry{Format::Rg32Uint, DXGI_FORMAT_R32G32_UINT},
    Entry{Format::Rg32Sint, DXGI_FORMAT_R32G32_SINT},
    Entry{Format::Rg32Float, DXGI_FORMAT_R32G32_FLOAT},
    Entry{Format::Rgb32Uint, DXGI_FORMAT_R32G32B32_UINT},
    Entry{Format::Rgb32Sint, DXGI_FORMAT_R32G32B32_SINT},
    Entry{Format::Rgb32Float, DXGI_FORMAT_R32G32B32_FLOAT},
    Entry{Format::Rgba32Uint, DXGI_FORMAT_R32G32B32A32_UINT},
    Entry{Format::Rgba32Sint, DXGI_FORMAT_R32G32B32A32_SINT},
    Entry{Format::Rgba32Float, DXGI_FORMAT_R32G32B32A32_FLOAT},
    Entry{Format::Rgb10a2Unorm, DXGI_FORMAT_R10G10B10A2_UNORM},
    Entry{Format::Rgb10a2Uint, DXGI_FORMAT_R10G10B10A2_UINT},
    Entry{Format::Rg11b10Float, DXGI_FORMAT_R11G11B10_FLOAT},
    Entry{Format::Rgb9e5Float, DXGI_FORMAT_R9G9B9E5_SHAREDEXP},
    Entry{Format::B5g6r5Unorm, DXGI_FORMAT_B5G6R5_UNORM},
    Entry{Format::Bgr5a1Unorm, DXGI_FORMAT_B5G5R5A1_UNORM},
    Entry{Format::Bgra4Unorm, DXGI_FORMAT_B4G4R4A4_UNORM},
    Entry{Format::D16Unorm, DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_D16_UNORM, DXGI_FORMAT_R16_TYPELESS},
    Entry{Format::D24UnormS8Uint, DXGI_FORMAT_R24_UNORM_X8_TYPELESS, DXGI_FORMAT_D24_UNORM_S8_UINT,
          DXGI_FORMAT_R24G8_TYPELESS},
    Entry{Format::D32Float, DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R32_TYPELESS},
    Entry{Format::D32FloatS8Uint, DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS, DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
          DXGI_FORMAT_R32G8X24_TYPELESS},
    Entry{Format::Bc1Unorm, DXGI_FORMAT_BC1_UNORM},
    Entry{Format::Bc1Srgb, DXGI_FORMAT_BC1_UNORM_SRGB},
    Entry{Format::Bc2Unorm, DXGI_FORMAT_BC2_UNORM},
    Entry{Format::Bc2Srgb, DXGI_FORMAT_BC2_UNORM_SRGB},
    Entry{Format::Bc3Unorm, DXGI_FORMAT_BC3_UNORM},
    Entry{Format::Bc3Srgb, DXGI_FORMAT_BC3_UNORM_SRGB},
    Entry{Format::Bc4Unorm, DXGI_FORMAT_BC4_UNORM},
    Entry{Format::Bc4Snorm, DXGI_FORMAT_BC4_SNORM},
    Entry{Format::Bc5Unorm, DXGI_FORMAT_BC5_UNORM},
    Entry{Format::Bc5Snorm, DXGI_FORMAT_BC5_SNORM},
    Entry{Format::Bc6hUfloat, DXGI_FORMAT_BC6H_UF16},
    Entry{Format::Bc6hSfloat, DXGI_FORMAT_BC6H_SF16},
    Entry{Format::Bc7Unorm, DXGI_FORMAT_BC7_UNORM},
    Entry{Format::Bc7Srgb, DXGI_FORMAT_BC7_UNORM_SRGB},
};

// Each way a format is checked for: the Support1 and Support2 bits a query of
// the device returns.
struct Support {
    UINT support1 = 0;
    UINT support2 = 0;
};

core::Result<Support> supportOf(ID3D12Device* device, DXGI_FORMAT format) {
    D3D12_FEATURE_DATA_FORMAT_SUPPORT data{.Format = format,
                                           .Support1 = D3D12_FORMAT_SUPPORT1_NONE,
                                           .Support2 = D3D12_FORMAT_SUPPORT2_NONE};
    if (const HRESULT result = device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &data, sizeof(data));
        FAILED(result)) {
        return failure(core::ErrorCode::RhiQueryFailed,
                       {.call = "ID3D12Device::CheckFeatureSupport(FORMAT_SUPPORT)", .result = result});
    }
    return Support{.support1 = static_cast<UINT>(data.Support1), .support2 = static_cast<UINT>(data.Support2)};
}

bool has(UINT bits, UINT flag) {
    return (bits & flag) != 0;
}

// D3D12_FORMAT_SUPPORT1 has no copy bit: CopyTextureRegion and CopyResource
// take any texture the device makes, so a format copies wherever the device
// makes a texture of it.
constexpr UINT kTextureBits = static_cast<UINT>(D3D12_FORMAT_SUPPORT1_TEXTURE1D) |
                              static_cast<UINT>(D3D12_FORMAT_SUPPORT1_TEXTURE2D) |
                              static_cast<UINT>(D3D12_FORMAT_SUPPORT1_TEXTURE3D) |
                              static_cast<UINT>(D3D12_FORMAT_SUPPORT1_TEXTURECUBE);

// Each sample count the format renders at, as bits whose values are the
// counts; a count the device won't take is left out, whatever the reason.
std::uint32_t sampleCountsFor(ID3D12Device* device, DXGI_FORMAT format) {
    std::uint32_t counts = 0;
    // Every power of two up to the most D3D12 defines, each asked.
    for (UINT count = 1; count <= D3D12_MAX_MULTISAMPLE_SAMPLE_COUNT; count *= 2) {
        D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS levels{.Format = format,
                                                             .SampleCount = count,
                                                             .Flags = D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE,
                                                             .NumQualityLevels = 0};
        if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &levels, sizeof(levels))) &&
            levels.NumQualityLevels != 0) {
            counts |= count;
        }
    }
    return counts;
}

// Each format and color space a swapchain presents in, and how DXGI is asked
// for it.
struct SwapchainEntry {
    SwapchainFormat format;
    DxgiSwapchainFormat dxgi;
};

constexpr std::array kSwapchainEntries{
    SwapchainEntry{.format = {.format = Format::Bgra8Unorm, .colorSpace = ColorSpace::Srgb},
                   .dxgi = {.format = DXGI_FORMAT_B8G8R8A8_UNORM,
                            .colorSpace = DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709}},
    SwapchainEntry{.format = {.format = Format::Rgba8Unorm, .colorSpace = ColorSpace::Srgb},
                   .dxgi = {.format = DXGI_FORMAT_R8G8B8A8_UNORM,
                            .colorSpace = DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709}},
    SwapchainEntry{.format = {.format = Format::Rgba16Float, .colorSpace = ColorSpace::SrgbExtendedLinear},
                   .dxgi = {.format = DXGI_FORMAT_R16G16B16A16_FLOAT,
                            .colorSpace = DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709}},
};

} // namespace

DxgiFormats dxgiFormatsFor(Format format) {
    const auto* const found = std::ranges::find(kEntries, format, &Entry::format);
    return found != kEntries.end() ? found->formats : DxgiFormats{};
}

core::Result<FormatCaps> formatCapsFor(ID3D12Device* device, Format format) {
    FormatCaps caps{.format = format};
    const DxgiFormats formats = dxgiFormatsFor(format);
    // DXGI_FORMAT_UNKNOWN is not queried: some implementations fail it
    // instead of answering that it supports nothing.
    if (formats.texture != DXGI_FORMAT_UNKNOWN) {
        const auto support = supportOf(device, formats.texture);
        if (!support) {
            return std::unexpected(support.error());
        }
        caps.sampled = has(support->support1, D3D12_FORMAT_SUPPORT1_SHADER_LOAD) ||
                       has(support->support1, D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE);
        caps.filterLinear = has(support->support1, D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE);
        caps.storage = has(support->support1, D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW);
        caps.storageAtomic = has(support->support2, D3D12_FORMAT_SUPPORT2_UAV_ATOMIC_ADD);
        caps.colorAttachment = has(support->support1, D3D12_FORMAT_SUPPORT1_RENDER_TARGET);
        caps.blend = has(support->support1, D3D12_FORMAT_SUPPORT1_BLENDABLE);
        caps.copySrc = has(support->support1, kTextureBits);
        caps.copyDst = caps.copySrc;
        if (caps.colorAttachment) {
            caps.sampleCounts = sampleCountsFor(device, formats.texture);
        }
    }
    if (formats.depth != DXGI_FORMAT_UNKNOWN) {
        const auto support = supportOf(device, formats.depth);
        if (!support) {
            return std::unexpected(support.error());
        }
        caps.depthStencil = has(support->support1, D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL);
        // A depth texture is made as its depth format, not its shader one.
        caps.copySrc = has(support->support1, kTextureBits);
        caps.copyDst = caps.copySrc;
        if (caps.depthStencil) {
            caps.sampleCounts = sampleCountsFor(device, formats.depth);
        }
    }
    return caps;
}

core::Result<std::vector<SwapchainFormat>> swapchainFormats(ID3D12Device* device) {
    std::vector<SwapchainFormat> formats;
    for (const SwapchainEntry& entry : kSwapchainEntries) {
        const auto support = supportOf(device, entry.dxgi.format);
        if (!support) {
            return std::unexpected(support.error());
        }
        if (has(support->support1, D3D12_FORMAT_SUPPORT1_DISPLAY)) {
            formats.push_back(entry.format);
        }
    }
    return formats;
}

DxgiSwapchainFormat swapchainFormatFor(Format format, ColorSpace colorSpace) {
    const auto asked = [format, colorSpace](const SwapchainEntry& entry) {
        return entry.format.format == format && entry.format.colorSpace == colorSpace;
    };
    const auto* const found = std::ranges::find_if(kSwapchainEntries, asked);
    return found != kSwapchainEntries.end() ? found->dxgi : DxgiSwapchainFormat{};
}

Format swapchainFormatOf(DXGI_FORMAT format) {
    const auto* const found = std::ranges::find(kSwapchainEntries, format,
                                                [](const SwapchainEntry& entry) { return entry.dxgi.format; });
    return found != kSwapchainEntries.end() ? found->format.format : Format::Undefined;
}

} // namespace bfr::rhi::d3d12
