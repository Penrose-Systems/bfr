#pragma once

// What D3D12 says about formats, and the formats a DXGI swapchain presents in.

#include "bfr/core/error.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/sparse.h"
#include "bfr/rhi/texture.h"

#include <d3d12.h>
#include <dxgi1_6.h>

#include <cstdint>
#include <vector>

namespace bfr::rhi::d3d12 {

/// The DXGI formats a BFR format is read and written as. DXGI_FORMAT_UNKNOWN
/// where DXGI has none.
struct DxgiFormats {
    /// What a shader reads, a color attachment writes and a swapchain holds.
    DXGI_FORMAT texture = DXGI_FORMAT_UNKNOWN;
    /// What a depth-stencil view is made as; unknown for a color format.
    DXGI_FORMAT depth = DXGI_FORMAT_UNKNOWN;
    /// What a depth texture's resource is made as when shaders read it too:
    /// typeless, so its depth view and its shader view may both be made of it.
    DXGI_FORMAT typeless = DXGI_FORMAT_UNKNOWN;
};

[[nodiscard]] DxgiFormats dxgiFormatsFor(Format format);

/// What device does with format: every field false and no sample counts for a
/// format DXGI lacks. Fails with RhiQueryFailed when the device won't say.
[[nodiscard]] core::Result<FormatCaps> formatCapsFor(ID3D12Device* device, Format format);

/// Whether device makes reserved resources of format with samples samples:
/// FORMAT_SUPPORT's D3D12_FORMAT_SUPPORT2_TILED, and above one sample
/// MULTISAMPLE_QUALITY_LEVELS asked with
/// D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_TILED_RESOURCE. Fails with
/// RhiQueryFailed when the device won't answer FORMAT_SUPPORT.
[[nodiscard]] core::Result<bool> tiledFor(ID3D12Device* device, DXGI_FORMAT format, std::uint32_t samples);

/// The standard tile shape Microsoft's documentation gives a texture of
/// format, dimension and samples, all 0 where it gives none.
[[nodiscard]] TileShape standardTileShape(Format format, TextureDimension dimension, std::uint32_t samples);

/// How a swapchain is made to present format in colorSpace.
struct DxgiSwapchainFormat {
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    DXGI_COLOR_SPACE_TYPE colorSpace = DXGI_COLOR_SPACE_RESERVED;
};

/// The formats and color spaces swapchainFormatFor has a setting for, of
/// those device says it displays (FORMAT_SUPPORT1_DISPLAY). DXGI has no call
/// that lists them. Fails with RhiQueryFailed when the query fails.
[[nodiscard]] core::Result<std::vector<SwapchainFormat>> swapchainFormats(ID3D12Device* device);

/// How DXGI is asked to present format in colorSpace; unset for a pair it has
/// no setting for.
[[nodiscard]] DxgiSwapchainFormat swapchainFormatFor(Format format, ColorSpace colorSpace);

/// The format of a swapchain DXGI made in format; Undefined for one no
/// swapchainFormats entry is made in.
[[nodiscard]] Format swapchainFormatOf(DXGI_FORMAT format);

} // namespace bfr::rhi::d3d12
