#pragma once

// What D3D12 says about formats and about a DXGI swapchain as a surface.

#include "bfr/core/error.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/format.h"

#include <d3d12.h>
#include <dxgi1_6.h>

#include <cstdint>
#include <span>

namespace bfr::rhi::d3d12 {

/// The DXGI formats a BFR format is read and written as. DXGI_FORMAT_UNKNOWN
/// where DXGI has none.
struct DxgiFormats {
    /// What a shader reads, a color attachment writes and a swapchain holds.
    DXGI_FORMAT texture = DXGI_FORMAT_UNKNOWN;
    /// What a depth-stencil view is made as; unknown for a color format.
    DXGI_FORMAT depth = DXGI_FORMAT_UNKNOWN;
};

[[nodiscard]] DxgiFormats dxgiFormatsFor(Format format);

/// What device does with format: every field false and no sample counts for a
/// format DXGI lacks. Fails with RhiQueryFailed when the device won't say.
[[nodiscard]] core::Result<FormatCaps> formatCapsFor(ID3D12Device* device, Format format);

/// What a swapchain may be on a device whose largest 2D texture is maxTexture2d
/// pixels, with tearing when DXGI allows it.
[[nodiscard]] SurfaceCaps surfaceCapsFor(std::uint32_t maxTexture2d, bool tearing);

/// The formats and color spaces a DXGI swapchain presents in, BFR's names only.
[[nodiscard]] std::span<const SurfaceFormat> surfaceFormats();

/// How a swapchain is made to present format in colorSpace.
struct SwapchainFormat {
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    DXGI_COLOR_SPACE_TYPE colorSpace = DXGI_COLOR_SPACE_RESERVED;
};

/// The swapchain setup for a pair surfaceFormats() lists; unset for any other.
[[nodiscard]] SwapchainFormat swapchainFormatFor(Format format, ColorSpace colorSpace);

} // namespace bfr::rhi::d3d12
