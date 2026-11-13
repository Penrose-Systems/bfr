#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/texture.h"
#include "rhi/enum_check.h"
#include "rhi/format_blocks.h"

#include <cstdint>
#include <span>

namespace bfr::rhi {

/// Fails with RhiInvalidCall when aspect is not one BFR has, or one format has
/// none of: Color of a depth or stencil format, or Depth or Stencil of one
/// without it. D3D12's and Metal's tiling queries call it: an aspect the
/// format lacks has no tiling to report.
[[nodiscard]] inline core::Result<void> checkAspect(Format format, TextureAspect aspect) {
    if (auto checked = checkKnown(aspect, "texture aspect"); !checked) {
        return checked;
    }
    const FormatBlock& block = blockOf(format);
    const bool has = aspect == TextureAspect::Depth     ? block.depthBytes != 0
                     : aspect == TextureAspect::Stencil ? block.stencilBytes != 0
                                                        : block.bytes != 0;
    if (!has) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "format {} has no aspect {}",
                             static_cast<std::uint32_t>(format), static_cast<std::uint32_t>(aspect));
    }
    return {};
}

/// Fails with RhiInvalidCall when a region's aspect is not one BFR has. Each
/// backend's buffer-texture copies call it before recording.
[[nodiscard]] inline core::Result<void> checkAspects(std::span<const BufferTextureCopy> regions) {
    for (const BufferTextureCopy& region : regions) {
        if (auto checked = checkKnown(region.aspect, "texture aspect"); !checked) {
            return checked;
        }
    }
    return {};
}

} // namespace bfr::rhi
