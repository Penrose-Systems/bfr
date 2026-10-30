#pragma once

// What a swapchain is built with once its surface has allowed the request.
// Every backend refuses a request the same way, with the same words, from
// what its surface reports as SurfaceCaps and SurfaceFormats.

#include "bfr/core/error.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/swapchain.h"

#include <cstdint>
#include <span>

namespace bfr::rhi {

/// SurfaceCaps::width and height when the swapchain's size sets the window's.
inline constexpr std::uint32_t kSizeFromSwapchain = 0xFFFFFFFF;

/// A size in pixels.
struct Extent {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

/// What a surface builds a request as. A zero extent means the window has no
/// area, and nothing can be built.
struct SwapchainBuild {
    std::uint32_t imageCount = 0;
    Extent extent;
};

/// request as a surface with caps and formats would build it, or why it won't:
/// RhiUnsupported, with a message that says what the surface allows. size is
/// the window's size in pixels, which the surface's own size overrides when it
/// reports one; a size the surface can't take is refused, not clamped.
[[nodiscard]] core::Result<SwapchainBuild> swapchainBuildFor(const SurfaceCaps& caps,
                                                             std::span<const SurfaceFormat> formats, Extent size,
                                                             const SwapchainRequest& request);

} // namespace bfr::rhi
