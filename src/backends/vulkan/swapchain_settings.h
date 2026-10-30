#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/swapchain.h"

#include <volk.h>

#include <cstdint>
#include <span>

namespace bfr::rhi::vulkan {

/// What a swapchain is built with. A zero extent means the window has no
/// area, and nothing can be built.
struct SwapchainSettings {
    VkSurfaceFormatKHR format;
    VkExtent2D extent;
    std::uint32_t imageCount;
    VkCompositeAlphaFlagBitsKHR compositeAlpha;
    VkPresentModeKHR presentMode;
};

/// The Vulkan present mode that is mode.
[[nodiscard]] VkPresentModeKHR presentModeFor(PresentMode mode);

/// The Vulkan composite alpha that is mode.
[[nodiscard]] VkCompositeAlphaFlagBitsKHR compositeAlphaFor(AlphaMode mode);

/// request as the surface would build it, or why it won't, as
/// rhi::swapchainBuildFor decides for every backend: RhiUnsupported, with a
/// message that says what the surface allows. formats and presentModes are
/// the surface's lists. size is the window's size in pixels, which the
/// surface's own extent overrides when it reports one.
[[nodiscard]] core::Result<SwapchainSettings> swapchainSettingsFor(const VkSurfaceCapabilitiesKHR& capabilities,
                                                                   std::span<const VkSurfaceFormatKHR> formats,
                                                                   std::span<const VkPresentModeKHR> presentModes,
                                                                   VkExtent2D size, const SwapchainRequest& request);

} // namespace bfr::rhi::vulkan
