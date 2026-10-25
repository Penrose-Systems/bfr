#pragma once

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
};

/// Chooses each setting from what the surface supports. formats is the
/// surface's format list, never empty. size is the window's size in pixels,
/// which the surface's own extent overrides when it reports one. images is
/// the image count asked for.
[[nodiscard]] SwapchainSettings chooseSwapchainSettings(const VkSurfaceCapabilitiesKHR& capabilities,
                                                        std::span<const VkSurfaceFormatKHR> formats, VkExtent2D size,
                                                        std::uint32_t images);

} // namespace bfr::rhi::vulkan
