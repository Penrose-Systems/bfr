#pragma once

#include "bfr/rhi/swapchain.h"

#include <volk.h>

#include <cstdint>

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

/// desc in Vulkan's names, each value as given. size is the window's size in
/// pixels, which the surface's current extent overrides when it reports one.
[[nodiscard]] SwapchainSettings swapchainSettingsFor(const VkSurfaceCapabilitiesKHR& capabilities, VkExtent2D size,
                                                     const SwapchainDesc& desc);

} // namespace bfr::rhi::vulkan
