#include "swapchain_settings.h"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <span>

namespace bfr::rhi::vulkan {

namespace {

// Ordered by preference: 8-bit, with the shaders encoding sRGB, so a cleared
// or read-back value is exactly the value written.
constexpr std::array kPreferredFormats{
    VkSurfaceFormatKHR{.format = VK_FORMAT_B8G8R8A8_UNORM, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR},
    VkSurfaceFormatKHR{.format = VK_FORMAT_R8G8B8A8_UNORM, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR},
};

// Opaque is not guaranteed; a surface supports at least one of these.
constexpr std::array kCompositeAlphas{
    VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
    VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
    VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
    VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
};

// One image shows while the next is drawn.
constexpr std::uint32_t kMinImages = 2;

// A surface reports this extent when the swapchain's extent decides its size,
// as Wayland's and headless surfaces do.
constexpr std::uint32_t kExtentFromSwapchain = std::numeric_limits<std::uint32_t>::max();

VkSurfaceFormatKHR chooseFormat(std::span<const VkSurfaceFormatKHR> formats) {
    for (const VkSurfaceFormatKHR& preferred : kPreferredFormats) {
        if (std::ranges::any_of(formats, [&preferred](const VkSurfaceFormatKHR& format) {
                return format.format == preferred.format && format.colorSpace == preferred.colorSpace;
            })) {
            return preferred;
        }
    }
    return formats.front();
}

VkExtent2D chooseExtent(const VkSurfaceCapabilitiesKHR& capabilities, VkExtent2D size) {
    if (capabilities.currentExtent.width != kExtentFromSwapchain) {
        return capabilities.currentExtent;
    }
    if (size.width == 0 || size.height == 0) {
        return {.width = 0, .height = 0};
    }
    return {
        .width = std::clamp(size.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
        .height = std::clamp(size.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height),
    };
}

std::uint32_t chooseImageCount(const VkSurfaceCapabilitiesKHR& capabilities, std::uint32_t images) {
    const std::uint32_t count = std::max({images, kMinImages, capabilities.minImageCount});
    // A maximum of 0 means there is none.
    return capabilities.maxImageCount == 0 ? count : std::min(count, capabilities.maxImageCount);
}

VkCompositeAlphaFlagBitsKHR chooseCompositeAlpha(VkCompositeAlphaFlagsKHR supported) {
    const auto* found = std::ranges::find_if(kCompositeAlphas, [supported](VkCompositeAlphaFlagBitsKHR alpha) {
        return (supported & static_cast<VkCompositeAlphaFlagsKHR>(alpha)) != 0;
    });
    return found == kCompositeAlphas.end() ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR : *found;
}

} // namespace

SwapchainSettings chooseSwapchainSettings(const VkSurfaceCapabilitiesKHR& capabilities,
                                          std::span<const VkSurfaceFormatKHR> formats, VkExtent2D size,
                                          std::uint32_t images) {
    return {
        .format = chooseFormat(formats),
        .extent = chooseExtent(capabilities, size),
        .imageCount = chooseImageCount(capabilities, images),
        .compositeAlpha = chooseCompositeAlpha(capabilities.supportedCompositeAlpha),
    };
}

} // namespace bfr::rhi::vulkan
