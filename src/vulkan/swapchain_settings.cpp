#include "swapchain_settings.h"

#include "error.h"
#include "swapchain.h"
#include "vulkan_caps.h"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cstdint>
#include <format>
#include <limits>
#include <span>
#include <string>

namespace bfr::rhi::vulkan {

namespace {

// A surface reports this extent when the swapchain's extent decides its size,
// as Wayland's and headless surfaces do.
constexpr std::uint32_t kExtentFromSwapchain = std::numeric_limits<std::uint32_t>::max();

core::Result<VkExtent2D> extentFor(const VkSurfaceCapabilitiesKHR& capabilities, VkExtent2D size) {
    if (capabilities.currentExtent.width != kExtentFromSwapchain) {
        return capabilities.currentExtent;
    }
    if (size.width == 0 || size.height == 0) {
        return VkExtent2D{.width = 0, .height = 0};
    }
    const VkExtent2D least = capabilities.minImageExtent;
    const VkExtent2D most = capabilities.maxImageExtent;
    if (size.width < least.width || size.height < least.height || size.width > most.width ||
        size.height > most.height) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             std::format("the window allows {}x{} to {}x{} pixels; {}x{} were asked", least.width,
                                         least.height, most.width, most.height, size.width, size.height));
    }
    return size;
}

core::Result<std::uint32_t> imageCountFor(const VkSurfaceCapabilitiesKHR& capabilities, std::uint32_t images) {
    if (images == 0) {
        return capabilities.minImageCount;
    }
    // A maximum of 0 means there is none.
    if (images >= capabilities.minImageCount &&
        (capabilities.maxImageCount == 0 || images <= capabilities.maxImageCount)) {
        return images;
    }
    const std::string allowed = capabilities.maxImageCount == 0
                                    ? std::format("{} or more", capabilities.minImageCount)
                                    : std::format("{} to {}", capabilities.minImageCount, capabilities.maxImageCount);
    return core::failure(core::ErrorCode::RhiUnsupported,
                         std::format("the window allows {} images; {} were asked", allowed, images));
}

core::Result<VkSurfaceFormatKHR> formatFor(std::span<const VkSurfaceFormatKHR> formats,
                                           const SwapchainRequest& request) {
    const VkSurfaceFormatKHR wanted{.format = vulkanFormat(request.format),
                                    .colorSpace = vulkanColorSpace(request.colorSpace)};
    const bool listed = std::ranges::any_of(formats, [&wanted](const VkSurfaceFormatKHR& format) {
        return format.format == wanted.format && format.colorSpace == wanted.colorSpace;
    });
    if (!listed) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the window does not list the format with the color space that were asked");
    }
    return wanted;
}

} // namespace

VkPresentModeKHR presentModeFor(PresentMode mode) {
    switch (mode) {
        case PresentMode::Fifo:
            return VK_PRESENT_MODE_FIFO_KHR;
        case PresentMode::FifoRelaxed:
            return VK_PRESENT_MODE_FIFO_RELAXED_KHR;
        case PresentMode::Mailbox:
            return VK_PRESENT_MODE_MAILBOX_KHR;
        case PresentMode::Immediate:
            return VK_PRESENT_MODE_IMMEDIATE_KHR;
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkCompositeAlphaFlagBitsKHR compositeAlphaFor(AlphaMode mode) {
    switch (mode) {
        case AlphaMode::Opaque:
            return VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        case AlphaMode::Premultiplied:
            return VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR;
        case AlphaMode::Postmultiplied:
            return VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR;
        case AlphaMode::Inherit:
            return VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
    }
    return VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
}

core::Result<SwapchainSettings> swapchainSettingsFor(const VkSurfaceCapabilitiesKHR& capabilities,
                                                     std::span<const VkSurfaceFormatKHR> formats,
                                                     std::span<const VkPresentModeKHR> presentModes, VkExtent2D size,
                                                     const SwapchainRequest& request) {
    const auto imageCount = imageCountFor(capabilities, request.imageCount);
    if (!imageCount) {
        return std::unexpected(imageCount.error());
    }
    const auto format = formatFor(formats, request);
    if (!format) {
        return std::unexpected(format.error());
    }
    const VkPresentModeKHR presentMode = presentModeFor(request.presentMode);
    if (std::ranges::find(presentModes, presentMode) == presentModes.end()) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             std::format("the window does not allow {} presenting", name(request.presentMode)));
    }
    const VkCompositeAlphaFlagBitsKHR compositeAlpha = compositeAlphaFor(request.alphaMode);
    if ((capabilities.supportedCompositeAlpha & static_cast<VkCompositeAlphaFlagsKHR>(compositeAlpha)) == 0) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             std::format("the window does not allow {} alpha", name(request.alphaMode)));
    }
    const auto extent = extentFor(capabilities, size);
    if (!extent) {
        return std::unexpected(extent.error());
    }
    return SwapchainSettings{.format = *format,
                             .extent = *extent,
                             .imageCount = *imageCount,
                             .compositeAlpha = compositeAlpha,
                             .presentMode = presentMode};
}

} // namespace bfr::rhi::vulkan
