#include "vulkan/swapchain_settings.h"

#include "bfr/core/error.h"
#include "bfr/rhi/swapchain.h"
#include "rhi/swapchain_build.h"
#include "vulkan/vulkan_caps.h"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <expected>
#include <span>

namespace bfr::rhi::vulkan {

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
    // The surface's answers in every backend's terms decide; only the Vulkan
    // names of what was decided are this backend's.
    const auto build = swapchainBuildFor(surfaceCapsOf(capabilities, presentModes), surfaceFormatsOf(formats),
                                         {.width = size.width, .height = size.height}, request);
    if (!build) {
        return std::unexpected(build.error());
    }
    return SwapchainSettings{
        .format = {.format = vulkanFormat(request.format), .colorSpace = vulkanColorSpace(request.colorSpace)},
        .extent = {.width = build->extent.width, .height = build->extent.height},
        .imageCount = build->imageCount,
        .compositeAlpha = compositeAlphaFor(request.alphaMode),
        .presentMode = presentModeFor(request.presentMode),
    };
}

} // namespace bfr::rhi::vulkan
