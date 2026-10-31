#include "vulkan/swapchain_settings.h"

#include "bfr/rhi/swapchain.h"
#include "vulkan/vulkan_caps.h"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <cstdint>

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

SwapchainSettings swapchainSettingsFor(const VkSurfaceCapabilitiesKHR& capabilities, VkExtent2D size,
                                       const SwapchainDesc& desc) {
    // A current extent of 0xFFFFFFFF says the swapchain's size sets the window's.
    constexpr std::uint32_t kSizeFromSwapchain = 0xFFFFFFFF;
    const VkExtent2D extent = capabilities.currentExtent.width == kSizeFromSwapchain ? size
                                                                                     : capabilities.currentExtent;
    return SwapchainSettings{
        .format = {.format = vulkanFormat(desc.format), .colorSpace = vulkanColorSpace(desc.colorSpace)},
        .extent = extent,
        .imageCount = desc.imageCount,
        .compositeAlpha = compositeAlphaFor(desc.alphaMode),
        .presentMode = presentModeFor(desc.presentMode),
    };
}

} // namespace bfr::rhi::vulkan
