#include "vulkan_swapchain.h"

#include "error.h"
#include "log.h"
#include "swapchain.h"
#include "swapchain_settings.h"
#include "vulkan_list.h"
#include "vulkan_surface.h"

#include <volk.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <format>
#include <string_view>
#include <utility>

namespace bfr::rhi::vulkan {

namespace {

VkSwapchainCreateInfoKHR createInfo(VkSurfaceKHR surface, const SwapchainSettings& settings,
                                    VkSurfaceTransformFlagBitsKHR transform) {
    VkSwapchainCreateInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    info.surface = surface;
    info.minImageCount = settings.imageCount;
    info.imageFormat = settings.format.format;
    info.imageColorSpace = settings.format.colorSpace;
    info.imageExtent = settings.extent;
    info.imageArrayLayers = 1;
    info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.preTransform = transform;
    info.compositeAlpha = settings.compositeAlpha;
    // Vsync: every surface supports it.
    info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    info.clipped = VK_TRUE;
    return info;
}

core::Result<SwapchainImages> refused(std::string_view function, VkResult result) {
    return core::failure(core::ErrorCode::RhiSwapchainFailed, std::format("{}: {}", function, string_VkResult(result)));
}

} // namespace

VulkanSwapchain::VulkanSwapchain(VkDevice device, const VolkDeviceTable& table, VulkanSurface surface,
                                 std::uint32_t imageCount)
    : device_(device), table_(&table), surface_(std::move(surface)), imageCount_(imageCount) {}

VulkanSwapchain::~VulkanSwapchain() {
    release();
}

core::Result<SwapchainImages> VulkanSwapchain::build(VkPhysicalDevice adapter, VkExtent2D size) {
    release();
    VkSurfaceKHR surface = surface_.handle();
    VkSurfaceCapabilitiesKHR capabilities{};
    if (const VkResult result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(adapter, surface, &capabilities);
        result != VK_SUCCESS) {
        return refused("vkGetPhysicalDeviceSurfaceCapabilitiesKHR", result);
    }
    if (auto listed = listInto(formats_, "vkGetPhysicalDeviceSurfaceFormatsKHR",
                               [adapter, surface](std::uint32_t* count, VkSurfaceFormatKHR* formats) {
                                   return vkGetPhysicalDeviceSurfaceFormatsKHR(adapter, surface, count, formats);
                               });
        !listed) {
        return core::failure(core::ErrorCode::RhiSwapchainFailed, listed.error());
    }
    if (formats_.empty()) {
        return core::failure(core::ErrorCode::RhiSwapchainFailed, "the surface lists no formats");
    }
    const SwapchainSettings settings = chooseSwapchainSettings(capabilities, formats_, size, imageCount_);
    if (settings.extent.width == 0 || settings.extent.height == 0) {
        BFR_LOG_DEBUG("swapchain left empty: the window has no area");
        return SwapchainImages{};
    }

    const VkSwapchainCreateInfoKHR info = createInfo(surface, settings, capabilities.currentTransform);
    // A failed call leaves the handle undefined, so the new swapchain is kept
    // only once it is complete.
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    if (const VkResult result = table_->vkCreateSwapchainKHR(device_, &info, nullptr, &swapchain);
        result != VK_SUCCESS) {
        return refused("vkCreateSwapchainKHR", result);
    }
    if (auto listed = listInto(images_, "vkGetSwapchainImagesKHR",
                               [this, swapchain](std::uint32_t* count, VkImage* listedImages) {
                                   return table_->vkGetSwapchainImagesKHR(device_, swapchain, count, listedImages);
                               });
        !listed) {
        table_->vkDestroySwapchainKHR(device_, swapchain, nullptr);
        images_.clear();
        return core::failure(core::ErrorCode::RhiSwapchainFailed, listed.error());
    }
    swapchain_ = swapchain;
    const auto count = static_cast<std::uint32_t>(images_.size());
    BFR_LOG_INFO("swapchain {}x{}, {} images, {}", settings.extent.width, settings.extent.height, count,
                 string_VkFormat(settings.format.format));
    return SwapchainImages{.width = settings.extent.width, .height = settings.extent.height, .count = count};
}

void VulkanSwapchain::release() {
    if (swapchain_ != VK_NULL_HANDLE) {
        table_->vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
    images_.clear();
}

} // namespace bfr::rhi::vulkan
