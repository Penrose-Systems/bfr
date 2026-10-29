#include "vulkan_swapchain.h"

#include "error.h"
#include "log.h"
#include "swapchain.h"
#include "swapchain_settings.h"
#include "vulkan_list.h"
#include "vulkan_result.h"
#include "vulkan_surface.h"

#include <volk.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <format>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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
    info.presentMode = settings.presentMode;
    info.clipped = VK_TRUE;
    return info;
}

core::Result<SwapchainImages> refused(std::string_view function, VkResult result) {
    return failure(core::ErrorCode::RhiSwapchainFailed, {.call = std::string{function}, .result = result});
}

// What a surface says it allows.
struct SurfaceAnswers {
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

core::Result<SurfaceAnswers> askSurface(const InstanceTable& vk, VkPhysicalDevice adapter, VkSurfaceKHR surface) {
    SurfaceAnswers answers;
    if (const VkResult result = vk.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(adapter, surface, &answers.capabilities);
        result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiSwapchainFailed,
                       {.call = "vkGetPhysicalDeviceSurfaceCapabilitiesKHR", .result = result});
    }
    auto formats = listAll<VkSurfaceFormatKHR>("vkGetPhysicalDeviceSurfaceFormatsKHR",
                                               [&vk, adapter, surface](std::uint32_t* count,
                                                                       VkSurfaceFormatKHR* listed) {
                                                   return vk.vkGetPhysicalDeviceSurfaceFormatsKHR(adapter, surface,
                                                                                                  count, listed);
                                               });
    if (!formats) {
        return failure(core::ErrorCode::RhiSwapchainFailed, formats.error());
    }
    auto modes = listAll<VkPresentModeKHR>("vkGetPhysicalDeviceSurfacePresentModesKHR",
                                           [&vk, adapter, surface](std::uint32_t* count, VkPresentModeKHR* listed) {
                                               return vk.vkGetPhysicalDeviceSurfacePresentModesKHR(adapter, surface,
                                                                                                   count, listed);
                                           });
    if (!modes) {
        return failure(core::ErrorCode::RhiSwapchainFailed, modes.error());
    }
    if (formats->empty()) {
        return core::failure(core::ErrorCode::RhiSwapchainFailed, "the surface lists no formats");
    }
    answers.formats = std::move(*formats);
    answers.presentModes = std::move(*modes);
    return answers;
}

} // namespace

core::Result<void> checkSwapchainRequest(const InstanceTable& vk, VkPhysicalDevice adapter, VkSurfaceKHR surface,
                                         const SwapchainRequest& request) {
    const auto answers = askSurface(vk, adapter, surface);
    if (!answers) {
        return std::unexpected(answers.error());
    }
    const auto settings = swapchainSettingsFor(answers->capabilities, answers->formats, answers->presentModes,
                                               VkExtent2D{}, request);
    if (!settings) {
        return std::unexpected(settings.error());
    }
    return {};
}

VulkanSwapchain::VulkanSwapchain(VkDevice device, const VolkDeviceTable& table, const InstanceTable& instanceTable,
                                 VulkanSurface surface, const SwapchainRequest& request, const core::Logger& logger)
    : device_(device),
      table_(&table),
      instanceTable_(&instanceTable),
      logger_(&logger),
      surface_(std::move(surface)),
      request_(request) {}

VulkanSwapchain::~VulkanSwapchain() {
    release();
}

core::Result<SwapchainImages> VulkanSwapchain::build(VkPhysicalDevice adapter, VkExtent2D size) {
    release();
    VkSurfaceKHR surface = surface_.handle();
    const auto answers = askSurface(*instanceTable_, adapter, surface);
    if (!answers) {
        return std::unexpected(answers.error());
    }
    const auto settings = swapchainSettingsFor(answers->capabilities, answers->formats, answers->presentModes, size,
                                               request_);
    if (!settings) {
        return std::unexpected(settings.error());
    }
    const auto imagesOf = [this](std::uint32_t width, std::uint32_t height, std::uint32_t count) {
        return SwapchainImages{.width = width,
                               .height = height,
                               .count = count,
                               .format = request_.format,
                               .colorSpace = request_.colorSpace,
                               .presentMode = request_.presentMode,
                               .alphaMode = request_.alphaMode};
    };
    if (settings->extent.width == 0 || settings->extent.height == 0) {
        BFR_LOG_DEBUG(*logger_, "swapchain left empty: the window has no area");
        return imagesOf(0, 0, 0);
    }

    const VkSwapchainCreateInfoKHR info = createInfo(surface, *settings, answers->capabilities.currentTransform);
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
        return failure(core::ErrorCode::RhiSwapchainFailed, listed.error());
    }
    swapchain_ = swapchain;
    const auto count = static_cast<std::uint32_t>(images_.size());
    BFR_LOG_INFO(*logger_, "swapchain {}x{}, {} images, {}, {}", settings->extent.width, settings->extent.height, count,
                 string_VkFormat(settings->format.format), name(request_.presentMode));
    return imagesOf(settings->extent.width, settings->extent.height, count);
}

void VulkanSwapchain::release() {
    if (swapchain_ != VK_NULL_HANDLE) {
        table_->vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
    images_.clear();
}

} // namespace bfr::rhi::vulkan
