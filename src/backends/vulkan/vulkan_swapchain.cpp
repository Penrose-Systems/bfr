#include "vulkan/vulkan_swapchain.h"

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/rhi/swapchain.h"
#include "vulkan/instance_table.h"
#include "vulkan/swapchain_settings.h"
#include "vulkan/vulkan_list.h"
#include "vulkan/vulkan_result.h"
#include "vulkan/vulkan_surface.h"

#include <volk.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <expected>
#include <format>
#include <limits>
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

// What surface says about itself for adapter.
core::Result<VkSurfaceCapabilitiesKHR> askSurface(const InstanceTable& vk, VkPhysicalDevice adapter,
                                                  VkSurfaceKHR surface) {
    VkSurfaceCapabilitiesKHR capabilities{};
    if (const VkResult result = vk.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(adapter, surface, &capabilities);
        result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiSwapchainFailed,
                       {.call = "vkGetPhysicalDeviceSurfaceCapabilitiesKHR", .result = result});
    }
    return capabilities;
}

} // namespace

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
    const auto capabilities = askSurface(*instanceTable_, adapter, surface);
    if (!capabilities) {
        return std::unexpected(capabilities.error());
    }
    const SwapchainSettings settings = swapchainSettingsFor(*capabilities, size, request_);
    const auto imagesOf = [this](std::uint32_t width, std::uint32_t height, std::uint32_t count) {
        return SwapchainImages{.width = width,
                               .height = height,
                               .count = count,
                               .format = request_.format,
                               .colorSpace = request_.colorSpace,
                               .presentMode = request_.presentMode,
                               .alphaMode = request_.alphaMode};
    };
    if (settings.extent.width == 0 || settings.extent.height == 0) {
        BFR_LOG_DEBUG(*logger_, "swapchain left empty: the window has no area");
        return imagesOf(0, 0, 0);
    }

    const VkSwapchainCreateInfoKHR info = createInfo(surface, settings, capabilities->currentTransform);
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
    extent_ = settings.extent;
    const auto count = static_cast<std::uint32_t>(images_.size());
    if (auto made = makeSync(count, settings.format.format); !made) {
        release();
        return std::unexpected(std::move(made.error()));
    }
    BFR_LOG_INFO(*logger_, "swapchain {}x{}, {} images, {}, {}", settings.extent.width, settings.extent.height, count,
                 string_VkFormat(settings.format.format), name(request_.presentMode));
    return imagesOf(settings.extent.width, settings.extent.height, count);
}

core::Result<VkSemaphore> VulkanSwapchain::makeSemaphore() {
    VkSemaphoreCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkSemaphore semaphore = VK_NULL_HANDLE;
    if (const VkResult result = table_->vkCreateSemaphore(device_, &info, nullptr, &semaphore); result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiSwapchainFailed, {.call = "vkCreateSemaphore", .result = result});
    }
    return semaphore;
}

core::Result<void> VulkanSwapchain::makeSync(std::uint32_t count, VkFormat format) {
    layouts_.assign(count, VK_IMAGE_LAYOUT_UNDEFINED);
    for (VkImage image : images_) {
        VkImageViewCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        info.image = image;
        info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        info.format = format;
        info.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                 .baseMipLevel = 0,
                                 .levelCount = 1,
                                 .baseArrayLayer = 0,
                                 .layerCount = 1};
        VkImageView view = VK_NULL_HANDLE;
        if (const VkResult result = table_->vkCreateImageView(device_, &info, nullptr, &view); result != VK_SUCCESS) {
            return failure(core::ErrorCode::RhiSwapchainFailed, {.call = "vkCreateImageView", .result = result});
        }
        views_.push_back(view);
    }
    for (std::uint32_t i = 0; i < count; ++i) {
        auto semaphore = makeSemaphore();
        if (!semaphore) {
            return std::unexpected(std::move(semaphore.error()));
        }
        presentSemaphores_.push_back(*semaphore);
    }
    for (std::uint32_t i = 0; i < count + 1; ++i) {
        auto semaphore = makeSemaphore();
        if (!semaphore) {
            return std::unexpected(std::move(semaphore.error()));
        }
        acquireSemaphores_.push_back(*semaphore);
    }
    nextAcquire_ = 0;
    return {};
}

void VulkanSwapchain::release() {
    for (VkImageView view : views_) {
        table_->vkDestroyImageView(device_, view, nullptr);
    }
    for (VkSemaphore semaphore : presentSemaphores_) {
        table_->vkDestroySemaphore(device_, semaphore, nullptr);
    }
    for (VkSemaphore semaphore : acquireSemaphores_) {
        table_->vkDestroySemaphore(device_, semaphore, nullptr);
    }
    views_.clear();
    presentSemaphores_.clear();
    acquireSemaphores_.clear();
    layouts_.clear();
    acquiredImage_.reset();
    submitted_ = false;
    if (swapchain_ != VK_NULL_HANDLE) {
        table_->vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
    images_.clear();
}

core::Result<void> VulkanSwapchain::acquire() {
    if (swapchain_ == VK_NULL_HANDLE) {
        return core::failure(core::ErrorCode::RhiOutOfDate, "the swapchain has no images; resize it");
    }
    if (acquiredImage_.has_value()) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "an image is already acquired; present it first");
    }
    VkSemaphore semaphore = acquireSemaphores_[nextAcquire_];
    std::uint32_t image = 0;
    const VkResult result = table_->vkAcquireNextImageKHR(device_, swapchain_,
                                                          std::numeric_limits<std::uint64_t>::max(), semaphore,
                                                          VK_NULL_HANDLE, &image);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        // The semaphore is left signaled with nothing to wait on it, so it is
        // made again.
        table_->vkDestroySemaphore(device_, semaphore, nullptr);
        acquireSemaphores_[nextAcquire_] = VK_NULL_HANDLE;
        auto fresh = makeSemaphore();
        if (!fresh) {
            return std::unexpected(std::move(fresh.error()));
        }
        acquireSemaphores_[nextAcquire_] = *fresh;
        BFR_LOG_DEBUG(*logger_, "swapchain out of date on acquire");
        return core::failure(core::ErrorCode::RhiOutOfDate, "the swapchain no longer matches its window; resize it");
    }
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        return failure(core::ErrorCode::RhiSwapchainFailed, {.call = "vkAcquireNextImageKHR", .result = result});
    }
    acquiredImage_ = image;
    acquireUsed_ = nextAcquire_;
    nextAcquire_ = (nextAcquire_ + 1) % static_cast<std::uint32_t>(acquireSemaphores_.size());
    submitted_ = false;
    return {};
}

core::Result<AcquiredImage> VulkanSwapchain::acquired() const {
    if (!acquiredImage_.has_value()) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "no image is acquired");
    }
    const std::uint32_t image = *acquiredImage_;
    return AcquiredImage{.image = images_[image],
                         .view = views_[image],
                         .extent = extent_,
                         .layout = layouts_[image],
                         .acquireSemaphore = acquireSemaphores_[acquireUsed_],
                         .presentSemaphore = presentSemaphores_[image]};
}

void VulkanSwapchain::setLayout(VkImageLayout layout) {
    if (acquiredImage_.has_value()) {
        layouts_[*acquiredImage_] = layout;
    }
}

core::Result<void> VulkanSwapchain::present(VkQueue queue) {
    if (!acquiredImage_.has_value()) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "no image is acquired");
    }
    const std::uint32_t image = *acquiredImage_;
    // With nothing submitted, the image is shown as acquired; the acquire's
    // own signal is what the present waits on.
    const VkSemaphore wait = submitted_ ? presentSemaphores_[image] : acquireSemaphores_[acquireUsed_];
    VkPresentInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &wait;
    info.swapchainCount = 1;
    info.pSwapchains = &swapchain_;
    info.pImageIndices = &image;
    const VkResult result = table_->vkQueuePresentKHR(queue, &info);
    acquiredImage_.reset();
    submitted_ = false;
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        BFR_LOG_DEBUG(*logger_, "swapchain out of date on present");
        return core::failure(core::ErrorCode::RhiOutOfDate, "the swapchain no longer matches its window; resize it");
    }
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        return failure(core::ErrorCode::RhiSwapchainFailed, {.call = "vkQueuePresentKHR", .result = result});
    }
    return {};
}

} // namespace bfr::rhi::vulkan
