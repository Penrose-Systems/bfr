#include "vulkan_device.h"

#include "error.h"
#include "log.h"
#include "swapchain.h"
#include "vulkan_rhi.h"
#include "vulkan_swapchain.h"
#include "window.h"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <expected>
#include <utility>

namespace bfr::rhi::vulkan {

VulkanDevice::VulkanDevice(VulkanRhi& rhi, VkPhysicalDevice adapter, VkDevice device, std::uint32_t presentFamily)
    : rhi_(&rhi), adapter_(adapter), device_(device), presentFamily_(presentFamily) {
    volkLoadDeviceTable(&table_, device_);
}

core::Result<SwapchainHandle> VulkanDevice::createSwapchain(const core::window::NativeWindow& window,
                                                            std::uint32_t imageCount) {
    auto surface = rhi_->surfaceFor(window);
    if (!surface) {
        return std::unexpected(std::move(surface.error()));
    }
    return swapchains_.emplace(device_, table_, std::move(*surface), imageCount);
}

core::Result<SwapchainImages> VulkanDevice::resizeSwapchain(SwapchainHandle swapchain, std::uint32_t width,
                                                            std::uint32_t height) {
    VulkanSwapchain* found = swapchains_.find(swapchain);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the swapchain handle is stale");
    }
    if (!rhi_->presents(adapter_, presentFamily_, found->surface())) {
        found->release();
        return core::failure(core::ErrorCode::RhiPresentUnsupported,
                             "the device's present queue doesn't present to the window");
    }
    return found->build(adapter_, {.width = width, .height = height});
}

core::Result<void> VulkanDevice::destroySwapchain(SwapchainHandle swapchain) {
    if (swapchain.isNull() || swapchains_.erase(swapchain)) {
        return {};
    }
    return core::failure(core::ErrorCode::RhiInvalidHandle, "the swapchain handle is stale");
}

VulkanDevice::~VulkanDevice() {
    // A failed wait leaves nothing to do but release.
    static_cast<void>(table_.vkDeviceWaitIdle(device_));
    // Swapchains are the device's children, so they go first.
    if (swapchains_.size() != 0) {
        BFR_LOG_WARN("{} Vulkan swapchains were left when the device went", swapchains_.size());
        swapchains_.clear();
    }
    table_.vkDestroyDevice(device_, nullptr);
    BFR_LOG_DEBUG("Vulkan device destroyed");
}

} // namespace bfr::rhi::vulkan
