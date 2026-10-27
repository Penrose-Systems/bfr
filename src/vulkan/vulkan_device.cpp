#include "vulkan_device.h"

#include "caps.h"
#include "error.h"
#include "log.h"
#include "swapchain.h"
#include "vulkan_caps.h"
#include "vulkan_rhi.h"
#include "vulkan_swapchain.h"
#include "window.h"

#include <volk.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <expected>
#include <format>
#include <utility>

namespace bfr::rhi::vulkan {

VulkanDevice::VulkanDevice(VulkanRhi& rhi, VkPhysicalDevice adapter, VkDevice device, std::uint32_t presentFamily,
                           bool hasBudget)
    : rhi_(&rhi),
      adapter_(adapter),
      device_(device),
      presentFamily_(presentFamily),
      hasBudget_(hasBudget),
      swapchains_("swapchains", kSwapchainCapacity, kSwapchainLimit, rhi.logger()) {
    volkLoadDeviceTable(&table_, device_);
}

core::Result<SwapchainHandle> VulkanDevice::createSwapchain(const core::window::NativeWindow& window,
                                                            std::uint32_t imageCount) {
    auto surface = rhi_->surfaceFor(window);
    if (!surface) {
        return std::unexpected(std::move(surface.error()));
    }
    // Fixed for the surface, so asked once.
    const auto presents = rhi_->presents(adapter_, presentFamily_, *surface);
    if (!presents) {
        return std::unexpected(presents.error());
    }
    if (!*presents) {
        return core::failure(core::ErrorCode::RhiPresentUnsupported,
                             "the device's present queue doesn't present to the window");
    }
    const SwapchainHandle swapchain = swapchains_.emplace(device_, table_, std::move(*surface), imageCount,
                                                          rhi_->logger());
    if (swapchain.isNull()) {
        return core::failure(core::ErrorCode::RhiLimitReached,
                             std::format("a device holds at most {} swapchains", kSwapchainLimit));
    }
    return swapchain;
}

core::Result<SwapchainImages> VulkanDevice::resizeSwapchain(SwapchainHandle swapchain, std::uint32_t width,
                                                            std::uint32_t height) {
    VulkanSwapchain* found = swapchains_.find(swapchain);
    if (found == nullptr) {
        return core::failure(core::ErrorCode::RhiInvalidHandle, "the swapchain handle is stale");
    }
    return found->build(adapter_, {.width = width, .height = height});
}

core::Result<void> VulkanDevice::destroySwapchain(SwapchainHandle swapchain) {
    if (swapchain.isNull()) {
        return {};
    }
    // Destroyed at once, not retired against GPU progress: its window is the
    // caller's to close once this returns. Once queues submit, a wait on the
    // swapchain's last use goes first.
    if (swapchains_.erase(swapchain)) {
        return {};
    }
    return core::failure(core::ErrorCode::RhiInvalidHandle, "the swapchain handle is stale");
}

core::Result<MemoryBudget> VulkanDevice::budget() const {
    if (!hasBudget_) {
        return core::failure(core::ErrorCode::RhiUnsupported, "the driver lacks VK_EXT_memory_budget");
    }
    return readBudget(adapter_);
}

VulkanDevice::~VulkanDevice() {
    // A failed wait leaves nothing to do but release, and no result to report
    // it through, so it is sent as an error whatever BFR_LOGGING says.
    if (const VkResult result = table_.vkDeviceWaitIdle(device_); result != VK_SUCCESS) {
        rhi_->logger().send(core::Severity::Error,
                            std::format("vkDeviceWaitIdle: {}; destroying the device anyway", string_VkResult(result)));
    }
    // Swapchains are the device's children, so they go first.
    if (swapchains_.size() != 0) {
        BFR_LOG_WARN(rhi_->logger(), "{} Vulkan swapchains were left when the device went", swapchains_.size());
    }
    swapchains_.clear();
    table_.vkDestroyDevice(device_, nullptr);
    BFR_LOG_DEBUG(rhi_->logger(), "Vulkan device destroyed");
}

} // namespace bfr::rhi::vulkan
