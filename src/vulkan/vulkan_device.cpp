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
#include <format>
#include <utility>

namespace bfr::rhi::vulkan {

namespace {

// The GPU progress objects retire against. No queue submits yet, so nothing
// retired can still be in use and token 0 has always completed; once queues
// submit, a retire takes the last value submitted and a collect what has
// completed. Whatever is still retired when the device goes, the destructor
// collects once the GPU is idle.
constexpr std::uint64_t kNothingSubmitted = 0;

} // namespace

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
    const SwapchainHandle swapchain = swapchains_.emplace(device_, table_, std::move(*surface), imageCount);
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
    if (!rhi_->presents(adapter_, presentFamily_, found->surface())) {
        found->release();
        return core::failure(core::ErrorCode::RhiPresentUnsupported,
                             "the device's present queue doesn't present to the window");
    }
    return found->build(adapter_, {.width = width, .height = height});
}

core::Result<void> VulkanDevice::destroySwapchain(SwapchainHandle swapchain) {
    if (swapchain.isNull()) {
        return {};
    }
    if (swapchains_.retire(swapchain, kNothingSubmitted)) {
        swapchains_.collect(kNothingSubmitted);
        return {};
    }
    return core::failure(core::ErrorCode::RhiInvalidHandle, "the swapchain handle is stale");
}

VulkanDevice::~VulkanDevice() {
    // A failed wait leaves nothing to do but release.
    static_cast<void>(table_.vkDeviceWaitIdle(device_));
    // Swapchains are the device's children, so they go first: those still
    // live, and those retired but not yet collected.
    if (swapchains_.size() != 0) {
        BFR_LOG_WARN("{} Vulkan swapchains were left when the device went", swapchains_.size());
    }
    swapchains_.clear();
    table_.vkDestroyDevice(device_, nullptr);
    BFR_LOG_DEBUG("Vulkan device destroyed");
}

} // namespace bfr::rhi::vulkan
