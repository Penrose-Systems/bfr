#pragma once

#include "device.h"
#include "error.h"
#include "pool.h"
#include "swapchain.h"
#include "vulkan_swapchain.h"
#include "window.h"

#include <volk.h>

#include <cstdint>

namespace bfr::rhi::vulkan {

class VulkanRhi;

class VulkanDevice final : public Device {
public:
    /// Takes ownership of device, which rhi made on adapter; rhi outlives it.
    /// presentFamily is the queue family it presents from.
    VulkanDevice(VulkanRhi& rhi, VkPhysicalDevice adapter, VkDevice device, std::uint32_t presentFamily);
    ~VulkanDevice() override;
    VulkanDevice(const VulkanDevice&) = delete;
    VulkanDevice& operator=(const VulkanDevice&) = delete;
    VulkanDevice(VulkanDevice&&) = delete;
    VulkanDevice& operator=(VulkanDevice&&) = delete;

    [[nodiscard]] core::Result<SwapchainHandle> createSwapchain(const core::window::NativeWindow& window,
                                                                std::uint32_t imageCount) override;
    [[nodiscard]] core::Result<SwapchainImages> resizeSwapchain(SwapchainHandle swapchain, std::uint32_t width,
                                                                std::uint32_t height) override;
    [[nodiscard]] core::Result<void> destroySwapchain(SwapchainHandle swapchain) override;

private:
    VulkanRhi* rhi_;
    VkPhysicalDevice adapter_;
    VkDevice device_;
    // Each device calls through its own table: volk's global pointers hold one
    // device's functions, and two adapters can have different drivers.
    VolkDeviceTable table_{};
    std::uint32_t presentFamily_;
    // Room for a few windows' swapchains before the pool grows.
    static constexpr std::uint32_t kSwapchainCapacity = 4;
    core::Pool<VulkanSwapchain, Swapchain> swapchains_{"swapchains", kSwapchainCapacity};
};

} // namespace bfr::rhi::vulkan
