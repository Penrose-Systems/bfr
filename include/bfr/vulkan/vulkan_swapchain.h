#pragma once

#include "error.h"
#include "log.h"
#include "swapchain.h"
#include "vulkan_surface.h"

#include <volk.h>

#include <cstdint>
#include <vector>

namespace bfr::rhi::vulkan {

/// One swapchain on a surface and the images it was last built with; empty
/// until built.
class VulkanSwapchain {
public:
    /// Builds on surface, which it takes over, through device's table, asking
    /// for imageCount images each time. It reports to logger, which outlives it.
    VulkanSwapchain(VkDevice device, const VolkDeviceTable& table, VulkanSurface surface, std::uint32_t imageCount,
                    const core::Logger& logger);
    ~VulkanSwapchain();
    VulkanSwapchain(const VulkanSwapchain&) = delete;
    VulkanSwapchain& operator=(const VulkanSwapchain&) = delete;
    VulkanSwapchain(VulkanSwapchain&&) = delete;
    VulkanSwapchain& operator=(VulkanSwapchain&&) = delete;

    /// Builds images as the surface allows on adapter, at size in pixels,
    /// replacing those it has; a window with no area
    /// leaves it empty. Fails with RhiSwapchainFailed, naming the call the API
    /// refused, and leaves it empty.
    [[nodiscard]] core::Result<SwapchainImages> build(VkPhysicalDevice adapter, VkExtent2D size);

    /// Destroys the swapchain and forgets its images, keeping their storage;
    /// the surface stays.
    void release();

    [[nodiscard]] const VulkanSurface& surface() const { return surface_; }

private:
    VkDevice device_;
    const VolkDeviceTable* table_;
    const core::Logger* logger_;
    VulkanSurface surface_;
    std::uint32_t imageCount_;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    // Refilled by each build, keeping their storage.
    std::vector<VkSurfaceFormatKHR> formats_;
    std::vector<VkImage> images_;
};

} // namespace bfr::rhi::vulkan
