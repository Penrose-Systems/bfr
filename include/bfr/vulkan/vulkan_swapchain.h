#pragma once

#include "error.h"
#include "instance_table.h"
#include "log.h"
#include "swapchain.h"
#include "vulkan_surface.h"

#include <volk.h>

#include <cstdint>
#include <vector>

namespace bfr::rhi::vulkan {

/// Whether surface lets adapter present as request says, asked on vk. Fails
/// with RhiUnsupported, and says what the surface allows, when it doesn't take
/// something in request; the size is not asked, as the window can change it.
[[nodiscard]] core::Result<void> checkSwapchainRequest(const InstanceTable& vk, VkPhysicalDevice adapter,
                                                       VkSurfaceKHR surface, const SwapchainRequest& request);

/// One swapchain on a surface and the images it was last built with; empty
/// until built.
class VulkanSwapchain {
public:
    /// Builds on surface, which it takes over, through device's table and the
    /// table of the instance the adapter belongs to, as request says each
    /// time. It reports to logger; the tables and logger outlive it.
    VulkanSwapchain(VkDevice device, const VolkDeviceTable& table, const InstanceTable& instanceTable,
                    VulkanSurface surface, const SwapchainRequest& request, const core::Logger& logger);
    ~VulkanSwapchain();
    VulkanSwapchain(const VulkanSwapchain&) = delete;
    VulkanSwapchain& operator=(const VulkanSwapchain&) = delete;
    VulkanSwapchain(VulkanSwapchain&&) = delete;
    VulkanSwapchain& operator=(VulkanSwapchain&&) = delete;

    /// Builds images as the request says on adapter, at size in pixels,
    /// replacing those it has; a window with no area leaves it empty. Fails
    /// with RhiUnsupported when the surface no longer allows the request, and
    /// RhiSwapchainFailed, naming the call the API refused. Either leaves it
    /// empty.
    [[nodiscard]] core::Result<SwapchainImages> build(VkPhysicalDevice adapter, VkExtent2D size);

    /// Destroys the swapchain and forgets its images, keeping their storage;
    /// the surface stays.
    void release();

    [[nodiscard]] const VulkanSurface& surface() const { return surface_; }

private:
    VkDevice device_;
    const VolkDeviceTable* table_;
    const InstanceTable* instanceTable_;
    const core::Logger* logger_;
    VulkanSurface surface_;
    SwapchainRequest request_;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    // Refilled by each build, keeping its storage.
    std::vector<VkImage> images_;
};

} // namespace bfr::rhi::vulkan
