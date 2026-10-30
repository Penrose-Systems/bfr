#pragma once

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/rhi/swapchain.h"
#include "vulkan/instance_table.h"
#include "vulkan/vulkan_surface.h"

#include <volk.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace bfr::rhi::vulkan {

/// Whether surface lets adapter present as request says, asked on vk. Fails
/// with RhiUnsupported, and says what the surface allows, when it doesn't take
/// something in request; the size is not asked, as the window can change it.
[[nodiscard]] core::Result<void> checkSwapchainRequest(const InstanceTable& vk, VkPhysicalDevice adapter,
                                                       VkSurfaceKHR surface, const SwapchainRequest& request);

/// A swapchain's acquired image, for rendering to and presenting.
struct AcquiredImage {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkExtent2D extent{};
    /// The image's layout now: undefined until first rendered to, then present.
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    /// Signaled when the image is ready; the submit that renders to it waits
    /// on this.
    VkSemaphore acquireSemaphore = VK_NULL_HANDLE;
    /// Signaled by the submit that renders to it; present waits on this.
    VkSemaphore presentSemaphore = VK_NULL_HANDLE;
};

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
    /// replacing those it has, which the GPU must be done with; a window with
    /// no area leaves it empty. Fails with RhiUnsupported when the surface no
    /// longer allows the request, and RhiSwapchainFailed, naming the call the
    /// API refused. Either leaves it empty.
    [[nodiscard]] core::Result<SwapchainImages> build(VkPhysicalDevice adapter, VkExtent2D size);

    /// Destroys the swapchain and forgets its images, keeping their storage;
    /// the surface stays.
    void release();

    /// Takes the next image. Fails with RhiOutOfDate when there are no images
    /// or the surface changed, RhiInvalidCall when an image is already
    /// acquired, and RhiSwapchainFailed when the API refuses.
    [[nodiscard]] core::Result<void> acquire();

    /// The acquired image. Fails with RhiInvalidCall when none is.
    [[nodiscard]] core::Result<AcquiredImage> acquired() const;

    /// Records the acquired image's layout after a barrier.
    void setLayout(VkImageLayout layout);

    /// Whether a submit already signals the acquired image's present semaphore.
    [[nodiscard]] bool submitted() const { return submitted_; }

    /// Records that a submit renders to the acquired image.
    void setSubmitted() { submitted_ = true; }

    /// Presents the acquired image on queue once the submit that rendered to
    /// it is done, or at once when none did; either way no image is acquired
    /// afterwards. Fails with RhiOutOfDate when the surface changed, the image
    /// still presented when it could be, RhiInvalidCall when no image is
    /// acquired, and RhiSwapchainFailed when the API refuses.
    [[nodiscard]] core::Result<void> present(VkQueue queue);

    [[nodiscard]] const VulkanSurface& surface() const { return surface_; }

private:
    // Views, layouts and semaphores for count images, and one more acquire
    // semaphore than images, so each is waited on before it is reused.
    [[nodiscard]] core::Result<void> makeSync(std::uint32_t count, VkFormat format);
    [[nodiscard]] core::Result<VkSemaphore> makeSemaphore();

    VkDevice device_;
    const VolkDeviceTable* table_;
    const InstanceTable* instanceTable_;
    const core::Logger* logger_;
    VulkanSurface surface_;
    SwapchainRequest request_;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkExtent2D extent_{};
    // Refilled by each build, keeping their storage.
    std::vector<VkImage> images_;
    std::vector<VkImageView> views_;
    std::vector<VkImageLayout> layouts_;
    std::vector<VkSemaphore> presentSemaphores_;
    std::vector<VkSemaphore> acquireSemaphores_;
    std::uint32_t nextAcquire_ = 0;
    std::uint32_t acquireUsed_ = 0;
    std::optional<std::uint32_t> acquiredImage_;
    bool submitted_ = false;
};

} // namespace bfr::rhi::vulkan
