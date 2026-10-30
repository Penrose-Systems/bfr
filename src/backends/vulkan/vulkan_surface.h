#pragma once

#include "bfr/core/error.h"
#include "bfr/core/window.h"
#include "vulkan/instance_table.h"

#include <volk.h>

#include <span>

namespace bfr::rhi::vulkan {

/// The surface extensions for each window system this build can create
/// surfaces on. Each needs VK_KHR_surface too.
[[nodiscard]] std::span<const char* const> surfaceExtensions();

/// Owns one surface and destroys it.
class VulkanSurface {
public:
    /// Takes ownership of surface, which instance made; headless when its
    /// window has no window system.
    VulkanSurface(VkInstance instance, const InstanceTable& vk, VkSurfaceKHR surface, bool headless);
    ~VulkanSurface();
    VulkanSurface(const VulkanSurface&) = delete;
    VulkanSurface& operator=(const VulkanSurface&) = delete;
    /// Takes the surface over, leaving other owning nothing.
    VulkanSurface(VulkanSurface&& other) noexcept;
    VulkanSurface& operator=(VulkanSurface&&) = delete;

    [[nodiscard]] VkSurfaceKHR handle() const { return surface_; }
    [[nodiscard]] bool headless() const { return headless_; }

private:
    VkInstance instance_;
    const InstanceTable* vk_;
    VkSurfaceKHR surface_;
    bool headless_;
};

/// A surface on window, made with instance, which has enabled extensions.
/// Fails with RhiSurfaceFailed when there is no window, when its window
/// system's surface extension is unavailable, or when the API refuses the
/// surface.
[[nodiscard]] core::Result<VulkanSurface> createSurface(VkInstance instance, const InstanceTable& vk,
                                                        std::span<const char* const> extensions,
                                                        const core::window::NativeWindow& window);

} // namespace bfr::rhi::vulkan
