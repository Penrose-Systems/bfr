#pragma once

#include "caps.h"
#include "error.h"
#include "format.h"
#include "instance_table.h"

#include <volk.h>

#include <optional>
#include <span>
#include <string>
#include <vector>

namespace bfr::rhi::vulkan {

// Each query below goes through vk, the table of the instance adapter belongs to.

/// The floor's features, which every listed adapter has and every device turns
/// on. vulkan12.pNext points at vulkan13, so a create info chains vulkan12.
class FloorFeatures {
public:
    FloorFeatures();
    ~FloorFeatures() = default;
    FloorFeatures(const FloorFeatures&) = delete;
    FloorFeatures& operator=(const FloorFeatures&) = delete;
    FloorFeatures(FloorFeatures&&) = delete;
    FloorFeatures& operator=(FloorFeatures&&) = delete;

    [[nodiscard]] const VkPhysicalDeviceVulkan12Features& head() const { return vulkan12_; }

private:
    VkPhysicalDeviceVulkan12Features vulkan12_{};
    VkPhysicalDeviceVulkan13Features vulkan13_{};
};

/// The floor's features adapter lacks, by their Vulkan names.
[[nodiscard]] std::vector<std::string> missingFloorFeatures(const InstanceTable& vk, VkPhysicalDevice adapter);

/// Everything about adapter that stays the same while it runs. extensions are
/// the device extensions it lists. These queries can't fail.
[[nodiscard]] Caps readCaps(const InstanceTable& vk, VkPhysicalDevice adapter,
                            std::span<const VkExtensionProperties> extensions);

/// What adapter does with format. Fails with RhiQueryFailed, or as codeFor
/// says, when the API fails other than to say the format is unsupported.
[[nodiscard]] core::Result<FormatCaps> readFormatCaps(const InstanceTable& vk, VkPhysicalDevice adapter, Format format);

/// What a swapchain for surface may be on adapter. Fails with
/// RhiSurfaceFailed, or as codeFor says, when the query fails.
[[nodiscard]] core::Result<SurfaceCaps> readSurfaceCaps(const InstanceTable& vk, VkPhysicalDevice adapter,
                                                        VkSurfaceKHR surface);

/// The formats a swapchain for surface may use on adapter, those with names
/// here. Fails as readSurfaceCaps does.
[[nodiscard]] core::Result<std::vector<SurfaceFormat>> readSurfaceFormats(const InstanceTable& vk,
                                                                          VkPhysicalDevice adapter,
                                                                          VkSurfaceKHR surface);

/// The heaps' budget and usage, which needs VK_EXT_memory_budget on adapter.
[[nodiscard]] MemoryBudget readBudget(const InstanceTable& vk, VkPhysicalDevice adapter);

[[nodiscard]] VkFormat vulkanFormat(Format format);

/// The Vulkan color space that is colorSpace.
[[nodiscard]] VkColorSpaceKHR vulkanColorSpace(ColorSpace colorSpace);

/// The Format for format; nullopt for one without a name here.
[[nodiscard]] std::optional<Format> formatOf(VkFormat format);

} // namespace bfr::rhi::vulkan
