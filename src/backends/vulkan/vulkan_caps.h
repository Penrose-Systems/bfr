#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/format.h"
#include "vulkan/instance_table.h"

#include <volk.h>

#include <optional>
#include <span>
#include <string>
#include <vector>

namespace bfr::rhi::vulkan {

// Each query below goes through vk, the table of the instance adapter belongs to.

/// The features a device turns on, as adapter reports them: every one in the
/// core structs, and every one of each extension below that it lists, which
/// the device then enables. The caps read the same structs, so they say what
/// the device has. core heads the chain a create info's pNext takes; the chain
/// points into itself, so it stays where it is made.
struct DeviceFeatures {
    DeviceFeatures(const InstanceTable& vk, VkPhysicalDevice adapter,
                   std::span<const VkExtensionProperties> extensions);
    ~DeviceFeatures() = default;
    DeviceFeatures(const DeviceFeatures&) = delete;
    DeviceFeatures& operator=(const DeviceFeatures&) = delete;
    DeviceFeatures(DeviceFeatures&&) = delete;
    DeviceFeatures& operator=(DeviceFeatures&&) = delete;

    VkPhysicalDeviceFeatures2 core{};
    VkPhysicalDeviceVulkan11Features vulkan11{};
    VkPhysicalDeviceVulkan12Features vulkan12{};
    VkPhysicalDeviceVulkan13Features vulkan13{};
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloat{};
    VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT imageAtomicInt64{};
    VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR barycentric{};
    VkPhysicalDeviceCustomBorderColorFeaturesEXT customBorderColor{};
    /// A portability driver, such as MoltenVK, lists the extension, and a
    /// device on that adapter must enable it.
    VkPhysicalDevicePortabilitySubsetFeaturesKHR portability{};
    /// Queue levels: the extension lets a device's queues take one.
    VkPhysicalDeviceGlobalPriorityQueryFeaturesKHR globalPriority{};
    /// A device with no queues needs its maintenance9.
    VkPhysicalDeviceMaintenance9FeaturesKHR maintenance9{};
    /// Primitive restart on list topologies.
    VkPhysicalDevicePrimitiveTopologyListRestartFeaturesEXT listRestart{};
    /// The extensions whose structs are chained, which the device enables.
    std::vector<const char*> enabledExtensions;
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
/// RhiWindowFailed, or as codeFor says, when the query fails.
[[nodiscard]] core::Result<SwapchainCaps> readSurfaceCaps(const InstanceTable& vk, VkPhysicalDevice adapter,
                                                          VkSurfaceKHR surface);

/// capabilities and presentModes, as a surface reports them, as SwapchainCaps.
[[nodiscard]] SwapchainCaps surfaceCapsOf(const VkSurfaceCapabilitiesKHR& capabilities,
                                          std::span<const VkPresentModeKHR> presentModes);

/// formats, as a surface lists them, those with names here.
[[nodiscard]] std::vector<SwapchainFormat> surfaceFormatsOf(std::span<const VkSurfaceFormatKHR> formats);

/// The formats a swapchain for surface may use on adapter, those with names
/// here. Fails as readSurfaceCaps does.
[[nodiscard]] core::Result<std::vector<SwapchainFormat>> readSurfaceFormats(const InstanceTable& vk,
                                                                            VkPhysicalDevice adapter,
                                                                            VkSurfaceKHR surface);

/// The heaps' budget and usage, which needs VK_EXT_memory_budget on adapter.
[[nodiscard]] MemoryBudget readBudget(const InstanceTable& vk, VkPhysicalDevice adapter);

/// format's Vulkan format, VK_FORMAT_UNDEFINED for Undefined; format is
/// Undefined or one of kFormats.
[[nodiscard]] VkFormat vulkanFormat(Format format);

/// The Vulkan color space that is colorSpace.
[[nodiscard]] VkColorSpaceKHR vulkanColorSpace(ColorSpace colorSpace);

/// The Format for format; nullopt for one without a name here.
[[nodiscard]] std::optional<Format> formatOf(VkFormat format);

} // namespace bfr::rhi::vulkan
