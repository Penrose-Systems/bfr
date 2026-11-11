#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/format.h"
#include "vulkan/instance_table.h"

#include <volk.h>

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace bfr::rhi::vulkan {

// Each query below goes through vk, the table of the instance adapter belongs to.
// version is the one a device on adapter uses, the lower of the instance's
// and adapter's (initialization.adoc:883-910), and is 1.2 or later; extensions
// are the device extensions adapter lists.

/// The oldest Vulkan BFR runs on.
inline constexpr std::uint32_t kMinimumVersion = VK_API_VERSION_1_2;

/// The keyframes a motion structure or instance has on Vulkan, times 0 and 1
/// (vertices, accelstructures.adoc:1207-1214; instance transforms, 2086-2090 and 2145-2149),
/// and the bytes between motion instance records (accelstructures.adoc:1796-1797).
inline constexpr std::uint32_t kVulkanMotionKeyframes = 2;
inline constexpr std::uint32_t kVulkanMotionInstanceStride = 160;

/// Whether a device at version has what Vulkan 1.3 made core.
[[nodiscard]] constexpr bool hasCore13(std::uint32_t version) {
    return version >= VK_API_VERSION_1_3;
}

/// Whether a device at version sets its queues' levels: core from 1.4, else
/// VK_KHR_global_priority where the adapter lists it.
[[nodiscard]] bool hasQueueLevels(std::uint32_t version, std::span<const VkExtensionProperties> extensions);

/// The features a device turns on, as adapter reports them: every one in the
/// structs of each core version up to version, and every one of each extension
/// below that it lists, which the device then enables. Below 1.3, the
/// extensions 1.3 made core stand in for its struct. The caps read the same
/// structs, so they say what the device has. core heads the chain a create
/// info's pNext takes; the chain points into itself, so it stays where it is
/// made.
struct DeviceFeatures {
    DeviceFeatures(const InstanceTable& vk, VkPhysicalDevice adapter, std::uint32_t version,
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
    VkPhysicalDeviceVulkan14Features vulkan14{};
    /// Below 1.3, what 1.3 made core, where adapter lists the extension.
    VkPhysicalDeviceDynamicRenderingFeaturesKHR dynamicRendering{};
    VkPhysicalDeviceSynchronization2FeaturesKHR synchronization2{};
    VkPhysicalDeviceExtendedDynamicStateFeaturesEXT extendedDynamicState{};
    VkPhysicalDeviceTextureCompressionASTCHDRFeaturesEXT astcHdr{};
    VkPhysicalDevice4444FormatsFeaturesEXT formats4444{};
    /// Its limits are maxBufferSize and the subgroup size range in the caps.
    VkPhysicalDeviceMaintenance4FeaturesKHR maintenance4{};
    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT subgroupSizeControl{};
    VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR integerDotProduct{};
    VkPhysicalDeviceShaderTerminateInvocationFeaturesKHR terminateInvocation{};
    VkPhysicalDeviceZeroInitializeWorkgroupMemoryFeaturesKHR zeroInitializeWorkgroupMemory{};
    VkPhysicalDeviceExtendedDynamicState2FeaturesEXT extendedDynamicState2{};
    /// The device turns its robustImageAccess off, as it does vulkan13's.
    VkPhysicalDeviceImageRobustnessFeaturesEXT imageRobustness{};
    VkPhysicalDeviceInlineUniformBlockFeaturesEXT inlineUniformBlock{};
    VkPhysicalDevicePipelineCreationCacheControlFeaturesEXT pipelineCreationCacheControl{};
    VkPhysicalDevicePrivateDataFeaturesEXT privateData{};
    VkPhysicalDeviceShaderDemoteToHelperInvocationFeaturesEXT demoteToHelperInvocation{};
    VkPhysicalDeviceTexelBufferAlignmentFeaturesEXT texelBufferAlignment{};
    VkPhysicalDeviceYcbcr2Plane444FormatsFeaturesEXT ycbcr2Plane444Formats{};
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloat{};
    VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT imageAtomicInt64{};
    VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR barycentric{};
    VkPhysicalDeviceCustomBorderColorFeaturesEXT customBorderColor{};
    /// A portability driver, such as MoltenVK, lists the extension, and a
    /// device on that adapter must enable it.
    VkPhysicalDevicePortabilitySubsetFeaturesKHR portability{};
    /// Queue levels: the extension lets a device's queues take one. From 1.4
    /// vulkan14 holds its feature, and this is not chained
    /// (VUID-VkDeviceCreateInfo-pNext-10360).
    VkPhysicalDeviceGlobalPriorityQueryFeaturesKHR globalPriority{};
    /// A device with no queues needs its maintenance9.
    VkPhysicalDeviceMaintenance9FeaturesKHR maintenance9{};
    /// Primitive restart on list topologies.
    VkPhysicalDevicePrimitiveTopologyListRestartFeaturesEXT listRestart{};
    /// Mesh and task shaders. The device clears its
    /// primitiveFragmentShadingRateMeshShader, which needs fragment shading
    /// rate turned on (VUID-VkPhysicalDeviceMeshShaderFeaturesEXT-primitiveFragmentShadingRateMeshShader-07033).
    VkPhysicalDeviceMeshShaderFeaturesEXT mesh{};
    /// Acceleration structures, with VK_KHR_deferred_host_operations, which
    /// the extension needs and which has no struct.
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructure{};
    VkPhysicalDeviceRayQueryFeaturesKHR rayQuery{};
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipeline{};
    /// Its rayTracingPipelineTraceRaysIndirect2 is traceRaysIndirect's.
    VkPhysicalDeviceRayTracingMaintenance1FeaturesKHR rayTracingMaintenance1{};
    VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR rayTracingPositionFetch{};
    /// The cross-vendor reorder; VK_NV_ray_tracing_invocation_reorder's is not
    /// enabled, so NV-only adapters report none.
    VkPhysicalDeviceRayTracingInvocationReorderFeaturesEXT invocationReorder{};
    VkPhysicalDevicePipelineLibraryGroupHandlesFeaturesEXT libraryGroupHandles{};
    VkPhysicalDeviceOpacityMicromapFeaturesEXT opacityMicromap{};
    VkPhysicalDeviceRayTracingMotionBlurFeaturesNV motionBlur{};
    VkPhysicalDevicePartitionedAccelerationStructureFeaturesNV partitioned{};
    VkPhysicalDeviceClusterAccelerationStructureFeaturesNV clusters{};
    /// VK_KHR_pipeline_library, which has no features struct, is enabled.
    bool pipelineLibrary = false;
    /// The extensions whose structs are chained, which the device enables.
    std::vector<const char*> enabledExtensions;
    std::uint32_t version = 0;

    /// What the device has of what 1.3 made core: its struct from 1.3, else
    /// the extension's.
    [[nodiscard]] bool hasDynamicRendering() const;
    [[nodiscard]] bool hasSynchronization2() const;
    /// vkCmdSetPrimitiveTopology: core from 1.3, else the extendedDynamicState
    /// feature (VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-03378).
    [[nodiscard]] bool hasDynamicTopology() const;
    [[nodiscard]] bool hasAstcHdr() const;
};

/// What BFR needs that features lack, by their Vulkan names: bindless takes
/// descriptor indexing with update-after-bind and partially bound tables;
/// queue progress takes timelines; barriers take synchronization2; rendering
/// takes dynamic rendering.
[[nodiscard]] std::vector<std::string> missingRequirements(const DeviceFeatures& features);

/// Everything about adapter that stays the same while it runs. These queries
/// can't fail.
[[nodiscard]] Caps readCaps(const InstanceTable& vk, VkPhysicalDevice adapter, std::uint32_t version,
                            std::span<const VkExtensionProperties> extensions);

/// Which formats added by 1.3 a device on adapter has: all of them from 1.3,
/// else those of the extensions adapter lists, which the device enables.
struct Formats13 {
    bool a4r4g4b4 = false;
    bool astcHdr = false;
};
[[nodiscard]] Formats13 formats13Of(std::uint32_t version, std::span<const VkExtensionProperties> extensions);

/// Whether a device may take format: one of a core version up to the one
/// it uses, or of an extension it enables (fundamentals.adoc:947-952). The
/// caps ask about no format the device can't take.
[[nodiscard]] bool hasFormat(Format format, const Formats13& formats);

/// What adapter does with format, none for one it may not be asked about
/// (hasFormat). Fails with RhiQueryFailed, or as codeFor says, when the API
/// fails other than to say the format is unsupported.
[[nodiscard]] core::Result<FormatCaps> readFormatCaps(const InstanceTable& vk, VkPhysicalDevice adapter,
                                                      std::uint32_t version,
                                                      std::span<const VkExtensionProperties> extensions, Format format);

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
