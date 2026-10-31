#include "vulkan/vulkan_caps.h"

#include "bfr/core/error.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/format.h"
#include "vulkan/instance_table.h"
#include "vulkan/vulkan_list.h"
#include "vulkan/vulkan_result.h"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace bfr::rhi::vulkan {

namespace {

// A floor feature: its Vulkan name, and its member in Features.
template <class Features>
struct FloorFeature {
    const char* name;
    VkBool32 Features::* member;
};

using Vulkan12 = VkPhysicalDeviceVulkan12Features;
using Vulkan13 = VkPhysicalDeviceVulkan13Features;

// Bindless takes descriptor indexing with update-after-bind; queue progress
// takes timelines; barriers take synchronization2; rendering takes dynamic
// rendering; GPU addresses take buffer device address.
constexpr std::array kFloor12{
    FloorFeature<Vulkan12>{.name = "descriptorIndexing", .member = &Vulkan12::descriptorIndexing},
    FloorFeature<Vulkan12>{.name = "runtimeDescriptorArray", .member = &Vulkan12::runtimeDescriptorArray},
    FloorFeature<Vulkan12>{.name = "descriptorBindingPartiallyBound",
                           .member = &Vulkan12::descriptorBindingPartiallyBound},
    FloorFeature<Vulkan12>{.name = "descriptorBindingVariableDescriptorCount",
                           .member = &Vulkan12::descriptorBindingVariableDescriptorCount},
    FloorFeature<Vulkan12>{.name = "descriptorBindingUpdateUnusedWhilePending",
                           .member = &Vulkan12::descriptorBindingUpdateUnusedWhilePending},
    FloorFeature<Vulkan12>{.name = "descriptorBindingSampledImageUpdateAfterBind",
                           .member = &Vulkan12::descriptorBindingSampledImageUpdateAfterBind},
    FloorFeature<Vulkan12>{.name = "descriptorBindingStorageImageUpdateAfterBind",
                           .member = &Vulkan12::descriptorBindingStorageImageUpdateAfterBind},
    FloorFeature<Vulkan12>{.name = "descriptorBindingStorageBufferUpdateAfterBind",
                           .member = &Vulkan12::descriptorBindingStorageBufferUpdateAfterBind},
    FloorFeature<Vulkan12>{.name = "shaderSampledImageArrayNonUniformIndexing",
                           .member = &Vulkan12::shaderSampledImageArrayNonUniformIndexing},
    FloorFeature<Vulkan12>{.name = "shaderStorageBufferArrayNonUniformIndexing",
                           .member = &Vulkan12::shaderStorageBufferArrayNonUniformIndexing},
    FloorFeature<Vulkan12>{.name = "shaderStorageImageArrayNonUniformIndexing",
                           .member = &Vulkan12::shaderStorageImageArrayNonUniformIndexing},
    FloorFeature<Vulkan12>{.name = "timelineSemaphore", .member = &Vulkan12::timelineSemaphore},
    FloorFeature<Vulkan12>{.name = "bufferDeviceAddress", .member = &Vulkan12::bufferDeviceAddress},
};

constexpr std::array kFloor13{
    FloorFeature<Vulkan13>{.name = "dynamicRendering", .member = &Vulkan13::dynamicRendering},
    FloorFeature<Vulkan13>{.name = "synchronization2", .member = &Vulkan13::synchronization2},
    FloorFeature<Vulkan13>{.name = "maintenance4", .member = &Vulkan13::maintenance4},
};

// Makes next the last link of the chain whose last pNext tail points at.
template <class Struct>
void link(void**& tail, Struct& next) {
    *tail = &next;
    tail = &next.pNext;
}

// The feature structs one query fills: the core ones always, and an
// extension's only when the adapter lists it, since chaining one it lacks is
// invalid. The chain points into itself, so it stays where it is made.
struct FeatureQuery {
    FeatureQuery(const InstanceTable& vk, VkPhysicalDevice adapter, std::span<const VkExtensionProperties> extensions) {
        core.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        vulkan11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        vulkan12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        vulkan13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        accelerationStructure.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
        rayTracingPipeline.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
        rayQuery.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
        mesh.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
        shadingRate.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR;
        atomicFloat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
        imageAtomicInt64.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT;
        barycentric.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR;

        void** tail = &core.pNext;
        link(tail, vulkan11);
        link(tail, vulkan12);
        link(tail, vulkan13);
        const auto linkIfListed = [&tail, extensions](const char* extension, auto& next) {
            if (lists(extensions, extension)) {
                link(tail, next);
            }
        };
        linkIfListed(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, accelerationStructure);
        linkIfListed(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, rayTracingPipeline);
        linkIfListed(VK_KHR_RAY_QUERY_EXTENSION_NAME, rayQuery);
        linkIfListed(VK_EXT_MESH_SHADER_EXTENSION_NAME, mesh);
        linkIfListed(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME, shadingRate);
        linkIfListed(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, atomicFloat);
        linkIfListed(VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME, imageAtomicInt64);
        linkIfListed(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME, barycentric);
        vk.vkGetPhysicalDeviceFeatures2(adapter, &core);
    }
    ~FeatureQuery() = default;
    FeatureQuery(const FeatureQuery&) = delete;
    FeatureQuery& operator=(const FeatureQuery&) = delete;
    FeatureQuery(FeatureQuery&&) = delete;
    FeatureQuery& operator=(FeatureQuery&&) = delete;

    VkPhysicalDeviceFeatures2 core{};
    VkPhysicalDeviceVulkan11Features vulkan11{};
    VkPhysicalDeviceVulkan12Features vulkan12{};
    VkPhysicalDeviceVulkan13Features vulkan13{};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructure{};
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipeline{};
    VkPhysicalDeviceRayQueryFeaturesKHR rayQuery{};
    VkPhysicalDeviceMeshShaderFeaturesEXT mesh{};
    VkPhysicalDeviceFragmentShadingRateFeaturesKHR shadingRate{};
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloat{};
    VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT imageAtomicInt64{};
    VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR barycentric{};
};

// The property structs one query fills, chained as FeatureQuery's are.
struct PropertyQuery {
    PropertyQuery(const InstanceTable& vk, VkPhysicalDevice adapter,
                  std::span<const VkExtensionProperties> extensions) {
        core.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        vulkan11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES;
        vulkan12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES;
        vulkan13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES;
        accelerationStructure.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
        rayTracingPipeline.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
        mesh.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT;
        shadingRate.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_PROPERTIES_KHR;

        void** tail = &core.pNext;
        link(tail, vulkan11);
        link(tail, vulkan12);
        link(tail, vulkan13);
        const auto linkIfListed = [&tail, extensions](const char* extension, auto& next) {
            if (lists(extensions, extension)) {
                link(tail, next);
            }
        };
        linkIfListed(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, accelerationStructure);
        linkIfListed(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, rayTracingPipeline);
        linkIfListed(VK_EXT_MESH_SHADER_EXTENSION_NAME, mesh);
        linkIfListed(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME, shadingRate);
        vk.vkGetPhysicalDeviceProperties2(adapter, &core);
    }
    ~PropertyQuery() = default;
    PropertyQuery(const PropertyQuery&) = delete;
    PropertyQuery& operator=(const PropertyQuery&) = delete;
    PropertyQuery(PropertyQuery&&) = delete;
    PropertyQuery& operator=(PropertyQuery&&) = delete;

    VkPhysicalDeviceProperties2 core{};
    VkPhysicalDeviceVulkan11Properties vulkan11{};
    VkPhysicalDeviceVulkan12Properties vulkan12{};
    VkPhysicalDeviceVulkan13Properties vulkan13{};
    VkPhysicalDeviceAccelerationStructurePropertiesKHR accelerationStructure{};
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingPipeline{};
    VkPhysicalDeviceMeshShaderPropertiesEXT mesh{};
    VkPhysicalDeviceFragmentShadingRatePropertiesKHR shadingRate{};
};

bool on(VkBool32 value) {
    return value == VK_TRUE;
}

AdapterInfo infoFrom(const PropertyQuery& properties) {
    const VkPhysicalDeviceProperties& core = properties.core.properties;
    return {
        .vendorId = core.vendorID,
        .deviceId = core.deviceID,
        .driverName = properties.vulkan12.driverName,
        .driverInfo = properties.vulkan12.driverInfo,
        .driverVersion = core.driverVersion,
        .apiMajor = VK_API_VERSION_MAJOR(core.apiVersion),
        .apiMinor = VK_API_VERSION_MINOR(core.apiVersion),
        .apiPatch = VK_API_VERSION_PATCH(core.apiVersion),
        .deviceUuid = std::to_array(properties.vulkan11.deviceUUID),
        .driverUuid = std::to_array(properties.vulkan11.driverUUID),
        .luid = std::to_array(properties.vulkan11.deviceLUID),
        .luidValid = on(properties.vulkan11.deviceLUIDValid),
        .cacheUuid = std::to_array(core.pipelineCacheUUID),
    };
}

Limits limitsFrom(const PropertyQuery& properties, const FeatureQuery& features) {
    const VkPhysicalDeviceLimits& core = properties.core.properties.limits;
    const bool shadingRateTexture = on(features.shadingRate.attachmentFragmentShadingRate);
    return {
        .maxTexture1d = core.maxImageDimension1D,
        .maxTexture2d = core.maxImageDimension2D,
        .maxTexture3d = core.maxImageDimension3D,
        .maxTextureCube = core.maxImageDimensionCube,
        .maxTextureLayers = core.maxImageArrayLayers,
        .maxBufferSize = properties.vulkan13.maxBufferSize,
        .maxAllocationSize = properties.vulkan11.maxMemoryAllocationSize,
        .maxAllocations = core.maxMemoryAllocationCount,
        .maxSamplerObjects = core.maxSamplerAllocationCount,
        .maxUniformBufferRange = core.maxUniformBufferRange,
        .maxStorageBufferRange = core.maxStorageBufferRange,
        .maxTexelBufferElements = core.maxTexelBufferElements,
        .maxPushConstantsSize = core.maxPushConstantsSize,
        .maxColorAttachments = core.maxColorAttachments,
        .maxRenderWidth = core.maxFramebufferWidth,
        .maxRenderHeight = core.maxFramebufferHeight,
        .maxRenderLayers = core.maxFramebufferLayers,
        .colorSampleCounts = core.framebufferColorSampleCounts,
        .integerSampleCounts = properties.vulkan12.framebufferIntegerColorSampleCounts,
        .depthSampleCounts = core.framebufferDepthSampleCounts,
        .stencilSampleCounts = core.framebufferStencilSampleCounts,
        .noAttachmentSampleCounts = core.framebufferNoAttachmentsSampleCounts,
        .storageTextureSampleCounts = core.storageImageSampleCounts,
        .maxViewports = core.maxViewports,
        .maxViewportWidth = core.maxViewportDimensions[0],
        .maxViewportHeight = core.maxViewportDimensions[1],
        .maxDrawIndirectCount = core.maxDrawIndirectCount,
        .maxClipDistances = core.maxClipDistances,
        .maxCullDistances = core.maxCullDistances,
        .maxSamplerAnisotropy = core.maxSamplerAnisotropy,
        .maxSamplerLodBias = core.maxSamplerLodBias,
        .minShadingRateTile = shadingRateTexture
                                  ? properties.shadingRate.minFragmentShadingRateAttachmentTexelSize.width
                                  : 0,
        .maxShadingRateTile = shadingRateTexture
                                  ? properties.shadingRate.maxFragmentShadingRateAttachmentTexelSize.width
                                  : 0,
        .minUniformBufferAlignment = core.minUniformBufferOffsetAlignment,
        .minStorageBufferAlignment = core.minStorageBufferOffsetAlignment,
        .minTexelBufferAlignment = core.minTexelBufferOffsetAlignment,
        .copyOffsetAlignment = core.optimalBufferCopyOffsetAlignment,
        .copyRowPitchAlignment = core.optimalBufferCopyRowPitchAlignment,
        .nonCoherentAtomSize = core.nonCoherentAtomSize,
        .bufferTextureGranularity = core.bufferImageGranularity,
        .timestampPeriod = core.timestampPeriod,
    };
}

BindlessLimits bindlessFrom(const PropertyQuery& properties) {
    const VkPhysicalDeviceVulkan12Properties& vulkan12 = properties.vulkan12;
    return {
        .maxTextures = vulkan12.maxDescriptorSetUpdateAfterBindSampledImages,
        .maxStorageTextures = vulkan12.maxDescriptorSetUpdateAfterBindStorageImages,
        .maxStorageBuffers = vulkan12.maxDescriptorSetUpdateAfterBindStorageBuffers,
        .maxUniformBuffers = vulkan12.maxDescriptorSetUpdateAfterBindUniformBuffers,
        .maxSamplers = vulkan12.maxDescriptorSetUpdateAfterBindSamplers,
        .maxPerStage = vulkan12.maxPerStageUpdateAfterBindResources,
    };
}

ComputeLimits computeFrom(const PropertyQuery& properties) {
    const VkPhysicalDeviceLimits& core = properties.core.properties.limits;
    return {
        .maxWorkgroupCount = std::to_array(core.maxComputeWorkGroupCount),
        .maxWorkgroupSize = std::to_array(core.maxComputeWorkGroupSize),
        .maxWorkgroupInvocations = core.maxComputeWorkGroupInvocations,
        .maxSharedMemory = core.maxComputeSharedMemorySize,
        .subgroupSize = properties.vulkan11.subgroupSize,
        .minSubgroupSize = properties.vulkan13.minSubgroupSize,
        .maxSubgroupSize = properties.vulkan13.maxSubgroupSize,
    };
}

MeshLimits meshFrom(const PropertyQuery& properties, const FeatureQuery& features) {
    if (!on(features.mesh.meshShader)) {
        return {};
    }
    const VkPhysicalDeviceMeshShaderPropertiesEXT& mesh = properties.mesh;
    return {
        .maxMeshWorkgroupSize = std::to_array(mesh.maxMeshWorkGroupSize),
        .maxMeshInvocations = mesh.maxMeshWorkGroupInvocations,
        .maxMeshOutputVertices = mesh.maxMeshOutputVertices,
        .maxMeshOutputPrimitives = mesh.maxMeshOutputPrimitives,
        .maxMeshSharedMemory = mesh.maxMeshSharedMemorySize,
        .maxTaskWorkgroupSize = on(features.mesh.taskShader) ? std::to_array(mesh.maxTaskWorkGroupSize)
                                                             : std::array<std::uint32_t, 3>{},
        .maxTaskInvocations = on(features.mesh.taskShader) ? mesh.maxTaskWorkGroupInvocations : 0,
        .maxTaskPayloadSize = on(features.mesh.taskShader) ? mesh.maxTaskPayloadSize : 0,
    };
}

RayTracingLimits rayTracingFrom(const PropertyQuery& properties, const FeatureQuery& features) {
    RayTracingLimits limits;
    if (on(features.accelerationStructure.accelerationStructure)) {
        const VkPhysicalDeviceAccelerationStructurePropertiesKHR& structures = properties.accelerationStructure;
        limits.maxGeometries = structures.maxGeometryCount;
        limits.maxInstances = structures.maxInstanceCount;
        limits.maxPrimitives = structures.maxPrimitiveCount;
        limits.scratchAlignment = structures.minAccelerationStructureScratchOffsetAlignment;
    }
    if (on(features.rayTracingPipeline.rayTracingPipeline)) {
        const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& pipeline = properties.rayTracingPipeline;
        limits.maxRecursionDepth = pipeline.maxRayRecursionDepth;
        limits.handleSize = pipeline.shaderGroupHandleSize;
        limits.handleAlignment = pipeline.shaderGroupHandleAlignment;
        limits.tableAlignment = pipeline.shaderGroupBaseAlignment;
    }
    return limits;
}

Features featuresFrom(const FeatureQuery& features, const PropertyQuery& properties,
                      std::span<const VkExtensionProperties> extensions) {
    const VkPhysicalDeviceFeatures& core = features.core.features;
    return {
        .rayQuery = on(features.rayQuery.rayQuery),
        .rayTracingPipeline = on(features.rayTracingPipeline.rayTracingPipeline),
        .accelerationStructure = on(features.accelerationStructure.accelerationStructure),
        .meshShader = on(features.mesh.meshShader),
        .taskShader = on(features.mesh.taskShader),
        .shadingRate = on(features.shadingRate.pipelineFragmentShadingRate),
        .shadingRateTexture = on(features.shadingRate.attachmentFragmentShadingRate),
        .shaderFloat16 = on(features.vulkan12.shaderFloat16),
        .shaderFloat64 = on(core.shaderFloat64),
        .shaderInt8 = on(features.vulkan12.shaderInt8),
        .shaderInt16 = on(core.shaderInt16),
        .shaderInt64 = on(core.shaderInt64),
        .atomicInt64 = on(features.vulkan12.shaderBufferInt64Atomics),
        .atomicInt64Texture = on(features.imageAtomicInt64.shaderImageInt64Atomics),
        .atomicFloat32Add = on(features.atomicFloat.shaderBufferFloat32AtomicAdd),
        .barycentrics = on(features.barycentric.fragmentShaderBarycentric),
        .conservativeRaster = lists(extensions, VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME),
        .geometryShader = on(core.geometryShader),
        .tessellation = on(core.tessellationShader),
        .multiview = on(features.vulkan11.multiview),
        .sampleRateShading = on(core.sampleRateShading),
        .dualSourceBlend = on(core.dualSrcBlend),
        .independentBlend = on(core.independentBlend),
        .logicOp = on(core.logicOp),
        .depthClamp = on(core.depthClamp),
        .depthBounds = on(core.depthBounds),
        .wireframe = on(core.fillModeNonSolid),
        .wideLines = on(core.wideLines),
        .samplerAnisotropy = on(core.samplerAnisotropy),
        .samplerMinmax = on(features.vulkan12.samplerFilterMinmax),
        .compressionBc = on(core.textureCompressionBC),
        .compressionEtc2 = on(core.textureCompressionETC2),
        .compressionAstc = on(core.textureCompressionASTC_LDR),
        .compressionAstcHdr = on(features.vulkan13.textureCompressionASTC_HDR),
        .multiDrawIndirect = on(core.multiDrawIndirect),
        .drawIndirectCount = on(features.vulkan12.drawIndirectCount),
        .drawIndirectFirstInstance = on(core.drawIndirectFirstInstance),
        .storageReadWithoutFormat = on(core.shaderStorageImageReadWithoutFormat),
        .storageWriteWithoutFormat = on(core.shaderStorageImageWriteWithoutFormat),
        .timestampsOnAllQueues = on(properties.core.properties.limits.timestampComputeAndGraphics),
        .pipelineStatistics = on(core.pipelineStatisticsQuery),
        .occlusionQueryPrecise = on(core.occlusionQueryPrecise),
        .sparseBinding = on(core.sparseBinding),
        .sparseTextures = on(core.sparseResidencyImage2D),
        .sparseBuffers = on(core.sparseResidencyBuffer),
    };
}

Memory memoryFrom(const InstanceTable& vk, VkPhysicalDevice adapter, VkPhysicalDeviceType type) {
    VkPhysicalDeviceMemoryProperties properties{};
    vk.vkGetPhysicalDeviceMemoryProperties(adapter, &properties);
    const auto vulkanTypes = std::span{properties.memoryTypes}.first(properties.memoryTypeCount);
    const auto vulkanHeaps = std::span{properties.memoryHeaps}.first(properties.memoryHeapCount);
    Memory memory{.heaps = {},
                  .types = {},
                  .unified = type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU || type == VK_PHYSICAL_DEVICE_TYPE_CPU};
    for (const VkMemoryType& vulkanType : vulkanTypes) {
        const VkMemoryPropertyFlags flags = vulkanType.propertyFlags;
        memory.types.push_back({
            .heap = vulkanType.heapIndex,
            .deviceLocal = (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0,
            .hostVisible = (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0,
            .hostCoherent = (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0,
            .hostCached = (flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0,
            .lazilyAllocated = (flags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) != 0,
        });
    }
    for (std::uint32_t heap = 0; heap < vulkanHeaps.size(); ++heap) {
        memory.heaps.push_back({
            .size = vulkanHeaps[heap].size,
            .deviceLocal = (vulkanHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0,
            .hostVisible = std::ranges::any_of(memory.types,
                                               [heap](const MemoryType& memoryType) {
                                                   return memoryType.heap == heap && memoryType.hostVisible;
                                               }),
        });
    }
    return memory;
}

// Graphics counts every family that runs graphics; compute, those that run
// compute but not graphics; copy, those that run neither. Each kind's
// timestamp bits are the most any of its families has.
Queues queuesFrom(const InstanceTable& vk, VkPhysicalDevice adapter) {
    std::uint32_t count = 0;
    vk.vkGetPhysicalDeviceQueueFamilyProperties(adapter, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vk.vkGetPhysicalDeviceQueueFamilyProperties(adapter, &count, families.data());
    Queues queues;
    for (const VkQueueFamilyProperties& family : families) {
        const VkQueueFlags flags = family.queueFlags;
        if ((flags & VK_QUEUE_GRAPHICS_BIT) != 0) {
            queues.graphics += family.queueCount;
            queues.graphicsTimestampBits = std::max(queues.graphicsTimestampBits, family.timestampValidBits);
        }
        else if ((flags & VK_QUEUE_COMPUTE_BIT) != 0) {
            queues.compute += family.queueCount;
            queues.computeTimestampBits = std::max(queues.computeTimestampBits, family.timestampValidBits);
        }
        else if ((flags & VK_QUEUE_TRANSFER_BIT) != 0) {
            queues.copy += family.queueCount;
            queues.copyTimestampBits = std::max(queues.copyTimestampBits, family.timestampValidBits);
        }
    }
    return queues;
}

struct FormatPair {
    Format format;
    VkFormat vulkan;
};

constexpr std::array kVulkanFormats{
    FormatPair{.format = Format::R8Unorm, .vulkan = VK_FORMAT_R8_UNORM},
    FormatPair{.format = Format::R8Snorm, .vulkan = VK_FORMAT_R8_SNORM},
    FormatPair{.format = Format::R8Uint, .vulkan = VK_FORMAT_R8_UINT},
    FormatPair{.format = Format::R8Sint, .vulkan = VK_FORMAT_R8_SINT},
    FormatPair{.format = Format::Rg8Unorm, .vulkan = VK_FORMAT_R8G8_UNORM},
    FormatPair{.format = Format::Rg8Snorm, .vulkan = VK_FORMAT_R8G8_SNORM},
    FormatPair{.format = Format::Rg8Uint, .vulkan = VK_FORMAT_R8G8_UINT},
    FormatPair{.format = Format::Rg8Sint, .vulkan = VK_FORMAT_R8G8_SINT},
    FormatPair{.format = Format::Rgba8Unorm, .vulkan = VK_FORMAT_R8G8B8A8_UNORM},
    FormatPair{.format = Format::Rgba8Srgb, .vulkan = VK_FORMAT_R8G8B8A8_SRGB},
    FormatPair{.format = Format::Rgba8Snorm, .vulkan = VK_FORMAT_R8G8B8A8_SNORM},
    FormatPair{.format = Format::Rgba8Uint, .vulkan = VK_FORMAT_R8G8B8A8_UINT},
    FormatPair{.format = Format::Rgba8Sint, .vulkan = VK_FORMAT_R8G8B8A8_SINT},
    FormatPair{.format = Format::Bgra8Unorm, .vulkan = VK_FORMAT_B8G8R8A8_UNORM},
    FormatPair{.format = Format::Bgra8Srgb, .vulkan = VK_FORMAT_B8G8R8A8_SRGB},
    FormatPair{.format = Format::R16Unorm, .vulkan = VK_FORMAT_R16_UNORM},
    FormatPair{.format = Format::R16Snorm, .vulkan = VK_FORMAT_R16_SNORM},
    FormatPair{.format = Format::R16Uint, .vulkan = VK_FORMAT_R16_UINT},
    FormatPair{.format = Format::R16Sint, .vulkan = VK_FORMAT_R16_SINT},
    FormatPair{.format = Format::R16Float, .vulkan = VK_FORMAT_R16_SFLOAT},
    FormatPair{.format = Format::Rg16Unorm, .vulkan = VK_FORMAT_R16G16_UNORM},
    FormatPair{.format = Format::Rg16Snorm, .vulkan = VK_FORMAT_R16G16_SNORM},
    FormatPair{.format = Format::Rg16Uint, .vulkan = VK_FORMAT_R16G16_UINT},
    FormatPair{.format = Format::Rg16Sint, .vulkan = VK_FORMAT_R16G16_SINT},
    FormatPair{.format = Format::Rg16Float, .vulkan = VK_FORMAT_R16G16_SFLOAT},
    FormatPair{.format = Format::Rgba16Unorm, .vulkan = VK_FORMAT_R16G16B16A16_UNORM},
    FormatPair{.format = Format::Rgba16Snorm, .vulkan = VK_FORMAT_R16G16B16A16_SNORM},
    FormatPair{.format = Format::Rgba16Uint, .vulkan = VK_FORMAT_R16G16B16A16_UINT},
    FormatPair{.format = Format::Rgba16Sint, .vulkan = VK_FORMAT_R16G16B16A16_SINT},
    FormatPair{.format = Format::Rgba16Float, .vulkan = VK_FORMAT_R16G16B16A16_SFLOAT},
    FormatPair{.format = Format::R32Uint, .vulkan = VK_FORMAT_R32_UINT},
    FormatPair{.format = Format::R32Sint, .vulkan = VK_FORMAT_R32_SINT},
    FormatPair{.format = Format::R32Float, .vulkan = VK_FORMAT_R32_SFLOAT},
    FormatPair{.format = Format::Rg32Uint, .vulkan = VK_FORMAT_R32G32_UINT},
    FormatPair{.format = Format::Rg32Sint, .vulkan = VK_FORMAT_R32G32_SINT},
    FormatPair{.format = Format::Rg32Float, .vulkan = VK_FORMAT_R32G32_SFLOAT},
    FormatPair{.format = Format::Rgb32Uint, .vulkan = VK_FORMAT_R32G32B32_UINT},
    FormatPair{.format = Format::Rgb32Sint, .vulkan = VK_FORMAT_R32G32B32_SINT},
    FormatPair{.format = Format::Rgb32Float, .vulkan = VK_FORMAT_R32G32B32_SFLOAT},
    FormatPair{.format = Format::Rgba32Uint, .vulkan = VK_FORMAT_R32G32B32A32_UINT},
    FormatPair{.format = Format::Rgba32Sint, .vulkan = VK_FORMAT_R32G32B32A32_SINT},
    FormatPair{.format = Format::Rgba32Float, .vulkan = VK_FORMAT_R32G32B32A32_SFLOAT},
    FormatPair{.format = Format::Rgb10a2Unorm, .vulkan = VK_FORMAT_A2B10G10R10_UNORM_PACK32},
    FormatPair{.format = Format::Rgb10a2Uint, .vulkan = VK_FORMAT_A2B10G10R10_UINT_PACK32},
    FormatPair{.format = Format::Bgr10a2Unorm, .vulkan = VK_FORMAT_A2R10G10B10_UNORM_PACK32},
    FormatPair{.format = Format::Rg11b10Float, .vulkan = VK_FORMAT_B10G11R11_UFLOAT_PACK32},
    FormatPair{.format = Format::Rgb9e5Float, .vulkan = VK_FORMAT_E5B9G9R9_UFLOAT_PACK32},
    FormatPair{.format = Format::B5g6r5Unorm, .vulkan = VK_FORMAT_R5G6B5_UNORM_PACK16},
    FormatPair{.format = Format::Bgr5a1Unorm, .vulkan = VK_FORMAT_A1R5G5B5_UNORM_PACK16},
    FormatPair{.format = Format::Bgra4Unorm, .vulkan = VK_FORMAT_A4R4G4B4_UNORM_PACK16},
    FormatPair{.format = Format::D16Unorm, .vulkan = VK_FORMAT_D16_UNORM},
    FormatPair{.format = Format::D24UnormS8Uint, .vulkan = VK_FORMAT_D24_UNORM_S8_UINT},
    FormatPair{.format = Format::D32Float, .vulkan = VK_FORMAT_D32_SFLOAT},
    FormatPair{.format = Format::D32FloatS8Uint, .vulkan = VK_FORMAT_D32_SFLOAT_S8_UINT},
    FormatPair{.format = Format::S8Uint, .vulkan = VK_FORMAT_S8_UINT},
    FormatPair{.format = Format::Bc1Unorm, .vulkan = VK_FORMAT_BC1_RGBA_UNORM_BLOCK},
    FormatPair{.format = Format::Bc1Srgb, .vulkan = VK_FORMAT_BC1_RGBA_SRGB_BLOCK},
    FormatPair{.format = Format::Bc2Unorm, .vulkan = VK_FORMAT_BC2_UNORM_BLOCK},
    FormatPair{.format = Format::Bc2Srgb, .vulkan = VK_FORMAT_BC2_SRGB_BLOCK},
    FormatPair{.format = Format::Bc3Unorm, .vulkan = VK_FORMAT_BC3_UNORM_BLOCK},
    FormatPair{.format = Format::Bc3Srgb, .vulkan = VK_FORMAT_BC3_SRGB_BLOCK},
    FormatPair{.format = Format::Bc4Unorm, .vulkan = VK_FORMAT_BC4_UNORM_BLOCK},
    FormatPair{.format = Format::Bc4Snorm, .vulkan = VK_FORMAT_BC4_SNORM_BLOCK},
    FormatPair{.format = Format::Bc5Unorm, .vulkan = VK_FORMAT_BC5_UNORM_BLOCK},
    FormatPair{.format = Format::Bc5Snorm, .vulkan = VK_FORMAT_BC5_SNORM_BLOCK},
    FormatPair{.format = Format::Bc6hUfloat, .vulkan = VK_FORMAT_BC6H_UFLOAT_BLOCK},
    FormatPair{.format = Format::Bc6hSfloat, .vulkan = VK_FORMAT_BC6H_SFLOAT_BLOCK},
    FormatPair{.format = Format::Bc7Unorm, .vulkan = VK_FORMAT_BC7_UNORM_BLOCK},
    FormatPair{.format = Format::Bc7Srgb, .vulkan = VK_FORMAT_BC7_SRGB_BLOCK},
    FormatPair{.format = Format::Etc2Rgb8Unorm, .vulkan = VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK},
    FormatPair{.format = Format::Etc2Rgb8Srgb, .vulkan = VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK},
    FormatPair{.format = Format::Etc2Rgb8a1Unorm, .vulkan = VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK},
    FormatPair{.format = Format::Etc2Rgb8a1Srgb, .vulkan = VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK},
    FormatPair{.format = Format::Etc2Rgba8Unorm, .vulkan = VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK},
    FormatPair{.format = Format::Etc2Rgba8Srgb, .vulkan = VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK},
    FormatPair{.format = Format::EacR11Unorm, .vulkan = VK_FORMAT_EAC_R11_UNORM_BLOCK},
    FormatPair{.format = Format::EacR11Snorm, .vulkan = VK_FORMAT_EAC_R11_SNORM_BLOCK},
    FormatPair{.format = Format::EacRg11Unorm, .vulkan = VK_FORMAT_EAC_R11G11_UNORM_BLOCK},
    FormatPair{.format = Format::EacRg11Snorm, .vulkan = VK_FORMAT_EAC_R11G11_SNORM_BLOCK},
    FormatPair{.format = Format::Astc4x4Unorm, .vulkan = VK_FORMAT_ASTC_4x4_UNORM_BLOCK},
    FormatPair{.format = Format::Astc4x4Srgb, .vulkan = VK_FORMAT_ASTC_4x4_SRGB_BLOCK},
    FormatPair{.format = Format::Astc4x4Float, .vulkan = VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK},
    FormatPair{.format = Format::Astc5x4Unorm, .vulkan = VK_FORMAT_ASTC_5x4_UNORM_BLOCK},
    FormatPair{.format = Format::Astc5x4Srgb, .vulkan = VK_FORMAT_ASTC_5x4_SRGB_BLOCK},
    FormatPair{.format = Format::Astc5x4Float, .vulkan = VK_FORMAT_ASTC_5x4_SFLOAT_BLOCK},
    FormatPair{.format = Format::Astc5x5Unorm, .vulkan = VK_FORMAT_ASTC_5x5_UNORM_BLOCK},
    FormatPair{.format = Format::Astc5x5Srgb, .vulkan = VK_FORMAT_ASTC_5x5_SRGB_BLOCK},
    FormatPair{.format = Format::Astc5x5Float, .vulkan = VK_FORMAT_ASTC_5x5_SFLOAT_BLOCK},
    FormatPair{.format = Format::Astc6x5Unorm, .vulkan = VK_FORMAT_ASTC_6x5_UNORM_BLOCK},
    FormatPair{.format = Format::Astc6x5Srgb, .vulkan = VK_FORMAT_ASTC_6x5_SRGB_BLOCK},
    FormatPair{.format = Format::Astc6x5Float, .vulkan = VK_FORMAT_ASTC_6x5_SFLOAT_BLOCK},
    FormatPair{.format = Format::Astc6x6Unorm, .vulkan = VK_FORMAT_ASTC_6x6_UNORM_BLOCK},
    FormatPair{.format = Format::Astc6x6Srgb, .vulkan = VK_FORMAT_ASTC_6x6_SRGB_BLOCK},
    FormatPair{.format = Format::Astc6x6Float, .vulkan = VK_FORMAT_ASTC_6x6_SFLOAT_BLOCK},
    FormatPair{.format = Format::Astc8x5Unorm, .vulkan = VK_FORMAT_ASTC_8x5_UNORM_BLOCK},
    FormatPair{.format = Format::Astc8x5Srgb, .vulkan = VK_FORMAT_ASTC_8x5_SRGB_BLOCK},
    FormatPair{.format = Format::Astc8x5Float, .vulkan = VK_FORMAT_ASTC_8x5_SFLOAT_BLOCK},
    FormatPair{.format = Format::Astc8x6Unorm, .vulkan = VK_FORMAT_ASTC_8x6_UNORM_BLOCK},
    FormatPair{.format = Format::Astc8x6Srgb, .vulkan = VK_FORMAT_ASTC_8x6_SRGB_BLOCK},
    FormatPair{.format = Format::Astc8x6Float, .vulkan = VK_FORMAT_ASTC_8x6_SFLOAT_BLOCK},
    FormatPair{.format = Format::Astc8x8Unorm, .vulkan = VK_FORMAT_ASTC_8x8_UNORM_BLOCK},
    FormatPair{.format = Format::Astc8x8Srgb, .vulkan = VK_FORMAT_ASTC_8x8_SRGB_BLOCK},
    FormatPair{.format = Format::Astc8x8Float, .vulkan = VK_FORMAT_ASTC_8x8_SFLOAT_BLOCK},
    FormatPair{.format = Format::Astc10x5Unorm, .vulkan = VK_FORMAT_ASTC_10x5_UNORM_BLOCK},
    FormatPair{.format = Format::Astc10x5Srgb, .vulkan = VK_FORMAT_ASTC_10x5_SRGB_BLOCK},
    FormatPair{.format = Format::Astc10x5Float, .vulkan = VK_FORMAT_ASTC_10x5_SFLOAT_BLOCK},
    FormatPair{.format = Format::Astc10x6Unorm, .vulkan = VK_FORMAT_ASTC_10x6_UNORM_BLOCK},
    FormatPair{.format = Format::Astc10x6Srgb, .vulkan = VK_FORMAT_ASTC_10x6_SRGB_BLOCK},
    FormatPair{.format = Format::Astc10x6Float, .vulkan = VK_FORMAT_ASTC_10x6_SFLOAT_BLOCK},
    FormatPair{.format = Format::Astc10x8Unorm, .vulkan = VK_FORMAT_ASTC_10x8_UNORM_BLOCK},
    FormatPair{.format = Format::Astc10x8Srgb, .vulkan = VK_FORMAT_ASTC_10x8_SRGB_BLOCK},
    FormatPair{.format = Format::Astc10x8Float, .vulkan = VK_FORMAT_ASTC_10x8_SFLOAT_BLOCK},
    FormatPair{.format = Format::Astc10x10Unorm, .vulkan = VK_FORMAT_ASTC_10x10_UNORM_BLOCK},
    FormatPair{.format = Format::Astc10x10Srgb, .vulkan = VK_FORMAT_ASTC_10x10_SRGB_BLOCK},
    FormatPair{.format = Format::Astc10x10Float, .vulkan = VK_FORMAT_ASTC_10x10_SFLOAT_BLOCK},
    FormatPair{.format = Format::Astc12x10Unorm, .vulkan = VK_FORMAT_ASTC_12x10_UNORM_BLOCK},
    FormatPair{.format = Format::Astc12x10Srgb, .vulkan = VK_FORMAT_ASTC_12x10_SRGB_BLOCK},
    FormatPair{.format = Format::Astc12x10Float, .vulkan = VK_FORMAT_ASTC_12x10_SFLOAT_BLOCK},
    FormatPair{.format = Format::Astc12x12Unorm, .vulkan = VK_FORMAT_ASTC_12x12_UNORM_BLOCK},
    FormatPair{.format = Format::Astc12x12Srgb, .vulkan = VK_FORMAT_ASTC_12x12_SRGB_BLOCK},
    FormatPair{.format = Format::Astc12x12Float, .vulkan = VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK},
};
static_assert(kVulkanFormats.size() == kFormats.size());

struct ColorSpacePair {
    ColorSpace colorSpace;
    VkColorSpaceKHR vulkan;
};

constexpr std::array kVulkanColorSpaces{
    ColorSpacePair{.colorSpace = ColorSpace::Srgb, .vulkan = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR},
    ColorSpacePair{.colorSpace = ColorSpace::SrgbExtendedLinear, .vulkan = VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT},
    ColorSpacePair{.colorSpace = ColorSpace::SrgbExtended, .vulkan = VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT},
    ColorSpacePair{.colorSpace = ColorSpace::DisplayP3, .vulkan = VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT},
    ColorSpacePair{.colorSpace = ColorSpace::DisplayP3Linear, .vulkan = VK_COLOR_SPACE_DISPLAY_P3_LINEAR_EXT},
    ColorSpacePair{.colorSpace = ColorSpace::DciP3, .vulkan = VK_COLOR_SPACE_DCI_P3_NONLINEAR_EXT},
    ColorSpacePair{.colorSpace = ColorSpace::Bt709, .vulkan = VK_COLOR_SPACE_BT709_NONLINEAR_EXT},
    ColorSpacePair{.colorSpace = ColorSpace::Bt709Linear, .vulkan = VK_COLOR_SPACE_BT709_LINEAR_EXT},
    ColorSpacePair{.colorSpace = ColorSpace::Bt2020Linear, .vulkan = VK_COLOR_SPACE_BT2020_LINEAR_EXT},
    ColorSpacePair{.colorSpace = ColorSpace::Hdr10Pq, .vulkan = VK_COLOR_SPACE_HDR10_ST2084_EXT},
    ColorSpacePair{.colorSpace = ColorSpace::Hdr10Hlg, .vulkan = VK_COLOR_SPACE_HDR10_HLG_EXT},
    ColorSpacePair{.colorSpace = ColorSpace::DolbyVision, .vulkan = VK_COLOR_SPACE_DOLBYVISION_EXT},
    ColorSpacePair{.colorSpace = ColorSpace::AdobeRgb, .vulkan = VK_COLOR_SPACE_ADOBERGB_NONLINEAR_EXT},
    ColorSpacePair{.colorSpace = ColorSpace::AdobeRgbLinear, .vulkan = VK_COLOR_SPACE_ADOBERGB_LINEAR_EXT},
    ColorSpacePair{.colorSpace = ColorSpace::PassThrough, .vulkan = VK_COLOR_SPACE_PASS_THROUGH_EXT},
    ColorSpacePair{.colorSpace = ColorSpace::DisplayNative, .vulkan = VK_COLOR_SPACE_DISPLAY_NATIVE_AMD},
};

std::optional<ColorSpace> colorSpaceOf(VkColorSpaceKHR colorSpace) {
    for (const ColorSpacePair& pair : kVulkanColorSpaces) {
        if (pair.vulkan == colorSpace) {
            return pair.colorSpace;
        }
    }
    return std::nullopt;
}

DisplayRotation rotationOf(VkSurfaceTransformFlagBitsKHR transform) {
    switch (transform) {
        case VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR:
            return DisplayRotation::None;
        case VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR:
            return DisplayRotation::Rotate90;
        case VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR:
            return DisplayRotation::Rotate180;
        case VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR:
            return DisplayRotation::Rotate270;
        default:
            return DisplayRotation::Other;
    }
}

bool has(VkFormatFeatureFlags2 features, VkFormatFeatureFlags2 feature) {
    return (features & feature) != 0;
}

} // namespace

FloorFeatures::FloorFeatures() {
    vulkan12_.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12_.pNext = &vulkan13_;
    vulkan13_.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    for (const auto& feature : kFloor12) {
        vulkan12_.*feature.member = VK_TRUE;
    }
    for (const auto& feature : kFloor13) {
        vulkan13_.*feature.member = VK_TRUE;
    }
}

std::vector<std::string> missingFloorFeatures(const InstanceTable& vk, VkPhysicalDevice adapter) {
    const FeatureQuery features(vk, adapter, {});
    std::vector<std::string> missing;
    for (const auto& feature : kFloor12) {
        if (!on(features.vulkan12.*feature.member)) {
            missing.emplace_back(feature.name);
        }
    }
    for (const auto& feature : kFloor13) {
        if (!on(features.vulkan13.*feature.member)) {
            missing.emplace_back(feature.name);
        }
    }
    return missing;
}

Caps readCaps(const InstanceTable& vk, VkPhysicalDevice adapter, std::span<const VkExtensionProperties> extensions) {
    const FeatureQuery features(vk, adapter, extensions);
    const PropertyQuery properties(vk, adapter, extensions);
    return {
        .info = infoFrom(properties),
        .limits = limitsFrom(properties, features),
        .bindless = bindlessFrom(properties),
        .compute = computeFrom(properties),
        .mesh = meshFrom(properties, features),
        .rayTracing = rayTracingFrom(properties, features),
        .features = featuresFrom(features, properties, extensions),
        .memory = memoryFrom(vk, adapter, properties.core.properties.deviceType),
        .queues = queuesFrom(vk, adapter),
    };
}

core::Result<FormatCaps> readFormatCaps(const InstanceTable& vk, VkPhysicalDevice adapter, Format format) {
    const VkFormat vulkan = vulkanFormat(format);
    VkFormatProperties3 properties3{};
    properties3.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3;
    VkFormatProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2;
    properties.pNext = &properties3;
    vk.vkGetPhysicalDeviceFormatProperties2(adapter, vulkan, &properties);
    const VkFormatFeatureFlags2 texture = properties3.optimalTilingFeatures;
    const VkFormatFeatureFlags2 buffer = properties3.bufferFeatures;
    FormatCaps caps{
        .format = format,
        .sampled = has(texture, VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT),
        .filterLinear = has(texture, VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT),
        .filterMinmax = has(texture, VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_MINMAX_BIT),
        .storage = has(texture, VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT),
        .storageAtomic = has(texture, VK_FORMAT_FEATURE_2_STORAGE_IMAGE_ATOMIC_BIT),
        .colorAttachment = has(texture, VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT),
        .blend = has(texture, VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BLEND_BIT),
        .depthStencil = has(texture, VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT),
        .copySrc = has(texture, VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT),
        .copyDst = has(texture, VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT),
        .texelBuffer = has(buffer, VK_FORMAT_FEATURE_2_UNIFORM_TEXEL_BUFFER_BIT),
        .storageTexelBuffer = has(buffer, VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_BIT),
        .storageTexelBufferAtomic = has(buffer, VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_ATOMIC_BIT),
    };
    if (!caps.colorAttachment && !caps.depthStencil) {
        return caps;
    }
    VkPhysicalDeviceImageFormatInfo2 info{};
    info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2;
    info.format = vulkan;
    info.type = VK_IMAGE_TYPE_2D;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = caps.colorAttachment ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                                      : VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    VkImageFormatProperties2 image{};
    image.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2;
    const VkResult result = vk.vkGetPhysicalDeviceImageFormatProperties2(adapter, &info, &image);
    if (result == VK_SUCCESS) {
        caps.sampleCounts = image.imageFormatProperties.sampleCounts;
    }
    else if (result != VK_ERROR_FORMAT_NOT_SUPPORTED) {
        return failure(core::ErrorCode::RhiQueryFailed,
                       {.call = "vkGetPhysicalDeviceImageFormatProperties2", .result = result});
    }
    return caps;
}

core::Result<SwapchainCaps> readSurfaceCaps(const InstanceTable& vk, VkPhysicalDevice adapter, VkSurfaceKHR surface) {
    VkSurfaceCapabilitiesKHR capabilities{};
    if (const VkResult result = vk.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(adapter, surface, &capabilities);
        result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiWindowFailed,
                       {.call = "vkGetPhysicalDeviceSurfaceCapabilitiesKHR", .result = result});
    }
    const auto modes = listAll<VkPresentModeKHR>(
        "vkGetPhysicalDeviceSurfacePresentModesKHR",
        [&vk, adapter, surface](std::uint32_t* count, VkPresentModeKHR* listed) {
            return vk.vkGetPhysicalDeviceSurfacePresentModesKHR(adapter, surface, count, listed);
        });
    if (!modes) {
        return failure(core::ErrorCode::RhiWindowFailed, modes.error());
    }
    return surfaceCapsOf(capabilities, *modes);
}

SwapchainCaps surfaceCapsOf(const VkSurfaceCapabilitiesKHR& capabilities,
                            std::span<const VkPresentModeKHR> presentModes) {
    const auto presents = [presentModes](VkPresentModeKHR mode) {
        return std::ranges::find(presentModes, mode) != presentModes.end();
    };
    const VkCompositeAlphaFlagsKHR alpha = capabilities.supportedCompositeAlpha;
    return SwapchainCaps{
        .minImageCount = capabilities.minImageCount,
        .maxImageCount = capabilities.maxImageCount,
        .width = capabilities.currentExtent.width,
        .height = capabilities.currentExtent.height,
        .minWidth = capabilities.minImageExtent.width,
        .minHeight = capabilities.minImageExtent.height,
        .maxWidth = capabilities.maxImageExtent.width,
        .maxHeight = capabilities.maxImageExtent.height,
        .rotation = rotationOf(capabilities.currentTransform),
        .fifo = presents(VK_PRESENT_MODE_FIFO_KHR),
        .fifoRelaxed = presents(VK_PRESENT_MODE_FIFO_RELAXED_KHR),
        .mailbox = presents(VK_PRESENT_MODE_MAILBOX_KHR),
        .immediate = presents(VK_PRESENT_MODE_IMMEDIATE_KHR),
        .alphaOpaque = (alpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) != 0,
        .alphaPremultiplied = (alpha & VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR) != 0,
        .alphaPostmultiplied = (alpha & VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR) != 0,
        .alphaInherit = (alpha & VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR) != 0,
    };
}

core::Result<std::vector<SwapchainFormat>> readSurfaceFormats(const InstanceTable& vk, VkPhysicalDevice adapter,
                                                              VkSurfaceKHR surface) {
    const auto listed = listAll<VkSurfaceFormatKHR>("vkGetPhysicalDeviceSurfaceFormatsKHR",
                                                    [&vk, adapter, surface](std::uint32_t* count,
                                                                            VkSurfaceFormatKHR* formats) {
                                                        return vk.vkGetPhysicalDeviceSurfaceFormatsKHR(adapter, surface,
                                                                                                       count, formats);
                                                    });
    if (!listed) {
        return failure(core::ErrorCode::RhiWindowFailed, listed.error());
    }
    return surfaceFormatsOf(*listed);
}

std::vector<SwapchainFormat> surfaceFormatsOf(std::span<const VkSurfaceFormatKHR> formats) {
    std::vector<SwapchainFormat> named;
    for (const VkSurfaceFormatKHR& vulkan : formats) {
        const auto format = formatOf(vulkan.format);
        const auto colorSpace = colorSpaceOf(vulkan.colorSpace);
        if (format && colorSpace) {
            named.push_back({.format = *format, .colorSpace = *colorSpace});
        }
    }
    return named;
}

MemoryBudget readBudget(const InstanceTable& vk, VkPhysicalDevice adapter) {
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budgets{};
    budgets.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    VkPhysicalDeviceMemoryProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    properties.pNext = &budgets;
    vk.vkGetPhysicalDeviceMemoryProperties2(adapter, &properties);
    const std::uint32_t heaps = properties.memoryProperties.memoryHeapCount;
    const auto budget = std::span{budgets.heapBudget}.first(heaps);
    const auto usage = std::span{budgets.heapUsage}.first(heaps);
    MemoryBudget memory;
    for (std::size_t heap = 0; heap < heaps; ++heap) {
        memory.heaps.push_back({.budget = budget[heap], .usage = usage[heap]});
    }
    return memory;
}

VkFormat vulkanFormat(Format format) {
    return std::ranges::find(kVulkanFormats, format, &FormatPair::format)->vulkan;
}

VkColorSpaceKHR vulkanColorSpace(ColorSpace colorSpace) {
    return std::ranges::find(kVulkanColorSpaces, colorSpace, &ColorSpacePair::colorSpace)->vulkan;
}

std::optional<Format> formatOf(VkFormat format) {
    for (const FormatPair& pair : kVulkanFormats) {
        if (pair.vulkan == format) {
            return pair.format;
        }
    }
    return std::nullopt;
}

} // namespace bfr::rhi::vulkan
