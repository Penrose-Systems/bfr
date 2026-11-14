#include "vulkan/vulkan_caps.h"

#include "bfr/core/error.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/pipeline.h"
#include "vulkan/instance_table.h"
#include "vulkan/queue_families.h"
#include "vulkan/vulkan_list.h"
#include "vulkan/vulkan_pipeline.h"
#include "vulkan/vulkan_result.h"

#include <volk.h>
#include <vulkan/vulkan_beta.h>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace bfr::rhi::vulkan {

namespace {

using Vulkan12 = VkPhysicalDeviceVulkan12Features;

// A Vulkan 1.2 feature BFR needs: its Vulkan name, and its member.
struct Requirement {
    const char* name;
    VkBool32 Vulkan12::* member;
};

// Bindless takes descriptor indexing with update-after-bind and partially
// bound tables of the three kinds of descriptor it holds, indexed by a value
// that may differ across invocations for sampled textures and storage buffers;
// queue progress takes timelines. Storage textures indexed that way are
// reported (storageTextureNonUniformIndexing): descriptorIndexing's minimum
// set (features.adoc:910-913) leaves that feature out, as vk.xml's
// requirements for it say (vkdocs xml/vk.xml:20980-20996).
constexpr std::array kRequired12{
    Requirement{.name = "descriptorIndexing", .member = &Vulkan12::descriptorIndexing},
    Requirement{.name = "runtimeDescriptorArray", .member = &Vulkan12::runtimeDescriptorArray},
    Requirement{.name = "descriptorBindingPartiallyBound", .member = &Vulkan12::descriptorBindingPartiallyBound},
    Requirement{.name = "descriptorBindingSampledImageUpdateAfterBind",
                .member = &Vulkan12::descriptorBindingSampledImageUpdateAfterBind},
    Requirement{.name = "descriptorBindingStorageImageUpdateAfterBind",
                .member = &Vulkan12::descriptorBindingStorageImageUpdateAfterBind},
    Requirement{.name = "descriptorBindingStorageBufferUpdateAfterBind",
                .member = &Vulkan12::descriptorBindingStorageBufferUpdateAfterBind},
    Requirement{.name = "shaderSampledImageArrayNonUniformIndexing",
                .member = &Vulkan12::shaderSampledImageArrayNonUniformIndexing},
    Requirement{.name = "shaderStorageBufferArrayNonUniformIndexing",
                .member = &Vulkan12::shaderStorageBufferArrayNonUniformIndexing},
    Requirement{.name = "timelineSemaphore", .member = &Vulkan12::timelineSemaphore},
};

// Makes next the last link of the chain whose last pNext tail points at.
template <class Struct>
void link(void**& tail, Struct& next) {
    *tail = &next;
    tail = &next.pNext;
}

// The property structs one query fills: those of each core version up to
// version, and an extension's only when the adapter lists it, since chaining
// one it lacks is invalid. Below 1.3, the extensions 1.3 made core stand in
// for its struct. The chain points into itself, so it stays where it is made.
struct PropertyQuery {
    PropertyQuery(const InstanceTable& vk, VkPhysicalDevice adapter, std::uint32_t version,
                  std::span<const VkExtensionProperties> extensions)
        : at13(hasCore13(version)) {
        core.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        vulkan11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES;
        vulkan12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES;
        vulkan13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES;
        maintenance4.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES_KHR;
        subgroupSizeControl.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT;
        accelerationStructure.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
        rayTracingPipeline.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
        invocationReorder.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_PROPERTIES_EXT;
        opacityMicromap.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_PROPERTIES_EXT;
        partitioned.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PARTITIONED_ACCELERATION_STRUCTURE_PROPERTIES_NV;
        clusters.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_PROPERTIES_NV;
        mesh.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT;
        shadingRate.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_PROPERTIES_KHR;
        customBorderColor.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_PROPERTIES_EXT;

        void** tail = &core.pNext;
        link(tail, vulkan11);
        link(tail, vulkan12);
        const auto linkIfListed = [&tail, extensions](const char* extension, auto& next) {
            if (lists(extensions, extension)) {
                link(tail, next);
            }
        };
        if (at13) {
            link(tail, vulkan13);
        }
        else {
            linkIfListed(VK_KHR_MAINTENANCE_4_EXTENSION_NAME, maintenance4);
            linkIfListed(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME, subgroupSizeControl);
        }
        linkIfListed(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, accelerationStructure);
        linkIfListed(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, rayTracingPipeline);
        linkIfListed(VK_EXT_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME, invocationReorder);
        linkIfListed(VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME, opacityMicromap);
        linkIfListed(VK_NV_PARTITIONED_ACCELERATION_STRUCTURE_EXTENSION_NAME, partitioned);
        linkIfListed(VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME, clusters);
        linkIfListed(VK_EXT_MESH_SHADER_EXTENSION_NAME, mesh);
        linkIfListed(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME, shadingRate);
        linkIfListed(VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME, customBorderColor);
        vk.vkGetPhysicalDeviceProperties2(adapter, &core);
    }
    ~PropertyQuery() = default;
    PropertyQuery(const PropertyQuery&) = delete;
    PropertyQuery& operator=(const PropertyQuery&) = delete;
    PropertyQuery(PropertyQuery&&) = delete;
    PropertyQuery& operator=(PropertyQuery&&) = delete;

    // What 1.3 made core: its struct's from 1.3, else the extension's, 0
    // where the adapter lists neither.
    [[nodiscard]] VkDeviceSize maxBufferSize() const {
        return at13 ? vulkan13.maxBufferSize : maintenance4.maxBufferSize;
    }
    [[nodiscard]] std::uint32_t minSubgroupSize() const {
        return at13 ? vulkan13.minSubgroupSize : subgroupSizeControl.minSubgroupSize;
    }
    [[nodiscard]] std::uint32_t maxSubgroupSize() const {
        return at13 ? vulkan13.maxSubgroupSize : subgroupSizeControl.maxSubgroupSize;
    }

    bool at13;
    VkPhysicalDeviceProperties2 core{};
    VkPhysicalDeviceVulkan11Properties vulkan11{};
    VkPhysicalDeviceVulkan12Properties vulkan12{};
    VkPhysicalDeviceVulkan13Properties vulkan13{};
    VkPhysicalDeviceMaintenance4PropertiesKHR maintenance4{};
    VkPhysicalDeviceSubgroupSizeControlPropertiesEXT subgroupSizeControl{};
    VkPhysicalDeviceAccelerationStructurePropertiesKHR accelerationStructure{};
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingPipeline{};
    VkPhysicalDeviceRayTracingInvocationReorderPropertiesEXT invocationReorder{};
    VkPhysicalDeviceOpacityMicromapPropertiesEXT opacityMicromap{};
    VkPhysicalDevicePartitionedAccelerationStructurePropertiesNV partitioned{};
    VkPhysicalDeviceClusterAccelerationStructurePropertiesNV clusters{};
    VkPhysicalDeviceMeshShaderPropertiesEXT mesh{};
    VkPhysicalDeviceFragmentShadingRatePropertiesKHR shadingRate{};
    VkPhysicalDeviceCustomBorderColorPropertiesEXT customBorderColor{};
};

bool on(VkBool32 value) {
    return value == VK_TRUE;
}

// Builds read their inputs and scratch by device address
// (VUID-vkGetBufferDeviceAddress-bufferDeviceAddress-03324).
bool hasAccelerationStructures(const DeviceFeatures& features) {
    return on(features.accelerationStructure.accelerationStructure) && on(features.vulkan12.bufferDeviceAddress);
}

// A custom border colour needs both features: BFR's sampler desc gives no
// format to go with it (VUID-VkSamplerCustomBorderColorCreateInfoEXT-format-04014).
bool customBorderColors(const DeviceFeatures& features) {
    return on(features.customBorderColor.customBorderColors) &&
           on(features.customBorderColor.customBorderColorWithoutFormat);
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

Limits limitsFrom(const PropertyQuery& properties, const DeviceFeatures& features) {
    const VkPhysicalDeviceLimits& core = properties.core.properties.limits;
    return {
        .maxTexture1d = core.maxImageDimension1D,
        .maxTexture2d = core.maxImageDimension2D,
        .maxTexture3d = core.maxImageDimension3D,
        .maxTextureCube = core.maxImageDimensionCube,
        .maxTextureLayers = core.maxImageArrayLayers,
        .maxBufferSize = properties.maxBufferSize(),
        .maxAllocationSize = properties.vulkan11.maxMemoryAllocationSize,
        .maxAllocations = core.maxMemoryAllocationCount,
        .maxSamplerObjects = core.maxSamplerAllocationCount,
        .maxUniformBufferRange = core.maxUniformBufferRange,
        .maxStorageBufferRange = core.maxStorageBufferRange,
        .maxTexelBufferElements = core.maxTexelBufferElements,
        .maxPushConstantsSize = core.maxPushConstantsSize,
        // The device's count, up to the kMaxColorTargets a rendering holds.
        .maxColorAttachments = std::min(core.maxColorAttachments, kMaxColorTargets),
        .maxRenderWidth = core.maxFramebufferWidth,
        .maxRenderHeight = core.maxFramebufferHeight,
        .maxRenderLayers = core.maxFramebufferLayers,
        .colorSampleCounts = core.framebufferColorSampleCounts,
        .integerSampleCounts = properties.vulkan12.framebufferIntegerColorSampleCounts,
        .depthSampleCounts = core.framebufferDepthSampleCounts,
        .stencilSampleCounts = core.framebufferStencilSampleCounts,
        .noAttachmentSampleCounts = core.framebufferNoAttachmentsSampleCounts,
        .storageTextureSampleCounts = core.storageImageSampleCounts,
        // A pipeline and setViewport take one viewport.
        .maxViewports = kViewports,
        .maxViewportWidth = core.maxViewportDimensions[0],
        .maxViewportHeight = core.maxViewportDimensions[1],
        .maxDrawIndirectCount = core.maxDrawIndirectCount,
        .maxClipDistances = core.maxClipDistances,
        .maxCullDistances = core.maxCullDistances,
        .maxSamplerAnisotropy = core.maxSamplerAnisotropy,
        .maxSamplerLodBias = core.maxSamplerLodBias,
        .maxCustomBorderColorSamplers = customBorderColors(features)
                                            ? properties.customBorderColor.maxCustomBorderColorSamplers
                                            : 0,
        // Set by readCaps, from the features it reports.
        .minShadingRateTile = {},
        .maxShadingRateTile = {},
        .minUniformBufferAlignment = core.minUniformBufferOffsetAlignment,
        .minStorageBufferAlignment = core.minStorageBufferOffsetAlignment,
        .minTexelBufferAlignment = core.minTexelBufferOffsetAlignment,
        .copyOffsetAlignment = core.optimalBufferCopyOffsetAlignment,
        .copyRowPitchAlignment = core.optimalBufferCopyRowPitchAlignment,
        .nonCoherentAtomSize = core.nonCoherentAtomSize,
        .bufferTextureGranularity = core.bufferImageGranularity,
    };
}

BindlessLimits bindlessFrom(const PropertyQuery& properties, const DeviceFeatures& features) {
    const VkPhysicalDeviceVulkan12Properties& vulkan12 = properties.vulkan12;
    // The table updates after bind, as the others do, which needs its feature
    // (VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-descriptorBindingAccelerationStructureUpdateAfterBind-03570).
    const bool structures = hasAccelerationStructures(features) &&
                            on(features.accelerationStructure.descriptorBindingAccelerationStructureUpdateAfterBind);
    return {
        .maxTextures = vulkan12.maxDescriptorSetUpdateAfterBindSampledImages,
        .maxStorageTextures = vulkan12.maxDescriptorSetUpdateAfterBindStorageImages,
        .maxStorageBuffers = vulkan12.maxDescriptorSetUpdateAfterBindStorageBuffers,
        // The table updates after bind, as the others do, which needs its
        // feature
        // (VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-descriptorBindingUniformBufferUpdateAfterBind-03005).
        .maxUniformBuffers = on(features.vulkan12.descriptorBindingUniformBufferUpdateAfterBind)
                                 ? vulkan12.maxDescriptorSetUpdateAfterBindUniformBuffers
                                 : 0,
        .maxSamplers = vulkan12.maxDescriptorSetUpdateAfterBindSamplers,
        .maxPerStage = vulkan12.maxPerStageUpdateAfterBindResources,
        .maxAccelerationStructures =
            structures ? properties.accelerationStructure.maxDescriptorSetUpdateAfterBindAccelerationStructures : 0,
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
        .minSubgroupSize = properties.minSubgroupSize(),
        .maxSubgroupSize = properties.maxSubgroupSize(),
    };
}

Features featuresFrom(const DeviceFeatures& features, const PropertyQuery& properties,
                      std::span<const VkExtensionProperties> extensions) {
    const VkPhysicalDeviceFeatures& core = features.core.features;
    return {
        // Vulkan's ray queries run in every stage.
        .rayQuery = hasAccelerationStructures(features) && on(features.rayQuery.rayQuery),
        .rayQueryGraphics = hasAccelerationStructures(features) && on(features.rayQuery.rayQuery),
        .rayTracingPipeline = hasAccelerationStructures(features) && on(features.rayTracingPipeline.rayTracingPipeline),
        // BFR's indirect trace is vkCmdTraceRaysIndirect2KHR.
        .traceRaysIndirect = hasAccelerationStructures(features) &&
                             on(features.rayTracingPipeline.rayTracingPipeline) &&
                             on(features.rayTracingMaintenance1.rayTracingPipelineTraceRaysIndirect2),
        .rayPrimitiveCulling = hasAccelerationStructures(features) &&
                               on(features.rayTracingPipeline.rayTraversalPrimitiveCulling),
        .accelerationStructure = hasAccelerationStructures(features),
        .bufferAddress = on(features.vulkan12.bufferDeviceAddress),
        .rayPositionFetch = hasAccelerationStructures(features) &&
                            on(features.rayTracingPositionFetch.rayTracingPositionFetch),
        // Core to VK_KHR_acceleration_structure (vkCmdCopyAccelerationStructureToMemoryKHR).
        .accelerationStructureSerialization = hasAccelerationStructures(features),
        // Ray tracing maintenance1's query types (queries.adoc:472, 476).
        .accelerationStructureCurrentSize = hasAccelerationStructures(features) &&
                                            on(features.rayTracingMaintenance1.rayTracingMaintenance1),
        .serializedBottomCount = hasAccelerationStructures(features) &&
                                 on(features.rayTracingMaintenance1.rayTracingMaintenance1),
        .rayInvocationReorder = hasAccelerationStructures(features) &&
                                on(features.rayTracingPipeline.rayTracingPipeline) &&
                                on(features.invocationReorder.rayTracingInvocationReorder),
        .rayTracingPipelineLibrary = hasAccelerationStructures(features) &&
                                     on(features.rayTracingPipeline.rayTracingPipeline) && features.pipelineLibrary,
        .pipelineLibraryIdentifiers = hasAccelerationStructures(features) &&
                                      on(features.rayTracingPipeline.rayTracingPipeline) && features.pipelineLibrary &&
                                      on(features.libraryGroupHandles.pipelineLibraryGroupHandles),
        .rayTracingCaptureReplay = hasAccelerationStructures(features) &&
                                   on(features.rayTracingPipeline.rayTracingPipeline) &&
                                   on(features.rayTracingPipeline.rayTracingPipelineShaderGroupHandleCaptureReplay),
        .rayTracingCaptureReplayMixed =
            hasAccelerationStructures(features) && on(features.rayTracingPipeline.rayTracingPipeline) &&
            on(features.rayTracingPipeline.rayTracingPipelineShaderGroupHandleCaptureReplay) &&
            on(features.rayTracingPipeline.rayTracingPipelineShaderGroupHandleCaptureReplayMixed),
        .accelerationStructureIndirectBuild = hasAccelerationStructures(features) &&
                                              on(features.accelerationStructure.accelerationStructureIndirectBuild),
        .opacityMicromap = hasAccelerationStructures(features) && on(features.opacityMicromap.micromap),
        .rayTracingMotion = hasAccelerationStructures(features) && on(features.rayTracingPipeline.rayTracingPipeline) &&
                            on(features.motionBlur.rayTracingMotionBlur),
        .rayTracingMotionIndirect = hasAccelerationStructures(features) &&
                                    on(features.rayTracingPipeline.rayTracingPipeline) &&
                                    on(features.motionBlur.rayTracingMotionBlur) &&
                                    on(features.motionBlur.rayTracingMotionBlurPipelineTraceRaysIndirect),
        .partitionedTop = hasAccelerationStructures(features) &&
                          on(features.partitioned.partitionedAccelerationStructure),
        .clusterAccelerationStructure = hasAccelerationStructures(features) &&
                                        on(features.clusters.clusterAccelerationStructure),
        // A replayed structure's buffer and memory are replayed too
        // (VUID-VkAccelerationStructureCreateInfoKHR-deviceAddress-09490,
        // VUID-VkBufferCreateInfo-flags-03338).
        .accelerationStructureCaptureReplay = hasAccelerationStructures(features) &&
                                              on(features.accelerationStructure.accelerationStructureCaptureReplay) &&
                                              on(features.vulkan12.bufferDeviceAddressCaptureReplay),
        .meshShader = on(features.mesh.meshShader),
        .taskShader = on(features.mesh.taskShader),
        // BFR has no call for shading rates or conservative raster yet (roadmap step
        // 14), so the device enables none of their extensions.
        .shadingRate = false,
        .shadingRateTexture = false,
        .shaderFloat16 = on(features.vulkan12.shaderFloat16),
        .shaderFloat64 = on(core.shaderFloat64),
        .shaderInt8 = on(features.vulkan12.shaderInt8),
        .shaderInt16 = on(core.shaderInt16),
        .shaderInt64 = on(core.shaderInt64),
        .atomicInt64 = on(features.vulkan12.shaderBufferInt64Atomics),
        .atomicInt64Texture = on(features.imageAtomicInt64.shaderImageInt64Atomics),
        .atomicFloat32Add = on(features.atomicFloat.shaderBufferFloat32AtomicAdd),
        .barycentrics = on(features.barycentric.fragmentShaderBarycentric),
        .conservativeRaster = false,
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
        .samplerMirrorClampToEdge = on(features.vulkan12.samplerMirrorClampToEdge),
        // The spec forbids neither with anisotropy, though its Texel
        // Anisotropic Filtering section warns that vendors read nearest
        // filters differently. Integer border presets and unnormalized
        // coordinates are core 1.0, with no feature.
        .samplerAnisotropyNearestMip = on(core.samplerAnisotropy),
        .samplerAnisotropyNearestFilter = on(core.samplerAnisotropy),
        .samplerCustomBorderColor = customBorderColors(features),
        .samplerIntegerBorderColor = true,
        .samplerUnnormalizedCoordinates = true,
        // Through 2D views of a 3D image, which a portability driver says
        // whether it makes (VUID-VkImageCreateInfo-imageView2DOn3DImage-04459).
        .volumeColorTargets = !lists(extensions, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME) ||
                              on(features.portability.imageView2DOn3DImage),
        // Core, but a portability driver says whether it has it
        // (VUID-VkSamplerCreateInfo-samplerMipLodBias-04467).
        .samplerLodBias = !lists(extensions, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME) ||
                          on(features.portability.samplerMipLodBias),
        .compressionBc = on(core.textureCompressionBC),
        .compressionEtc2 = on(core.textureCompressionETC2),
        .compressionAstc = on(core.textureCompressionASTC_LDR),
        .compressionAstcHdr = features.hasAstcHdr(),
        .multiDrawIndirect = on(core.multiDrawIndirect),
        .drawIndirectCount = on(features.vulkan12.drawIndirectCount),
        .drawIndirectFirstInstance = on(core.drawIndirectFirstInstance),
        // Core, but a portability driver says whether it has them
        // (VUID-VkPipelineInputAssemblyStateCreateInfo-triangleFans-04452).
        .triangleFans = !lists(extensions, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME) ||
                        on(features.portability.triangleFans),
        // VUID-VkPipelineInputAssemblyStateCreateInfo-topology-06252.
        .listRestart = on(features.listRestart.primitiveTopologyListRestart),
        .dynamicTopology = features.hasDynamicTopology(),
        .meshMultiview = on(features.mesh.multiviewMeshShader),
        .meshStatistics = on(features.mesh.meshShaderQueries),
        .alphaToOne = on(core.alphaToOne),
        // pSampleMask is core 1.0, with no feature.
        .sampleMask = true,
        // VkPipelineCache is core 1.0, with no feature.
        .pipelineCache = true,
        .storageReadWithoutFormat = on(core.shaderStorageImageReadWithoutFormat),
        .storageWriteWithoutFormat = on(core.shaderStorageImageWriteWithoutFormat),
        .storageTextureNonUniformIndexing = on(features.vulkan12.shaderStorageImageArrayNonUniformIndexing),
        .uniformBufferNonUniformIndexing = on(features.vulkan12.shaderUniformBufferArrayNonUniformIndexing),
        .timestampsOnAllQueues = on(properties.core.properties.limits.timestampComputeAndGraphics),
        // VK_QUERY_TYPE_OCCLUSION is core 1.0, with no feature
        // (VUID-vkCmdBeginQuery-queryType-00803 asks only a graphics queue).
        .queries = true,
        .pipelineStatistics = on(core.pipelineStatisticsQuery),
        .occlusionQueryPrecise = on(core.occlusionQueryPrecise),
    };
}

// The sparse features and properties Vulkan reports. Vulkan has no heap
// kinds: a resource maps to any VkDeviceMemory of a type in its
// memoryTypeBits, so textures and buffers share one where binding is on. Each
// layer's tail sits at imageMipTailOffset plus the layer's imageMipTailStride
// (VkSparseImageMemoryRequirements), unless the format reports one tail for
// all. Vulkan sets the tile sizes itself, so tileBytes is empty.
Sparse sparseFrom(const DeviceFeatures& features, const PropertyQuery& properties) {
    const VkPhysicalDeviceFeatures& core = features.core.features;
    const VkPhysicalDeviceProperties& device = properties.core.properties;
    const VkPhysicalDeviceSparseProperties& sparse = device.sparseProperties;
    const bool binding = on(core.sparseBinding);
    return {
        .binding = binding,
        .buffers = on(core.sparseResidencyBuffer),
        .textures2D = on(core.sparseResidencyImage2D),
        .textures3D = on(core.sparseResidencyImage3D),
        .samples2 = on(core.sparseResidency2Samples),
        .samples4 = on(core.sparseResidency4Samples),
        .samples8 = on(core.sparseResidency8Samples),
        .samples16 = on(core.sparseResidency16Samples),
        .aliased = on(core.sparseResidencyAliased),
        .standard2DShape = on(sparse.residencyStandard2DBlockShape),
        .standard2DMultisampleShape = on(sparse.residencyStandard2DMultisampleBlockShape),
        .standard3DShape = on(sparse.residencyStandard3DBlockShape),
        .alignedMipSize = on(sparse.residencyAlignedMipSize),
        .nonResidentStrict = on(sparse.residencyNonResidentStrict),
        .shaderResidency = on(core.shaderResourceResidency),
        .mixedResources = binding,
        .layerTails = binding,
        .addressSpaceBytes = device.limits.sparseAddressSpaceSize,
        .addressBitsPerResource = 0,
        .addressBitsPerProcess = 0,
        .tileBytes = {},
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

// The levels a family lists in its VkQueueFamilyGlobalPriorityPropertiesKHR.
QueueLevels levelsFrom(const VkQueueFamilyGlobalPriorityPropertiesKHR& listed) {
    QueueLevels levels;
    for (const VkQueueGlobalPriorityKHR level : std::span{listed.priorities}.first(
             std::min<std::size_t>(listed.priorityCount, VK_MAX_GLOBAL_PRIORITY_SIZE_KHR))) {
        switch (level) {
            case VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR:
                levels.low = true;
                break;
            case VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR:
                levels.normal = true;
                break;
            case VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR:
                levels.high = true;
                break;
            case VK_QUEUE_GLOBAL_PRIORITY_REALTIME_KHR:
                levels.realtime = true;
                break;
            default:
                break;
        }
    }
    return levels;
}

// Each kind's queues, timestamp bits and levels on the family a device opens
// it on, none for a kind the device can't open. The levels are each family's
// list where the adapter has VK_KHR_global_priority, none without it; ranks
// is the adapter's discreteQueuePriorities.
// The flags of the family each kind of queue opens on; 0 for a kind with none.
struct KindFlags {
    VkQueueFlags graphics = 0;
    VkQueueFlags compute = 0;
    VkQueueFlags copy = 0;
};

// The stages a timestamp takes on a queue of a family with flags: graphics
// stages only on a graphics queue (synchronization.adoc:838-841), compute on
// a compute queue, copies on a graphics, compute or transfer one, and the
// start and end of the work before it on any; none where flags is 0. The
// task and mesh stages need their features (VUID-vkCmdWriteTimestamp2-stage-03934,
// -03935); the ray tracing and build stages need theirs (-07947, -10751) and
// a compute queue (vkdocs xml/vk.xml:35792-35797).
TimestampStages stagesFor(VkQueueFlags flags, const Features& features) {
    const bool graphics = (flags & VK_QUEUE_GRAPHICS_BIT) != 0;
    const bool compute = (flags & VK_QUEUE_COMPUTE_BIT) != 0;
    const bool any = flags != 0;
    return {.end = any,
            .start = any,
            .vertex = graphics,
            .fragment = graphics,
            .compute = compute,
            .copy = graphics || compute || (flags & VK_QUEUE_TRANSFER_BIT) != 0,
            .task = graphics && features.taskShader,
            .mesh = graphics && features.meshShader,
            .rayTracing = compute && features.rayTracingPipeline,
            .accelerationStructureBuild = compute && features.accelerationStructure};
}

Queues queuesFrom(const InstanceTable& vk, VkPhysicalDevice adapter, std::uint32_t version,
                  std::span<const VkExtensionProperties> extensions, KindFlags& kinds,
                  const VkPhysicalDeviceLimits& limits) {
    std::uint32_t count = 0;
    vk.vkGetPhysicalDeviceQueueFamilyProperties2(adapter, &count, nullptr);
    const bool listsLevels = hasQueueLevels(version, extensions);
    std::vector<VkQueueFamilyGlobalPriorityPropertiesKHR> listed(count);
    std::vector<VkQueueFamilyProperties2> queried(count);
    for (std::uint32_t index = 0; index < count; ++index) {
        listed[index].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_GLOBAL_PRIORITY_PROPERTIES_KHR;
        queried[index].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
        // Chaining an extension's struct the adapter lacks is invalid.
        queried[index].pNext = listsLevels ? &listed[index] : nullptr;
    }
    vk.vkGetPhysicalDeviceQueueFamilyProperties2(adapter, &count, queried.data());
    std::vector<VkQueueFamilyProperties> families(count);
    std::ranges::transform(queried, families.begin(), &VkQueueFamilyProperties2::queueFamilyProperties);
    // Without a surface. A surface may move each kind to a family of its kind
    // that presents to it, which may have other counts, timestamp bits and
    // levels; bfr_caps_queues says so.
    const QueueFamilies chosen = chooseQueueFamilies(families, {});
    const QueueCounts counts = countQueues(chosen, families);
    const auto flags = [&families](std::uint32_t queues, std::optional<std::uint32_t> family) -> VkQueueFlags {
        return queues == 0 ? 0 : families[*family].queueFlags;
    };
    kinds = {.graphics = flags(counts.graphics, chosen.graphics),
             .compute = flags(counts.compute, chosen.compute),
             .copy = flags(counts.copy, chosen.transfer)};
    const auto bits = [&families](std::uint32_t queues, std::optional<std::uint32_t> family) {
        return queues == 0 ? 0 : families[*family].timestampValidBits;
    };
    const auto levels = [&listed, listsLevels](std::uint32_t queues, std::optional<std::uint32_t> family) {
        return queues == 0 || !listsLevels ? QueueLevels{} : levelsFrom(listed[*family]);
    };
    const auto sparse = [&families](std::uint32_t queues, std::optional<std::uint32_t> family) {
        return queues != 0 && (families[*family].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) != 0;
    };
    // (0,0,0) is whole mip levels only (the spec's VkQueueFamilyProperties).
    const auto granularity = [&families](std::uint32_t queues, std::optional<std::uint32_t> family) -> CopyGranularity {
        if (queues == 0) {
            return {};
        }
        const VkExtent3D extent = families[*family].minImageTransferGranularity;
        return {.width = extent.width,
                .height = extent.height,
                .depth = extent.depth,
                .wholeLevels = extent.width == 0 && extent.height == 0 && extent.depth == 0};
    };
    return {
        .graphics = counts.graphics,
        .compute = counts.compute,
        .copy = counts.copy,
        .graphicsTimestampBits = bits(counts.graphics, chosen.graphics),
        .computeTimestampBits = bits(counts.compute, chosen.compute),
        .copyTimestampBits = bits(counts.copy, chosen.transfer),
        .ranks = limits.discreteQueuePriorities,
        .graphicsLevels = levels(counts.graphics, chosen.graphics),
        .computeLevels = levels(counts.compute, chosen.compute),
        .copyLevels = levels(counts.copy, chosen.transfer),
        .graphicsSparse = sparse(counts.graphics, chosen.graphics),
        .computeSparse = sparse(counts.compute, chosen.compute),
        .copySparse = sparse(counts.copy, chosen.transfer),
        .graphicsGranularity = granularity(counts.graphics, chosen.graphics),
        .computeGranularity = granularity(counts.compute, chosen.compute),
        .copyGranularity = granularity(counts.copy, chosen.transfer),
    };
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
    FormatPair{.format = Format::R64Uint, .vulkan = VK_FORMAT_R64_UINT},
    FormatPair{.format = Format::R64Sint, .vulkan = VK_FORMAT_R64_SINT},
    FormatPair{.format = Format::R64Float, .vulkan = VK_FORMAT_R64_SFLOAT},
    FormatPair{.format = Format::Rg64Uint, .vulkan = VK_FORMAT_R64G64_UINT},
    FormatPair{.format = Format::Rg64Sint, .vulkan = VK_FORMAT_R64G64_SINT},
    FormatPair{.format = Format::Rg64Float, .vulkan = VK_FORMAT_R64G64_SFLOAT},
    FormatPair{.format = Format::Rgb64Uint, .vulkan = VK_FORMAT_R64G64B64_UINT},
    FormatPair{.format = Format::Rgb64Sint, .vulkan = VK_FORMAT_R64G64B64_SINT},
    FormatPair{.format = Format::Rgb64Float, .vulkan = VK_FORMAT_R64G64B64_SFLOAT},
    FormatPair{.format = Format::Rgba64Uint, .vulkan = VK_FORMAT_R64G64B64A64_UINT},
    FormatPair{.format = Format::Rgba64Sint, .vulkan = VK_FORMAT_R64G64B64A64_SINT},
    FormatPair{.format = Format::Rgba64Float, .vulkan = VK_FORMAT_R64G64B64A64_SFLOAT},
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

DeviceFeatures::DeviceFeatures(const InstanceTable& vk, VkPhysicalDevice adapter, std::uint32_t version,
                               std::span<const VkExtensionProperties> extensions)
    : version(version) {
    core.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    vulkan11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    vulkan12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    vulkan14.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
    dynamicRendering.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR;
    synchronization2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR;
    extendedDynamicState.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT;
    astcHdr.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXTURE_COMPRESSION_ASTC_HDR_FEATURES_EXT;
    formats4444.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_4444_FORMATS_FEATURES_EXT;
    maintenance4.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES_KHR;
    subgroupSizeControl.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT;
    integerDotProduct.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR;
    terminateInvocation.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_TERMINATE_INVOCATION_FEATURES_KHR;
    zeroInitializeWorkgroupMemory.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES_KHR;
    extendedDynamicState2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT;
    imageRobustness.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES_EXT;
    inlineUniformBlock.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_FEATURES_EXT;
    pipelineCreationCacheControl.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES_EXT;
    privateData.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES_EXT;
    demoteToHelperInvocation.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES_EXT;
    texelBufferAlignment.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXEL_BUFFER_ALIGNMENT_FEATURES_EXT;
    ycbcr2Plane444Formats.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_YCBCR_2_PLANE_444_FORMATS_FEATURES_EXT;
    atomicFloat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
    imageAtomicInt64.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT;
    barycentric.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR;
    customBorderColor.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT;
    portability.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PORTABILITY_SUBSET_FEATURES_KHR;
    globalPriority.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GLOBAL_PRIORITY_QUERY_FEATURES_KHR;
    maintenance9.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_9_FEATURES_KHR;
    listRestart.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIMITIVE_TOPOLOGY_LIST_RESTART_FEATURES_EXT;
    mesh.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
    accelerationStructure.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    rayQuery.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    rayTracingPipeline.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rayTracingMaintenance1.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_MAINTENANCE_1_FEATURES_KHR;
    rayTracingPositionFetch.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR;
    invocationReorder.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_EXT;
    libraryGroupHandles.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_LIBRARY_GROUP_HANDLES_FEATURES_EXT;
    opacityMicromap.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES_EXT;
    motionBlur.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_MOTION_BLUR_FEATURES_NV;
    partitioned.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PARTITIONED_ACCELERATION_STRUCTURE_FEATURES_NV;
    clusters.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_FEATURES_NV;

    void** tail = &core.pNext;
    link(tail, vulkan11);
    link(tail, vulkan12);
    // Chaining an extension's struct the adapter lacks is invalid.
    const auto linkIfListed = [this, &tail, extensions](const char* extension, auto& next) {
        if (lists(extensions, extension)) {
            link(tail, next);
            enabledExtensions.push_back(extension);
        }
    };
    const auto enableIfListed = [this, extensions](const char* extension) {
        if (lists(extensions, extension)) {
            enabledExtensions.push_back(extension);
        }
    };
    // Below 1.3, every extension 1.3 made core that the adapter lists, so a
    // device has at 1.2 what it would have at 1.3.
    if (hasCore13(version)) {
        link(tail, vulkan13);
    }
    else {
        enableIfListed(VK_KHR_COPY_COMMANDS_2_EXTENSION_NAME);
        enableIfListed(VK_KHR_FORMAT_FEATURE_FLAGS_2_EXTENSION_NAME);
        enableIfListed(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
        enableIfListed(VK_EXT_PIPELINE_CREATION_FEEDBACK_EXTENSION_NAME);
        enableIfListed(VK_EXT_TOOLING_INFO_EXTENSION_NAME);
        linkIfListed(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME, integerDotProduct);
        linkIfListed(VK_KHR_SHADER_TERMINATE_INVOCATION_EXTENSION_NAME, terminateInvocation);
        linkIfListed(VK_KHR_ZERO_INITIALIZE_WORKGROUP_MEMORY_EXTENSION_NAME, zeroInitializeWorkgroupMemory);
        linkIfListed(VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME, extendedDynamicState2);
        linkIfListed(VK_EXT_IMAGE_ROBUSTNESS_EXTENSION_NAME, imageRobustness);
        linkIfListed(VK_EXT_INLINE_UNIFORM_BLOCK_EXTENSION_NAME, inlineUniformBlock);
        linkIfListed(VK_EXT_PIPELINE_CREATION_CACHE_CONTROL_EXTENSION_NAME, pipelineCreationCacheControl);
        linkIfListed(VK_EXT_PRIVATE_DATA_EXTENSION_NAME, privateData);
        linkIfListed(VK_EXT_SHADER_DEMOTE_TO_HELPER_INVOCATION_EXTENSION_NAME, demoteToHelperInvocation);
        linkIfListed(VK_EXT_TEXEL_BUFFER_ALIGNMENT_EXTENSION_NAME, texelBufferAlignment);
        linkIfListed(VK_EXT_YCBCR_2PLANE_444_FORMATS_EXTENSION_NAME, ycbcr2Plane444Formats);
        linkIfListed(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME, dynamicRendering);
        linkIfListed(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME, synchronization2);
        linkIfListed(VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME, extendedDynamicState);
        linkIfListed(VK_EXT_TEXTURE_COMPRESSION_ASTC_HDR_EXTENSION_NAME, astcHdr);
        linkIfListed(VK_EXT_4444_FORMATS_EXTENSION_NAME, formats4444);
        linkIfListed(VK_KHR_MAINTENANCE_4_EXTENSION_NAME, maintenance4);
        linkIfListed(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME, subgroupSizeControl);
    }
    if (version >= VK_API_VERSION_1_4) {
        link(tail, vulkan14);
        if (lists(extensions, VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME)) {
            enabledExtensions.push_back(VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME);
        }
    }
    else {
        linkIfListed(VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME, globalPriority);
    }
    linkIfListed(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME, portability);
    linkIfListed(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, atomicFloat);
    linkIfListed(VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME, imageAtomicInt64);
    linkIfListed(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME, barycentric);
    linkIfListed(VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME, customBorderColor);
    linkIfListed(VK_KHR_MAINTENANCE_9_EXTENSION_NAME, maintenance9);
    linkIfListed(VK_EXT_PRIMITIVE_TOPOLOGY_LIST_RESTART_EXTENSION_NAME, listRestart);
    linkIfListed(VK_EXT_MESH_SHADER_EXTENSION_NAME, mesh);
    // VK_KHR_acceleration_structure needs VK_KHR_deferred_host_operations
    // (vkdocs xml/vk.xml:24210); ray queries and pipelines, and their
    // maintenance, need acceleration structures.
    if (lists(extensions, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)) {
        enableIfListed(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        linkIfListed(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, accelerationStructure);
        linkIfListed(VK_KHR_RAY_QUERY_EXTENSION_NAME, rayQuery);
        linkIfListed(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, rayTracingPipeline);
        linkIfListed(VK_KHR_RAY_TRACING_MAINTENANCE_1_EXTENSION_NAME, rayTracingMaintenance1);
        linkIfListed(VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME, rayTracingPositionFetch);
        linkIfListed(VK_EXT_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME, invocationReorder);
        // VK_EXT_opacity_micromap needs acceleration structures and
        // synchronization2 (vkdocs xml/vk.xml:28172), which BFR's minimum has.
        linkIfListed(VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME, opacityMicromap);
        // VK_NV_ray_tracing_motion_blur needs ray tracing pipelines
        // (vkdocs xml/vk.xml:27270).
        if (lists(extensions, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)) {
            linkIfListed(VK_NV_RAY_TRACING_MOTION_BLUR_EXTENSION_NAME, motionBlur);
        }
        // VK_NV_partitioned_acceleration_structure needs acceleration
        // structures (vcpkg vk.xml:30757).
        linkIfListed(VK_NV_PARTITIONED_ACCELERATION_STRUCTURE_EXTENSION_NAME, partitioned);
        // VK_NV_cluster_acceleration_structure needs acceleration structures
        // (vcpkg vk.xml:30794).
        linkIfListed(VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME, clusters);
        // VK_EXT_pipeline_library_group_handles needs VK_KHR_pipeline_library
        // and VK_KHR_ray_tracing_pipeline (vkdocs xml/vk.xml:29884).
        if (lists(extensions, VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME)) {
            pipelineLibrary = true;
            enabledExtensions.push_back(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
            if (lists(extensions, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)) {
                linkIfListed(VK_EXT_PIPELINE_LIBRARY_GROUP_HANDLES_EXTENSION_NAME, libraryGroupHandles);
            }
        }
    }
    vk.vkGetPhysicalDeviceFeatures2(adapter, &core);
}

bool DeviceFeatures::hasDynamicRendering() const {
    return on(hasCore13(version) ? vulkan13.dynamicRendering : dynamicRendering.dynamicRendering);
}

bool DeviceFeatures::hasSynchronization2() const {
    return on(hasCore13(version) ? vulkan13.synchronization2 : synchronization2.synchronization2);
}

bool DeviceFeatures::hasDynamicTopology() const {
    return hasCore13(version) || on(extendedDynamicState.extendedDynamicState);
}

bool DeviceFeatures::hasAstcHdr() const {
    return on(hasCore13(version) ? vulkan13.textureCompressionASTC_HDR : astcHdr.textureCompressionASTC_HDR);
}

bool hasQueueLevels(std::uint32_t version, std::span<const VkExtensionProperties> extensions) {
    return version >= VK_API_VERSION_1_4 || lists(extensions, VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME);
}

std::vector<std::string> missingRequirements(const DeviceFeatures& features) {
    std::vector<std::string> missing;
    for (const Requirement& feature : kRequired12) {
        if (!on(features.vulkan12.*feature.member)) {
            missing.emplace_back(feature.name);
        }
    }
    if (!features.hasDynamicRendering()) {
        missing.emplace_back("dynamicRendering");
    }
    if (!features.hasSynchronization2()) {
        missing.emplace_back("synchronization2");
    }
    return missing;
}

// The limits of what reported says the device has: none while a feature
// reports false, as shading rate textures do until BFR has a
// call for them (roadmap step 14).
MeshLimits meshFrom(const PropertyQuery& properties, const Features& reported) {
    if (!reported.meshShader) {
        return {};
    }
    const VkPhysicalDeviceMeshShaderPropertiesEXT& mesh = properties.mesh;
    const bool task = reported.taskShader;
    constexpr std::array<std::uint32_t, kWorkgroupDimensions> none{};
    return {
        .maxMeshWorkgroupSize = std::to_array(mesh.maxMeshWorkGroupSize),
        .maxMeshInvocations = mesh.maxMeshWorkGroupInvocations,
        .maxMeshOutputVertices = mesh.maxMeshOutputVertices,
        .maxMeshOutputPrimitives = mesh.maxMeshOutputPrimitives,
        .maxMeshSharedMemory = mesh.maxMeshSharedMemorySize,
        .maxTaskWorkgroupSize = task ? std::to_array(mesh.maxTaskWorkGroupSize) : none,
        .maxTaskInvocations = task ? mesh.maxTaskWorkGroupInvocations : 0,
        .maxTaskPayloadSize = task ? mesh.maxTaskPayloadSize : 0,
        .maxTaskWorkgroupCount = task ? std::to_array(mesh.maxTaskWorkGroupCount) : none,
        .maxTaskWorkgroupTotalCount = task ? mesh.maxTaskWorkGroupTotalCount : 0,
        .maxTaskSharedMemory = task ? mesh.maxTaskSharedMemorySize : 0,
        .maxTaskPayloadAndSharedMemory = task ? mesh.maxTaskPayloadAndSharedMemorySize : 0,
        .maxMeshWorkgroupCount = std::to_array(mesh.maxMeshWorkGroupCount),
        .maxMeshWorkgroupTotalCount = mesh.maxMeshWorkGroupTotalCount,
        .maxMeshOutputMemory = mesh.maxMeshOutputMemorySize,
        .meshOutputPerVertexGranularity = mesh.meshOutputPerVertexGranularity,
        .meshOutputPerPrimitiveGranularity = mesh.meshOutputPerPrimitiveGranularity,
        .maxMeshPayloadAndSharedMemory = mesh.maxMeshPayloadAndSharedMemorySize,
        .maxMeshPayloadAndOutputMemory = mesh.maxMeshPayloadAndOutputMemorySize,
        .maxMeshOutputComponents = mesh.maxMeshOutputComponents,
        .maxMeshOutputLayers = mesh.maxMeshOutputLayers,
        .maxMeshMultiviewViews = mesh.maxMeshMultiviewViewCount,
    };
}

RayTracingLimits rayTracingFrom(const PropertyQuery& properties, const Features& reported) {
    RayTracingLimits limits;
    if (reported.accelerationStructure) {
        const VkPhysicalDeviceAccelerationStructurePropertiesKHR& structures = properties.accelerationStructure;
        limits.maxGeometries = structures.maxGeometryCount;
        limits.maxInstances = structures.maxInstanceCount;
        limits.maxPrimitives = structures.maxPrimitiveCount;
        limits.scratchAlignment = structures.minAccelerationStructureScratchOffsetAlignment;
        limits.instanceSize = sizeof(VkAccelerationStructureInstanceKHR);
        // Properties resolve with VK_QUERY_RESULT_64_BIT.
        limits.propertyEntryBytes = sizeof(std::uint64_t);
    }
    if (reported.rayTracingPipeline) {
        const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& pipeline = properties.rayTracingPipeline;
        limits.maxRecursionDepth = pipeline.maxRayRecursionDepth;
        limits.identifierSize = pipeline.shaderGroupHandleSize;
        limits.identifierAlignment = pipeline.shaderGroupHandleAlignment;
        limits.tableAlignment = pipeline.shaderGroupBaseAlignment;
        limits.maxShaderGroupStride = pipeline.maxShaderGroupStride;
        limits.maxTraceRays = pipeline.maxRayDispatchInvocationCount;
        limits.maxHitAttributeSize = pipeline.maxRayHitAttributeSize;
    }
    if (reported.rayTracingCaptureReplay) {
        limits.captureReplayIdentifierSize = properties.rayTracingPipeline.shaderGroupHandleCaptureReplaySize;
    }
    if (reported.rayTracingMotion) {
        limits.maxMotionKeyframes = kVulkanMotionKeyframes;
        limits.motionInstanceSize = kVulkanMotionInstanceStride;
    }
    if (reported.partitionedTop) {
        limits.maxPartitions = properties.partitioned.maxPartitionCount;
    }
    if (reported.clusterAccelerationStructure) {
        const VkPhysicalDeviceClusterAccelerationStructurePropertiesNV& clusters = properties.clusters;
        limits.maxClusterVertices = clusters.maxVerticesPerCluster;
        limits.maxClusterTriangles = clusters.maxTrianglesPerCluster;
        limits.clusterScratchAlignment = clusters.clusterScratchByteAlignment;
        limits.clusterAlignment = clusters.clusterByteAlignment;
        limits.clusterTemplateAlignment = clusters.clusterTemplateByteAlignment;
        limits.clusterBottomAlignment = clusters.clusterBottomLevelByteAlignment;
        limits.clusterTemplateBoundsAlignment = clusters.clusterTemplateBoundsByteAlignment;
        limits.maxClusterGeometryIndex = clusters.maxClusterGeometryIndex;
    }
    if (reported.opacityMicromap) {
        limits.maxOpacity2StateSubdivisionLevel = properties.opacityMicromap.maxOpacity2StateSubdivisionLevel;
        limits.maxOpacity4StateSubdivisionLevel = properties.opacityMicromap.maxOpacity4StateSubdivisionLevel;
    }
    if (reported.rayInvocationReorder) {
        const VkPhysicalDeviceRayTracingInvocationReorderPropertiesEXT& reorder = properties.invocationReorder;
        limits.reordersInvocations = reorder.rayTracingInvocationReorderReorderingHint ==
                                     VK_RAY_TRACING_INVOCATION_REORDER_MODE_REORDER_EXT;
        limits.maxBindingTableRecordIndex = reorder.maxShaderBindingTableRecordIndex;
    }
    return limits;
}

// The host clocks vkGetCalibratedTimestampsKHR reads beside the device's,
// none without the extension.
HostClocks hostClocksFrom(const InstanceTable& vk, VkPhysicalDevice adapter,
                          std::span<const VkExtensionProperties> extensions) {
    HostClocks clocks;
    if (calibrationExtension(extensions) == nullptr || vk.vkGetPhysicalDeviceCalibrateableTimeDomainsKHR == nullptr) {
        return clocks;
    }
    const auto domains = listAll<VkTimeDomainKHR>("vkGetPhysicalDeviceCalibrateableTimeDomainsKHR",
                                                  [&vk, adapter](std::uint32_t* count, VkTimeDomainKHR* items) {
                                                      return vk.vkGetPhysicalDeviceCalibrateableTimeDomainsKHR(adapter,
                                                                                                               count,
                                                                                                               items);
                                                  });
    if (!domains) {
        return clocks;
    }
    const auto has = [&domains](VkTimeDomainKHR domain) {
        return std::ranges::contains(*domains, domain);
    };
    // A host clock is only read beside the device's.
    if (has(VK_TIME_DOMAIN_DEVICE_KHR)) {
        clocks = {.monotonic = has(VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR),
                  .monotonicRaw = has(VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_KHR),
                  .performanceCounter = has(VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_KHR)};
    }
    return clocks;
}

// Vulkan writes a timestamp at each stage its family's queues run
// (stagesFor), in rendering and out of it, on a queue whose family has valid
// timestamp bits; it takes no granularity. A VK_QUERY_RESULT_64_BIT copy
// writes each as a uint64_t.
// Whether a queue of a family with flags records vkCmdResetQueryPool, which
// takes a graphics or compute queue (VUID-vkCmdResetQueryPool-commandBuffer-cmdpool).
bool resets(VkQueueFlags flags) {
    return (flags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) != 0;
}

Timestamps timestampsFrom(const InstanceTable& vk, VkPhysicalDevice adapter,
                          std::span<const VkExtensionProperties> extensions, const Queues& queues,
                          const KindFlags& kinds, const Features& features) {
    const TimestampStages graphics = stagesFor(kinds.graphics, features);
    return {.graphics = queues.graphicsTimestampBits != 0,
            .compute = queues.computeTimestampBits != 0,
            .copy = queues.copyTimestampBits != 0,
            .outsideRendering = graphics,
            .insideRendering = graphics,
            .computeStages = stagesFor(kinds.compute, features),
            .copyStages = stagesFor(kinds.copy, features),
            .graphicsResets = resets(kinds.graphics),
            .computeResets = resets(kinds.compute),
            .copyResets = resets(kinds.copy),
            .granularity = false,
            .entryBytes = sizeof(std::uint64_t),
            .hostClocks = hostClocksFrom(vk, adapter, extensions)};
}

// The ResolveMode bits a set of Vulkan resolve modes holds.
std::uint32_t resolveModesFrom(VkResolveModeFlags modes) {
    constexpr std::array kModes{
        std::pair{VK_RESOLVE_MODE_SAMPLE_ZERO_BIT, ResolveMode::SampleZero},
        std::pair{VK_RESOLVE_MODE_AVERAGE_BIT, ResolveMode::Average},
        std::pair{VK_RESOLVE_MODE_MIN_BIT, ResolveMode::Min},
        std::pair{VK_RESOLVE_MODE_MAX_BIT, ResolveMode::Max},
    };
    std::uint32_t bits = 0;
    for (const auto& [vulkan, mode] : kModes) {
        if ((modes & static_cast<VkResolveModeFlags>(vulkan)) != 0) {
            bits |= 1U << std::to_underlying(mode);
        }
    }
    return bits;
}

ResolveCaps resolveFrom(const PropertyQuery& properties) {
    const VkPhysicalDeviceVulkan12Properties& vulkan12 = properties.vulkan12;
    return {.depthModes = resolveModesFrom(vulkan12.supportedDepthResolveModes),
            .stencilModes = resolveModesFrom(vulkan12.supportedStencilResolveModes),
            .independent = on(vulkan12.independentResolve),
            .independentNone = on(vulkan12.independentResolveNone)};
}

Caps readCaps(const InstanceTable& vk, VkPhysicalDevice adapter, std::uint32_t version,
              std::span<const VkExtensionProperties> extensions, bool debugUtils) {
    const DeviceFeatures features(vk, adapter, version, extensions);
    const PropertyQuery properties(vk, adapter, version, extensions);
    Features reported = featuresFrom(features, properties, extensions);
    // VK_EXT_debug_utils labels command buffers and queues, and names objects.
    reported.debugLabels = debugUtils;
    reported.queueDebugLabels = debugUtils;
    reported.objectNames = debugUtils;
    KindFlags kinds;
    Caps caps{
        .info = infoFrom(properties),
        .limits = limitsFrom(properties, features),
        .bindless = bindlessFrom(properties, features),
        .compute = computeFrom(properties),
        .mesh = meshFrom(properties, reported),
        .rayTracing = rayTracingFrom(properties, reported),
        .features = reported,
        .memory = memoryFrom(vk, adapter, properties.core.properties.deviceType),
        .queues = queuesFrom(vk, adapter, version, extensions, kinds, properties.core.properties.limits),
        .timestamps = {},
        .sparse = sparseFrom(features, properties),
        .resolve = resolveFrom(properties),
    };
    caps.timestamps = timestampsFrom(vk, adapter, extensions, caps.queues, kinds, reported);
    if (reported.shadingRateTexture) {
        const VkExtent2D& smallest = properties.shadingRate.minFragmentShadingRateAttachmentTexelSize;
        const VkExtent2D& largest = properties.shadingRate.maxFragmentShadingRateAttachmentTexelSize;
        caps.limits.minShadingRateTile = {smallest.width, smallest.height};
        caps.limits.maxShadingRateTile = {largest.width, largest.height};
    }
    return caps;
}

Formats13 formats13Of(std::uint32_t version, std::span<const VkExtensionProperties> extensions) {
    const bool at13 = hasCore13(version);
    return {.a4r4g4b4 = at13 || lists(extensions, VK_EXT_4444_FORMATS_EXTENSION_NAME),
            .astcHdr = at13 || lists(extensions, VK_EXT_TEXTURE_COMPRESSION_ASTC_HDR_EXTENSION_NAME)};
}

bool hasFormat(Format format, const Formats13& formats) {
    const VkFormat vulkan = vulkanFormat(format);
    if (vulkan == VK_FORMAT_A4R4G4B4_UNORM_PACK16) {
        return formats.a4r4g4b4;
    }
    // The ASTC float formats run in one block (vulkan_core.h:2073-2086).
    const bool astcFloat = vulkan >= VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK && vulkan <= VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK;
    return !astcFloat || formats.astcHdr;
}

core::Result<FormatCaps> readFormatCaps(const InstanceTable& vk, VkPhysicalDevice adapter, std::uint32_t version,
                                        std::span<const VkExtensionProperties> extensions, Format format) {
    if (!hasFormat(format, formats13Of(version, extensions))) {
        return FormatCaps{.format = format};
    }
    const VkFormat vulkan = vulkanFormat(format);
    // The 64-bit flags, where the device has them; the 32-bit ones are the
    // same bits (VkFormatFeatureFlagBits2).
    VkFormatProperties3 properties3{};
    properties3.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3;
    VkFormatProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2;
    const bool flags2 = hasCore13(version) || lists(extensions, VK_KHR_FORMAT_FEATURE_FLAGS_2_EXTENSION_NAME);
    if (flags2) {
        properties.pNext = &properties3;
    }
    vk.vkGetPhysicalDeviceFormatProperties2(adapter, vulkan, &properties);
    const VkFormatFeatureFlags2 texture = flags2 ? properties3.optimalTilingFeatures
                                                 : properties.formatProperties.optimalTilingFeatures;
    const VkFormatFeatureFlags2 buffer = flags2 ? properties3.bufferFeatures
                                                : properties.formatProperties.bufferFeatures;
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
        .sampleCounts = 0,
        // Vulkan has no resolve feature: a resolve writes the target as an
        // attachment (VUID-vkCmdResolveImage-dstImage-02003), and the depth
        // and stencil modes are Caps::resolve's.
        .resolve = has(texture, VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT) ||
                   has(texture, VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT),
        .accelerationStructureVertices = has(buffer, VK_FORMAT_FEATURE_2_ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT_KHR),
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
    if (format == Format::Undefined) {
        return VK_FORMAT_UNDEFINED;
    }
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
