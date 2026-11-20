#include "vulkan/vulkan_features.h"

#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "rhi/feature_request.h"
#include "vulkan/vulkan_caps.h"
#include "vulkan/vulkan_list.h"

#include <volk.h>
#include <vulkan/vulkan_beta.h>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace bfr::rhi::vulkan {

namespace {

using Listed = std::span<const VkExtensionProperties>;

bool at13(const DeviceFeatures& chain) {
    return hasCore13(chain.version);
}

// A feature that is a member of a core version's struct.
template <VkBool32 VkPhysicalDeviceFeatures::* Member>
VulkanForm core10(DeviceFeatures& chain, Listed /*listed*/) {
    return {.members = {&(chain.core.features.*Member)}};
}

template <VkBool32 VkPhysicalDeviceVulkan11Features::* Member>
VulkanForm core11(DeviceFeatures& chain, Listed /*listed*/) {
    return {.members = {&(chain.vulkan11.*Member)}};
}

template <VkBool32 VkPhysicalDeviceVulkan12Features::* Member>
VulkanForm core12(DeviceFeatures& chain, Listed /*listed*/) {
    return {.members = {&(chain.vulkan12.*Member)}};
}

// Below 1.4 BFR enables none of the extensions 1.4 made these of, so there it
// has no form of them.
template <VkBool32 VkPhysicalDeviceVulkan14Features::* Member>
VulkanForm core14(DeviceFeatures& chain, Listed /*listed*/) {
    if (chain.version < VK_API_VERSION_1_4) {
        return {.available = false};
    }
    return {.members = {&(chain.vulkan14.*Member)}};
}

// A feature of an extension's struct.
template <auto Struct, auto Member>
VulkanForm extension(const char* name, DeviceFeatures& chain) {
    return {.extensions = {name}, .members = {&(chain.*Struct.*Member)}};
}

// A member of a portability driver's struct (VK_KHR_portability_subset, which
// the device enables where listed): core everywhere else, with nothing to
// enable. VVL tests the enabled member (sl_image.cpp:371, cc_descriptor.cpp:5475,
// cc_pipeline_graphics.cpp:212).
template <VkBool32 VkPhysicalDevicePortabilitySubsetFeaturesKHR::* Member>
VulkanForm portability(DeviceFeatures& chain, Listed listed) {
    if (!lists(listed, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)) {
        return {};
    }
    return {.members = {&(chain.portability.*Member)}};
}

using Core = VkPhysicalDeviceFeatures;
using Core11 = VkPhysicalDeviceVulkan11Features;
using Core12 = VkPhysicalDeviceVulkan12Features;
using Core13 = VkPhysicalDeviceVulkan13Features;
using Core14 = VkPhysicalDeviceVulkan14Features;
using D = DeviceFeatures;

struct VulkanFeature {
    bool FeatureRequest::* requested = nullptr;
    VulkanFormOf form = nullptr;
};

// One Vulkan form for each requestable feature, in kRequestableFeatures'
// order: the one place Vulkan names them, which the enabled chain, the
// adapter's caps and the device's are all read from.
constexpr std::array kVulkanFeatures{
    VulkanFeature{.requested = &FeatureRequest::swapchain,
                  .form = [](D& /*chain*/, Listed /*listed*/) -> VulkanForm {
                      return {.extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME}};
                  }},
    // The KHR extension, or the EXT one it was promoted from where only that
    // is listed.
    VulkanFeature{.requested = &FeatureRequest::timestampCalibration,
                  .form = [](D& /*chain*/, Listed listed) -> VulkanForm {
                      const char* calibration = calibrationExtension(listed);
                      if (calibration == nullptr) {
                          return {.available = false};
                      }
                      return {.extensions = {calibration}};
                  }},
    VulkanFeature{.requested = &FeatureRequest::robustBufferAccess, .form = core10<&Core::robustBufferAccess>},
    VulkanFeature{
        .requested = &FeatureRequest::robustImageAccess,
        .form = [](D& chain, Listed /*listed*/) -> VulkanForm {
            if (at13(chain)) {
                return {.members = {&chain.vulkan13.robustImageAccess}};
            }
            return extension<&D::imageRobustness, &VkPhysicalDeviceImageRobustnessFeaturesEXT::robustImageAccess>(
                VK_EXT_IMAGE_ROBUSTNESS_EXTENSION_NAME, chain);
        }},
    VulkanFeature{.requested = &FeatureRequest::rayQuery,
                  .form =
                      [](D& chain, Listed /*listed*/) {
                          return extension<&D::rayQuery, &VkPhysicalDeviceRayQueryFeaturesKHR::rayQuery>(
                              VK_KHR_RAY_QUERY_EXTENSION_NAME, chain);
                      }},
    // Vulkan's ray queries run in every stage.
    VulkanFeature{.requested = &FeatureRequest::rayQueryGraphics,
                  .form =
                      [](D& chain, Listed /*listed*/) {
                          return extension<&D::rayQuery, &VkPhysicalDeviceRayQueryFeaturesKHR::rayQuery>(
                              VK_KHR_RAY_QUERY_EXTENSION_NAME, chain);
                      }},
    VulkanFeature{.requested = &FeatureRequest::rayTracingPipeline,
                  .form =
                      [](D& chain, Listed /*listed*/) {
                          return extension<&D::rayTracingPipeline,
                                           &VkPhysicalDeviceRayTracingPipelineFeaturesKHR::rayTracingPipeline>(
                              VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, chain);
                      }},
    // BFR's indirect trace is vkCmdTraceRaysIndirect2KHR.
    VulkanFeature{.requested = &FeatureRequest::traceRaysIndirect,
                  .form =
                      [](D& chain, Listed /*listed*/) {
                          return extension<
                              &D::rayTracingMaintenance1,
                              &VkPhysicalDeviceRayTracingMaintenance1FeaturesKHR::rayTracingPipelineTraceRaysIndirect2>(
                              VK_KHR_RAY_TRACING_MAINTENANCE_1_EXTENSION_NAME, chain);
                      }},
    VulkanFeature{
        .requested = &FeatureRequest::rayPrimitiveCulling,
        .form =
            [](D& chain, Listed /*listed*/) {
                return extension<&D::rayTracingPipeline,
                                 &VkPhysicalDeviceRayTracingPipelineFeaturesKHR::rayTraversalPrimitiveCulling>(
                    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, chain);
            }},
    // VK_KHR_acceleration_structure needs VK_KHR_deferred_host_operations
    // (vkdocs xml/vk.xml:24210).
    VulkanFeature{.requested = &FeatureRequest::accelerationStructure,
                  .form = [](D& chain, Listed /*listed*/) -> VulkanForm {
                      return {.extensions = {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                                             VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME},
                              .members = {&chain.accelerationStructure.accelerationStructure}};
                  }},
    VulkanFeature{.requested = &FeatureRequest::bufferAddress, .form = core12<&Core12::bufferDeviceAddress>},
    VulkanFeature{
        .requested = &FeatureRequest::rayPositionFetch,
        .form =
            [](D& chain, Listed /*listed*/) {
                return extension<&D::rayTracingPositionFetch,
                                 &VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR::rayTracingPositionFetch>(
                    VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME, chain);
            }},
    // Ray tracing maintenance1's query types (queries.adoc:472, 476).
    VulkanFeature{.requested = &FeatureRequest::accelerationStructureCurrentSize,
                  .form =
                      [](D& chain, Listed /*listed*/) {
                          return extension<&D::rayTracingMaintenance1,
                                           &VkPhysicalDeviceRayTracingMaintenance1FeaturesKHR::rayTracingMaintenance1>(
                              VK_KHR_RAY_TRACING_MAINTENANCE_1_EXTENSION_NAME, chain);
                      }},
    VulkanFeature{.requested = &FeatureRequest::serializedBottomCount,
                  .form =
                      [](D& chain, Listed /*listed*/) {
                          return extension<&D::rayTracingMaintenance1,
                                           &VkPhysicalDeviceRayTracingMaintenance1FeaturesKHR::rayTracingMaintenance1>(
                              VK_KHR_RAY_TRACING_MAINTENANCE_1_EXTENSION_NAME, chain);
                      }},
    VulkanFeature{
        .requested = &FeatureRequest::rayInvocationReorder,
        .form =
            [](D& chain, Listed /*listed*/) {
                return extension<&D::invocationReorder,
                                 &VkPhysicalDeviceRayTracingInvocationReorderFeaturesEXT::rayTracingInvocationReorder>(
                    VK_EXT_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME, chain);
            }},
    // VK_KHR_pipeline_library has no features struct.
    VulkanFeature{.requested = &FeatureRequest::rayTracingPipelineLibrary,
                  .form = [](D& /*chain*/, Listed /*listed*/) -> VulkanForm {
                      return {.extensions = {VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME}};
                  }},
    // VK_EXT_pipeline_library_group_handles needs VK_KHR_pipeline_library
    // and VK_KHR_ray_tracing_pipeline (vkdocs xml/vk.xml:29884).
    VulkanFeature{
        .requested = &FeatureRequest::pipelineLibraryIdentifiers,
        .form =
            [](D& chain, Listed /*listed*/) {
                return extension<&D::libraryGroupHandles,
                                 &VkPhysicalDevicePipelineLibraryGroupHandlesFeaturesEXT::pipelineLibraryGroupHandles>(
                    VK_EXT_PIPELINE_LIBRARY_GROUP_HANDLES_EXTENSION_NAME, chain);
            }},
    VulkanFeature{
        .requested = &FeatureRequest::rayTracingCaptureReplay,
        .form =
            [](D& chain, Listed /*listed*/) {
                return extension<
                    &D::rayTracingPipeline,
                    &VkPhysicalDeviceRayTracingPipelineFeaturesKHR::rayTracingPipelineShaderGroupHandleCaptureReplay>(
                    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, chain);
            }},
    VulkanFeature{
        .requested = &FeatureRequest::rayTracingCaptureReplayMixed,
        .form =
            [](D& chain, Listed /*listed*/) {
                return extension<&D::rayTracingPipeline, &VkPhysicalDeviceRayTracingPipelineFeaturesKHR::
                                                             rayTracingPipelineShaderGroupHandleCaptureReplayMixed>(
                    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, chain);
            }},
    VulkanFeature{
        .requested = &FeatureRequest::accelerationStructureIndirectBuild,
        .form =
            [](D& chain, Listed /*listed*/) {
                return extension<&D::accelerationStructure,
                                 &VkPhysicalDeviceAccelerationStructureFeaturesKHR::accelerationStructureIndirectBuild>(
                    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, chain);
            }},
    // VK_EXT_opacity_micromap needs acceleration structures and
    // synchronization2 (vkdocs xml/vk.xml:28172), which BFR always enables.
    VulkanFeature{.requested = &FeatureRequest::opacityMicromap,
                  .form =
                      [](D& chain, Listed /*listed*/) {
                          return extension<&D::opacityMicromap, &VkPhysicalDeviceOpacityMicromapFeaturesEXT::micromap>(
                              VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME, chain);
                      }},
    // VK_NV_ray_tracing_motion_blur needs ray tracing pipelines
    // (vkdocs xml/vk.xml:27270).
    VulkanFeature{
        .requested = &FeatureRequest::rayTracingMotion,
        .form =
            [](D& chain, Listed /*listed*/) {
                return extension<&D::motionBlur, &VkPhysicalDeviceRayTracingMotionBlurFeaturesNV::rayTracingMotionBlur>(
                    VK_NV_RAY_TRACING_MOTION_BLUR_EXTENSION_NAME, chain);
            }},
    VulkanFeature{.requested = &FeatureRequest::rayTracingMotionIndirect,
                  .form =
                      [](D& chain, Listed /*listed*/) {
                          return extension<&D::motionBlur, &VkPhysicalDeviceRayTracingMotionBlurFeaturesNV::
                                                               rayTracingMotionBlurPipelineTraceRaysIndirect>(
                              VK_NV_RAY_TRACING_MOTION_BLUR_EXTENSION_NAME, chain);
                      }},
    // VK_NV_partitioned_acceleration_structure and
    // VK_NV_cluster_acceleration_structure need acceleration structures
    // (vcpkg vk.xml:30757, 30794).
    VulkanFeature{
        .requested = &FeatureRequest::partitionedTop,
        .form =
            [](D& chain, Listed /*listed*/) {
                return extension<
                    &D::partitioned,
                    &VkPhysicalDevicePartitionedAccelerationStructureFeaturesNV::partitionedAccelerationStructure>(
                    VK_NV_PARTITIONED_ACCELERATION_STRUCTURE_EXTENSION_NAME, chain);
            }},
    VulkanFeature{
        .requested = &FeatureRequest::clusterAccelerationStructure,
        .form =
            [](D& chain, Listed /*listed*/) {
                return extension<&D::clusters,
                                 &VkPhysicalDeviceClusterAccelerationStructureFeaturesNV::clusterAccelerationStructure>(
                    VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME, chain);
            }},
    // A replayed structure's buffer and memory are replayed too
    // (VUID-VkAccelerationStructureCreateInfoKHR-deviceAddress-09490,
    // VUID-VkBufferCreateInfo-flags-03338).
    VulkanFeature{.requested = &FeatureRequest::accelerationStructureCaptureReplay,
                  .form = [](D& chain, Listed /*listed*/) -> VulkanForm {
                      return {.extensions = {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME},
                              .members = {&chain.accelerationStructure.accelerationStructureCaptureReplay,
                                          &chain.vulkan12.bufferDeviceAddressCaptureReplay}};
                  }},
    VulkanFeature{.requested = &FeatureRequest::meshShader,
                  .form =
                      [](D& chain, Listed /*listed*/) {
                          return extension<&D::mesh, &VkPhysicalDeviceMeshShaderFeaturesEXT::meshShader>(
                              VK_EXT_MESH_SHADER_EXTENSION_NAME, chain);
                      }},
    VulkanFeature{.requested = &FeatureRequest::taskShader,
                  .form =
                      [](D& chain, Listed /*listed*/) {
                          return extension<&D::mesh, &VkPhysicalDeviceMeshShaderFeaturesEXT::taskShader>(
                              VK_EXT_MESH_SHADER_EXTENSION_NAME, chain);
                      }},
    VulkanFeature{.requested = &FeatureRequest::meshMultiview,
                  .form =
                      [](D& chain, Listed /*listed*/) {
                          return extension<&D::mesh, &VkPhysicalDeviceMeshShaderFeaturesEXT::multiviewMeshShader>(
                              VK_EXT_MESH_SHADER_EXTENSION_NAME, chain);
                      }},
    VulkanFeature{.requested = &FeatureRequest::meshStatistics,
                  .form =
                      [](D& chain, Listed /*listed*/) {
                          return extension<&D::mesh, &VkPhysicalDeviceMeshShaderFeaturesEXT::meshShaderQueries>(
                              VK_EXT_MESH_SHADER_EXTENSION_NAME, chain);
                      }},
    // primitiveFragmentShadingRateMeshShader needs fragment shading rate
    // (VUID-VkPhysicalDeviceMeshShaderFeaturesEXT-primitiveFragmentShadingRateMeshShader-07033),
    // which BFR has no call for.
    VulkanFeature{.requested = &FeatureRequest::meshShadingRate,
                  .form = [](D& /*chain*/, Listed /*listed*/) -> VulkanForm {
                      return {.available = false};
                  }},
    VulkanFeature{.requested = &FeatureRequest::shaderFloat16, .form = core12<&Core12::shaderFloat16>},
    VulkanFeature{.requested = &FeatureRequest::shaderFloat64, .form = core10<&Core::shaderFloat64>},
    VulkanFeature{.requested = &FeatureRequest::shaderInt8, .form = core12<&Core12::shaderInt8>},
    VulkanFeature{.requested = &FeatureRequest::shaderInt16, .form = core10<&Core::shaderInt16>},
    VulkanFeature{.requested = &FeatureRequest::shaderInt64, .form = core10<&Core::shaderInt64>},
    VulkanFeature{.requested = &FeatureRequest::atomicInt64, .form = core12<&Core12::shaderBufferInt64Atomics>},
    VulkanFeature{.requested = &FeatureRequest::atomicInt64Shared, .form = core12<&Core12::shaderSharedInt64Atomics>},
    VulkanFeature{.requested = &FeatureRequest::atomicInt64Texture,
                  .form =
                      [](D& chain, Listed /*listed*/) {
                          return extension<&D::imageAtomicInt64,
                                           &VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT::shaderImageInt64Atomics>(
                              VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME, chain);
                      }},
    VulkanFeature{.requested = &FeatureRequest::sparseAtomicInt64Texture,
                  .form =
                      [](D& chain, Listed /*listed*/) {
                          return extension<&D::imageAtomicInt64,
                                           &VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT::sparseImageInt64Atomics>(
                              VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME, chain);
                      }},
    VulkanFeature{.requested = &FeatureRequest::atomicFloat32,
                  .form =
                      [](D& chain, Listed /*listed*/) {
                          return extension<&D::atomicFloat,
                                           &VkPhysicalDeviceShaderAtomicFloatFeaturesEXT::shaderBufferFloat32Atomics>(
                              VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, chain);
                      }},
    VulkanFeature{.requested = &FeatureRequest::atomicFloat32Add,
                  .form =
                      [](D& chain, Listed /*listed*/) {
                          return extension<&D::atomicFloat,
                                           &VkPhysicalDeviceShaderAtomicFloatFeaturesEXT::shaderBufferFloat32AtomicAdd>(
                              VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, chain);
                      }},
    VulkanFeature{.requested = &FeatureRequest::atomicFloat64,
                  .form =
                      [](D& chain, Listed /*listed*/) {
                          return extension<&D::atomicFloat,
                                           &VkPhysicalDeviceShaderAtomicFloatFeaturesEXT::shaderBufferFloat64Atomics>(
                              VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, chain);
                      }},
    VulkanFeature{.requested = &FeatureRequest::atomicFloat64Add,
                  .form =
                      [](D& chain, Listed /*listed*/) {
                          return extension<&D::atomicFloat,
                                           &VkPhysicalDeviceShaderAtomicFloatFeaturesEXT::shaderBufferFloat64AtomicAdd>(
                              VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, chain);
                      }},
    VulkanFeature{.requested = &FeatureRequest::atomicFloat32Shared,
                  .form =
                      [](D& chain, Listed /*listed*/) {
                          return extension<&D::atomicFloat,
                                           &VkPhysicalDeviceShaderAtomicFloatFeaturesEXT::shaderSharedFloat32Atomics>(
                              VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, chain);
                      }},
    VulkanFeature{.requested = &FeatureRequest::atomicFloat32AddShared,
                  .form =
                      [](D& chain, Listed /*listed*/) {
                          return extension<&D::atomicFloat,
                                           &VkPhysicalDeviceShaderAtomicFloatFeaturesEXT::shaderSharedFloat32AtomicAdd>(
                              VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, chain);
                      }},
    VulkanFeature{.requested = &FeatureRequest::atomicFloat64Shared,
                  .form =
                      [](D& chain, Listed /*listed*/) {
                          return extension<&D::atomicFloat,
                                           &VkPhysicalDeviceShaderAtomicFloatFeaturesEXT::shaderSharedFloat64Atomics>(
                              VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, chain);
                      }},
    VulkanFeature{.requested = &FeatureRequest::atomicFloat64AddShared,
                  .form =
                      [](D& chain, Listed /*listed*/) {
                          return extension<&D::atomicFloat,
                                           &VkPhysicalDeviceShaderAtomicFloatFeaturesEXT::shaderSharedFloat64AtomicAdd>(
                              VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, chain);
                      }},
    VulkanFeature{.requested = &FeatureRequest::atomicFloat32Texture,
                  .form =
                      [](D& chain, Listed /*listed*/) {
                          return extension<&D::atomicFloat,
                                           &VkPhysicalDeviceShaderAtomicFloatFeaturesEXT::shaderImageFloat32Atomics>(
                              VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, chain);
                      }},
    VulkanFeature{.requested = &FeatureRequest::atomicFloat32AddTexture,
                  .form =
                      [](D& chain, Listed /*listed*/) {
                          return extension<&D::atomicFloat,
                                           &VkPhysicalDeviceShaderAtomicFloatFeaturesEXT::shaderImageFloat32AtomicAdd>(
                              VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, chain);
                      }},
    VulkanFeature{.requested = &FeatureRequest::sparseAtomicFloat32Texture,
                  .form =
                      [](D& chain, Listed /*listed*/) {
                          return extension<&D::atomicFloat,
                                           &VkPhysicalDeviceShaderAtomicFloatFeaturesEXT::sparseImageFloat32Atomics>(
                              VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, chain);
                      }},
    VulkanFeature{.requested = &FeatureRequest::sparseAtomicFloat32AddTexture,
                  .form =
                      [](D& chain, Listed /*listed*/) {
                          return extension<&D::atomicFloat,
                                           &VkPhysicalDeviceShaderAtomicFloatFeaturesEXT::sparseImageFloat32AtomicAdd>(
                              VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, chain);
                      }},
    VulkanFeature{
        .requested = &FeatureRequest::barycentrics,
        .form =
            [](D& chain, Listed /*listed*/) {
                return extension<&D::barycentric,
                                 &VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR::fragmentShaderBarycentric>(
                    VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME, chain);
            }},
    VulkanFeature{.requested = &FeatureRequest::geometryShader, .form = core10<&Core::geometryShader>},
    VulkanFeature{.requested = &FeatureRequest::tessellation, .form = core10<&Core::tessellationShader>},
    VulkanFeature{.requested = &FeatureRequest::multiview, .form = core11<&Core11::multiview>},
    VulkanFeature{.requested = &FeatureRequest::multiviewGeometryShader,
                  .form = core11<&Core11::multiviewGeometryShader>},
    VulkanFeature{.requested = &FeatureRequest::multiviewTessellationShader,
                  .form = core11<&Core11::multiviewTessellationShader>},
    VulkanFeature{.requested = &FeatureRequest::sampleRateShading, .form = core10<&Core::sampleRateShading>},
    VulkanFeature{.requested = &FeatureRequest::dualSourceBlend, .form = core10<&Core::dualSrcBlend>},
    VulkanFeature{.requested = &FeatureRequest::independentBlend, .form = core10<&Core::independentBlend>},
    VulkanFeature{.requested = &FeatureRequest::logicOp, .form = core10<&Core::logicOp>},
    VulkanFeature{.requested = &FeatureRequest::depthClamp, .form = core10<&Core::depthClamp>},
    VulkanFeature{.requested = &FeatureRequest::depthBiasClamp, .form = core10<&Core::depthBiasClamp>},
    VulkanFeature{.requested = &FeatureRequest::depthBounds, .form = core10<&Core::depthBounds>},
    VulkanFeature{.requested = &FeatureRequest::wireframe, .form = core10<&Core::fillModeNonSolid>},
    VulkanFeature{.requested = &FeatureRequest::wideLines, .form = core10<&Core::wideLines>},
    VulkanFeature{.requested = &FeatureRequest::samplerAnisotropy, .form = core10<&Core::samplerAnisotropy>},
    VulkanFeature{.requested = &FeatureRequest::samplerMinmax, .form = core12<&Core12::samplerFilterMinmax>},
    VulkanFeature{.requested = &FeatureRequest::samplerMirrorClampToEdge,
                  .form = core12<&Core12::samplerMirrorClampToEdge>},
    // A custom border colour needs both features: BFR's sampler desc gives no
    // format to go with it (VUID-VkSamplerCustomBorderColorCreateInfoEXT-format-04014).
    VulkanFeature{.requested = &FeatureRequest::samplerCustomBorderColor,
                  .form = [](D& chain, Listed /*listed*/) -> VulkanForm {
                      return {.extensions = {VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME},
                              .members = {&chain.customBorderColor.customBorderColors,
                                          &chain.customBorderColor.customBorderColorWithoutFormat}};
                  }},
    // Through 2D views of a 3D image (VUID-VkImageCreateInfo-imageView2DOn3DImage-04459).
    VulkanFeature{.requested = &FeatureRequest::volumeColorTargets,
                  .form = portability<&VkPhysicalDevicePortabilitySubsetFeaturesKHR::imageView2DOn3DImage>},
    // VUID-VkSamplerCreateInfo-samplerMipLodBias-04467.
    VulkanFeature{.requested = &FeatureRequest::samplerLodBias,
                  .form = portability<&VkPhysicalDevicePortabilitySubsetFeaturesKHR::samplerMipLodBias>},
    // Cube views of more than 6 layers (VUID-VkImageViewCreateInfo-viewType-01004).
    VulkanFeature{.requested = &FeatureRequest::imageCubeArray, .form = core10<&Core::imageCubeArray>},
    VulkanFeature{
        .requested = &FeatureRequest::compressionAstcHdr,
        .form = [](D& chain, Listed /*listed*/) -> VulkanForm {
            if (at13(chain)) {
                return {.members = {&chain.vulkan13.textureCompressionASTC_HDR}};
            }
            return extension<&D::astcHdr,
                             &VkPhysicalDeviceTextureCompressionASTCHDRFeaturesEXT::textureCompressionASTC_HDR>(
                VK_EXT_TEXTURE_COMPRESSION_ASTC_HDR_EXTENSION_NAME, chain);
        }},
    // Core from 1.3 with no feature (VkFormat's A4R4G4B4 entry).
    VulkanFeature{.requested = &FeatureRequest::formatBgra4Unorm,
                  .form = [](D& chain, Listed /*listed*/) -> VulkanForm {
                      if (at13(chain)) {
                          return {};
                      }
                      return extension<&D::formats4444, &VkPhysicalDevice4444FormatsFeaturesEXT::formatA4R4G4B4>(
                          VK_EXT_4444_FORMATS_EXTENSION_NAME, chain);
                  }},
    VulkanFeature{.requested = &FeatureRequest::multiDrawIndirect, .form = core10<&Core::multiDrawIndirect>},
    VulkanFeature{.requested = &FeatureRequest::drawIndirectCount, .form = core12<&Core12::drawIndirectCount>},
    VulkanFeature{.requested = &FeatureRequest::drawIndirectFirstInstance,
                  .form = core10<&Core::drawIndirectFirstInstance>},
    VulkanFeature{.requested = &FeatureRequest::fullDrawIndexUint32, .form = core10<&Core::fullDrawIndexUint32>},
    // VUID-VkPipelineInputAssemblyStateCreateInfo-triangleFans-04452.
    VulkanFeature{.requested = &FeatureRequest::triangleFans,
                  .form = portability<&VkPhysicalDevicePortabilitySubsetFeaturesKHR::triangleFans>},
    // VUID-VkPipelineInputAssemblyStateCreateInfo-topology-06252.
    VulkanFeature{.requested = &FeatureRequest::listRestart,
                  .form =
                      [](D& chain, Listed /*listed*/) {
                          return extension<
                              &D::listRestart,
                              &VkPhysicalDevicePrimitiveTopologyListRestartFeaturesEXT::primitiveTopologyListRestart>(
                              VK_EXT_PRIMITIVE_TOPOLOGY_LIST_RESTART_EXTENSION_NAME, chain);
                      }},
    // vkCmdSetPrimitiveTopology: core from 1.3, else the extendedDynamicState
    // feature (VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-03378).
    VulkanFeature{.requested = &FeatureRequest::dynamicTopology,
                  .form = [](D& chain, Listed /*listed*/) -> VulkanForm {
                      if (at13(chain)) {
                          return {};
                      }
                      return extension<&D::extendedDynamicState,
                                       &VkPhysicalDeviceExtendedDynamicStateFeaturesEXT::extendedDynamicState>(
                          VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME, chain);
                  }},
    VulkanFeature{.requested = &FeatureRequest::alphaToOne, .form = core10<&Core::alphaToOne>},
    VulkanFeature{.requested = &FeatureRequest::storageReadWithoutFormat,
                  .form = core10<&Core::shaderStorageImageReadWithoutFormat>},
    VulkanFeature{.requested = &FeatureRequest::storageWriteWithoutFormat,
                  .form = core10<&Core::shaderStorageImageWriteWithoutFormat>},
    VulkanFeature{.requested = &FeatureRequest::storageTextureMultisample,
                  .form = core10<&Core::shaderStorageImageMultisample>},
    VulkanFeature{.requested = &FeatureRequest::textureDynamicIndexing,
                  .form = core10<&Core::shaderSampledImageArrayDynamicIndexing>},
    VulkanFeature{.requested = &FeatureRequest::storageTextureDynamicIndexing,
                  .form = core10<&Core::shaderStorageImageArrayDynamicIndexing>},
    VulkanFeature{.requested = &FeatureRequest::storageBufferDynamicIndexing,
                  .form = core10<&Core::shaderStorageBufferArrayDynamicIndexing>},
    VulkanFeature{.requested = &FeatureRequest::uniformBufferDynamicIndexing,
                  .form = core10<&Core::shaderUniformBufferArrayDynamicIndexing>},
    VulkanFeature{.requested = &FeatureRequest::storageTextureNonUniformIndexing,
                  .form = core12<&Core12::shaderStorageImageArrayNonUniformIndexing>},
    VulkanFeature{.requested = &FeatureRequest::uniformBufferNonUniformIndexing,
                  .form = core12<&Core12::shaderUniformBufferArrayNonUniformIndexing>},
    // VUID-RuntimeSpirv-NonWritable-06341 and -06340.
    VulkanFeature{.requested = &FeatureRequest::vertexStoresAndAtomics,
                  .form = core10<&Core::vertexPipelineStoresAndAtomics>},
    VulkanFeature{.requested = &FeatureRequest::fragmentStoresAndAtomics,
                  .form = core10<&Core::fragmentStoresAndAtomics>},
    VulkanFeature{.requested = &FeatureRequest::tessellationGeometryPointSize,
                  .form = core10<&Core::shaderTessellationAndGeometryPointSize>},
    VulkanFeature{.requested = &FeatureRequest::textureGatherExtended,
                  .form = core10<&Core::shaderImageGatherExtended>},
    VulkanFeature{.requested = &FeatureRequest::clipDistance, .form = core10<&Core::shaderClipDistance>},
    VulkanFeature{.requested = &FeatureRequest::cullDistance, .form = core10<&Core::shaderCullDistance>},
    VulkanFeature{.requested = &FeatureRequest::shaderMinLod, .form = core10<&Core::shaderResourceMinLod>},
    VulkanFeature{.requested = &FeatureRequest::storageBuffer16bitAccess,
                  .form = core11<&Core11::storageBuffer16BitAccess>},
    VulkanFeature{.requested = &FeatureRequest::uniformAndStorageBuffer16bitAccess,
                  .form = core11<&Core11::uniformAndStorageBuffer16BitAccess>},
    VulkanFeature{.requested = &FeatureRequest::pushConstant16bit, .form = core11<&Core11::storagePushConstant16>},
    VulkanFeature{.requested = &FeatureRequest::inputOutput16bit, .form = core11<&Core11::storageInputOutput16>},
    VulkanFeature{.requested = &FeatureRequest::storageBuffer8bitAccess,
                  .form = core12<&Core12::storageBuffer8BitAccess>},
    VulkanFeature{.requested = &FeatureRequest::uniformAndStorageBuffer8bitAccess,
                  .form = core12<&Core12::uniformAndStorageBuffer8BitAccess>},
    VulkanFeature{.requested = &FeatureRequest::pushConstant8bit, .form = core12<&Core12::storagePushConstant8>},
    VulkanFeature{.requested = &FeatureRequest::variablePointersStorageBuffer,
                  .form = core11<&Core11::variablePointersStorageBuffer>},
    VulkanFeature{.requested = &FeatureRequest::variablePointers, .form = core11<&Core11::variablePointers>},
    VulkanFeature{.requested = &FeatureRequest::drawParameters, .form = core11<&Core11::shaderDrawParameters>},
    VulkanFeature{.requested = &FeatureRequest::scalarBlockLayout, .form = core12<&Core12::scalarBlockLayout>},
    VulkanFeature{.requested = &FeatureRequest::uniformBufferStandardLayout,
                  .form = core12<&Core12::uniformBufferStandardLayout>},
    VulkanFeature{.requested = &FeatureRequest::subgroupExtendedTypes,
                  .form = core12<&Core12::shaderSubgroupExtendedTypes>},
    VulkanFeature{.requested = &FeatureRequest::memoryModel, .form = core12<&Core12::vulkanMemoryModel>},
    VulkanFeature{.requested = &FeatureRequest::memoryModelDeviceScope,
                  .form = core12<&Core12::vulkanMemoryModelDeviceScope>},
    VulkanFeature{.requested = &FeatureRequest::memoryModelVisibilityChains,
                  .form = core12<&Core12::vulkanMemoryModelAvailabilityVisibilityChains>},
    VulkanFeature{.requested = &FeatureRequest::shaderOutputLayer, .form = core12<&Core12::shaderOutputLayer>},
    VulkanFeature{.requested = &FeatureRequest::subgroupBroadcastDynamicId,
                  .form = core12<&Core12::subgroupBroadcastDynamicId>},
    VulkanFeature{.requested = &FeatureRequest::demoteToHelperInvocation,
                  .form = [](D& chain, Listed /*listed*/) -> VulkanForm {
                      if (at13(chain)) {
                          return {.members = {&chain.vulkan13.shaderDemoteToHelperInvocation}};
                      }
                      return extension<
                          &D::demoteToHelperInvocation,
                          &VkPhysicalDeviceShaderDemoteToHelperInvocationFeaturesEXT::shaderDemoteToHelperInvocation>(
                          VK_EXT_SHADER_DEMOTE_TO_HELPER_INVOCATION_EXTENSION_NAME, chain);
                  }},
    VulkanFeature{.requested = &FeatureRequest::terminateInvocation,
                  .form = [](D& chain, Listed /*listed*/) -> VulkanForm {
                      if (at13(chain)) {
                          return {.members = {&chain.vulkan13.shaderTerminateInvocation}};
                      }
                      return extension<
                          &D::terminateInvocation,
                          &VkPhysicalDeviceShaderTerminateInvocationFeaturesKHR::shaderTerminateInvocation>(
                          VK_KHR_SHADER_TERMINATE_INVOCATION_EXTENSION_NAME, chain);
                  }},
    VulkanFeature{
        .requested = &FeatureRequest::zeroInitializeWorkgroupMemory,
        .form = [](D& chain, Listed /*listed*/) -> VulkanForm {
            if (at13(chain)) {
                return {.members = {&chain.vulkan13.shaderZeroInitializeWorkgroupMemory}};
            }
            return extension<
                &D::zeroInitializeWorkgroupMemory,
                &VkPhysicalDeviceZeroInitializeWorkgroupMemoryFeaturesKHR::shaderZeroInitializeWorkgroupMemory>(
                VK_KHR_ZERO_INITIALIZE_WORKGROUP_MEMORY_EXTENSION_NAME, chain);
        }},
    VulkanFeature{.requested = &FeatureRequest::integerDotProduct,
                  .form = [](D& chain, Listed /*listed*/) -> VulkanForm {
                      if (at13(chain)) {
                          return {.members = {&chain.vulkan13.shaderIntegerDotProduct}};
                      }
                      return extension<&D::integerDotProduct,
                                       &VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR::shaderIntegerDotProduct>(
                          VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME, chain);
                  }},
    // maintenance4: VUID-RuntimeSpirv-LocalSizeId-06434 and
    // VUID-RuntimeSpirv-maintenance4-06817.
    VulkanFeature{.requested = &FeatureRequest::localSizeId,
                  .form = [](D& chain, Listed /*listed*/) -> VulkanForm {
                      if (at13(chain)) {
                          return {.members = {&chain.vulkan13.maintenance4}};
                      }
                      return extension<&D::maintenance4, &VkPhysicalDeviceMaintenance4FeaturesKHR::maintenance4>(
                          VK_KHR_MAINTENANCE_4_EXTENSION_NAME, chain);
                  }},
    VulkanFeature{.requested = &FeatureRequest::subgroupRotate, .form = core14<&Core14::shaderSubgroupRotate>},
    VulkanFeature{.requested = &FeatureRequest::subgroupRotateClustered,
                  .form = core14<&Core14::shaderSubgroupRotateClustered>},
    VulkanFeature{.requested = &FeatureRequest::floatControls2, .form = core14<&Core14::shaderFloatControls2>},
    VulkanFeature{.requested = &FeatureRequest::expectAssume, .form = core14<&Core14::shaderExpectAssume>},
    // Core from 1.3; an extension with no features struct below it.
    VulkanFeature{.requested = &FeatureRequest::nonSemanticInfo,
                  .form = [](D& chain, Listed /*listed*/) -> VulkanForm {
                      if (at13(chain)) {
                          return {};
                      }
                      return {.extensions = {VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME}};
                  }},
    VulkanFeature{.requested = &FeatureRequest::pipelineStatistics, .form = core10<&Core::pipelineStatisticsQuery>},
    VulkanFeature{.requested = &FeatureRequest::occlusionQueryPrecise, .form = core10<&Core::occlusionQueryPrecise>},
    VulkanFeature{.requested = &FeatureRequest::sparseBinding, .form = core10<&Core::sparseBinding>},
    VulkanFeature{.requested = &FeatureRequest::sparseBuffers, .form = core10<&Core::sparseResidencyBuffer>},
    VulkanFeature{.requested = &FeatureRequest::sparseTextures2d, .form = core10<&Core::sparseResidencyImage2D>},
    VulkanFeature{.requested = &FeatureRequest::sparseTextures3d, .form = core10<&Core::sparseResidencyImage3D>},
    VulkanFeature{.requested = &FeatureRequest::sparseSamples2, .form = core10<&Core::sparseResidency2Samples>},
    VulkanFeature{.requested = &FeatureRequest::sparseSamples4, .form = core10<&Core::sparseResidency4Samples>},
    VulkanFeature{.requested = &FeatureRequest::sparseSamples8, .form = core10<&Core::sparseResidency8Samples>},
    VulkanFeature{.requested = &FeatureRequest::sparseSamples16, .form = core10<&Core::sparseResidency16Samples>},
    VulkanFeature{.requested = &FeatureRequest::sparseAliased, .form = core10<&Core::sparseResidencyAliased>},
    VulkanFeature{.requested = &FeatureRequest::sparseShaderResidency, .form = core10<&Core::shaderResourceResidency>},
};

constexpr bool inRequestableOrder() {
    for (std::size_t index = 0; index < kVulkanFeatures.size(); ++index) {
        if (kVulkanFeatures.at(index).requested != kRequestableFeatures.at(index).requested) {
            return false;
        }
    }
    return true;
}
static_assert(kVulkanFeatures.size() == kRequestableFeatures.size() && inRequestableOrder(),
              "one Vulkan form for each requestable feature, in its order");

// Whether form is had with extensions.
bool has(const VulkanForm& form, std::span<const VkExtensionProperties> extensions) {
    return form.available &&
           std::ranges::all_of(form.extensions,
                               [extensions](const char* name) { return name == nullptr || lists(extensions, name); }) &&
           std::ranges::all_of(form.members,
                               [](const VkBool32* member) { return member == nullptr || *member == VK_TRUE; });
}

// The extensions a device enables for request: those of BFR's own needs and
// the spec's, those of queue levels and of devices with no queues where
// listed, and each requested feature's, once each.
std::vector<const char*> namesFor(std::uint32_t version, std::span<const VkExtensionProperties> listed,
                                  const FeatureRequest& request) {
    // Only for the forms' extensions, which name no member of it.
    DeviceFeatures forms(version, {});
    std::vector<const char*> names;
    const auto add = [&names](const char* name) {
        if (name != nullptr &&
            std::ranges::none_of(names, [name](const char* added) { return std::string_view{added} == name; })) {
            names.push_back(name);
        }
    };
    if (!hasCore13(version)) {
        add(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
        add(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
    }
    for (const char* needed : {VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME, VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME,
                               VK_KHR_MAINTENANCE_9_EXTENSION_NAME}) {
        if (lists(listed, needed)) {
            add(needed);
        }
    }
    for (const VulkanFeature& feature : kVulkanFeatures) {
        if (request.*feature.requested) {
            std::ranges::for_each(feature.form(forms, listed).extensions, add);
        }
    }
    return names;
}

// names as an extension listing.
std::vector<VkExtensionProperties> listingOf(std::span<const char* const> names) {
    std::vector<VkExtensionProperties> listing(names.size());
    for (std::size_t index = 0; index < names.size(); ++index) {
        const std::string_view name{names[index]};
        std::ranges::copy(name.substr(0, VK_MAX_EXTENSION_NAME_SIZE - 1), listing[index].extensionName);
    }
    return listing;
}

} // namespace

std::array<bool, kRequestableFeatures.size()> requestableIn(DeviceFeatures& chain,
                                                            std::span<const VkExtensionProperties> listed,
                                                            std::span<const VkExtensionProperties> extensions) {
    std::array<bool, kRequestableFeatures.size()> had{};
    for (std::size_t index = 0; index < kVulkanFeatures.size(); ++index) {
        had.at(index) = has(kVulkanFeatures.at(index).form(chain, listed), extensions);
    }
    // A feature without a prerequisite is not had; each pass settles one more
    // level of prerequisites, until none changes.
    bool changed = true;
    while (changed) {
        changed = false;
        for (std::size_t index = 0; index < kRequestableFeatures.size(); ++index) {
            for (const auto prerequisite : kRequestableFeatures.at(index).prerequisites) {
                if (had.at(index) && prerequisite != nullptr && !had.at(requestableIndex(prerequisite))) {
                    had.at(index) = false;
                    changed = true;
                }
            }
        }
    }
    return had;
}

EnabledFeatures::EnabledFeatures(const DeviceFeatures& supported, std::span<const VkExtensionProperties> listed,
                                 const FeatureRequest& request, const DeviceDesc& desc)
    : names(namesFor(supported.version, listed, request)),
      extensions(listingOf(names)),
      chain(supported.version, extensions) {
    enableRequirements(chain);
    // Queue levels and devices with no queues, and a portability driver's
    // limits on what the core calls BFR makes do, as the adapter has them;
    // the requestable portability members follow the request below.
    chain.vulkan14.globalPriorityQuery = supported.vulkan14.globalPriorityQuery;
    chain.globalPriority.globalPriorityQuery = supported.globalPriority.globalPriorityQuery;
    chain.maintenance9.maintenance9 = supported.maintenance9.maintenance9;
    void* const portabilityNext = chain.portability.pNext;
    chain.portability = supported.portability;
    chain.portability.pNext = portabilityNext;
    chain.portability.imageView2DOn3DImage = VK_FALSE;
    chain.portability.samplerMipLodBias = VK_FALSE;
    chain.portability.triangleFans = VK_FALSE;
    // The tables update after bind, as the others do, which needs their
    // features
    // (VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-descriptorBindingUniformBufferUpdateAfterBind-03005,
    // -descriptorBindingAccelerationStructureUpdateAfterBind-03570).
    chain.vulkan12.descriptorBindingUniformBufferUpdateAfterBind =
        desc.maxBindlessUniformBuffers != 0 ? supported.vulkan12.descriptorBindingUniformBufferUpdateAfterBind
                                            : VK_FALSE;
    chain.accelerationStructure.descriptorBindingAccelerationStructureUpdateAfterBind =
        desc.maxBindlessAccelerationStructures != 0 && request.accelerationStructure
            ? supported.accelerationStructure.descriptorBindingAccelerationStructureUpdateAfterBind
            : VK_FALSE;
    for (const VulkanFeature& feature : kVulkanFeatures) {
        if (!(request.*feature.requested)) {
            continue;
        }
        for (VkBool32* member : feature.form(chain, listed).members) {
            if (member != nullptr) {
                *member = VK_TRUE;
            }
        }
    }
}

} // namespace bfr::rhi::vulkan
