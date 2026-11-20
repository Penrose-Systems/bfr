#pragma once

#include "bfr/core/error.h"
#include "bfr/core/text.h"
#include "bfr/rhi/caps.h"

#include <array>
#include <cstddef>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace bfr::rhi {

/// A feature a device may be created with: its name in bfr.yaml, its
/// FeatureRequest member, and where an adapter's caps offer it: its Features
/// member, else its Sparse member, else, with neither, any host clock
/// calibrateTimestamps reads. A request that names it must name its
/// prerequisites too; noCall marks one BFR has no call for on any backend.
struct RequestableFeature {
    std::string_view name;
    bool FeatureRequest::* requested = nullptr;
    bool Features::* feature = nullptr;
    bool Sparse::* sparse = nullptr;
    std::array<bool FeatureRequest::*, 2> prerequisites{};
    bool noCall = false;
};

// The prerequisites are Vulkan's: those its VUIDs tie one feature to
// (VUID-VkPhysicalDeviceVariablePointersFeatures-variablePointers-01431,
// VUID-VkPhysicalDeviceMultiviewFeatures-multiviewGeometryShader-00580 and
// -multiviewTessellationShader-00581,
// VUID-VkPhysicalDeviceMeshShaderFeaturesEXT-multiviewMeshShader-07032,
// VUID-VkPhysicalDeviceRayTracingPipelineFeaturesKHR-rayTracingPipelineShaderGroupHandleCaptureReplayMixed-03575,
// VUID-VkDeviceCreateInfo-None-04896, -04897 and -04898; a resident or
// aliased resource is a sparse-bound one, VUID-VkBufferCreateInfo-flags-00918
// and -00915, VUID-VkImageCreateInfo-flags-00987 and -00969), those its
// extensions depend on (vk.xml: ray queries, pipelines, micromaps and
// partitioned and cluster structures on acceleration structures; motion,
// reordering and libraries on ray tracing pipelines), and those BFR's caps
// already composed: builds read their inputs by device address
// (VUID-vkGetBufferDeviceAddress-bufferDeviceAddress-03324), indirect traces
// and replayed identifiers are of ray tracing pipelines, and
// ray_query_graphics is ray_query's Vulkan feature. A request without them is
// refused on every backend, so it means the same on each.
inline constexpr std::array kRequestableFeatures{
    RequestableFeature{.name = "swapchain", .requested = &FeatureRequest::swapchain, .feature = &Features::swapchain},
    RequestableFeature{.name = "timestamp_calibration", .requested = &FeatureRequest::timestampCalibration},
    RequestableFeature{.name = "robust_buffer_access",
                       .requested = &FeatureRequest::robustBufferAccess,
                       .feature = &Features::robustBufferAccess},
    RequestableFeature{.name = "robust_image_access",
                       .requested = &FeatureRequest::robustImageAccess,
                       .feature = &Features::robustImageAccess},
    RequestableFeature{.name = "ray_query",
                       .requested = &FeatureRequest::rayQuery,
                       .feature = &Features::rayQuery,
                       .prerequisites = {&FeatureRequest::accelerationStructure}},
    RequestableFeature{.name = "ray_query_graphics",
                       .requested = &FeatureRequest::rayQueryGraphics,
                       .feature = &Features::rayQueryGraphics,
                       .prerequisites = {&FeatureRequest::rayQuery}},
    RequestableFeature{.name = "ray_tracing_pipeline",
                       .requested = &FeatureRequest::rayTracingPipeline,
                       .feature = &Features::rayTracingPipeline,
                       .prerequisites = {&FeatureRequest::accelerationStructure}},
    RequestableFeature{.name = "trace_rays_indirect",
                       .requested = &FeatureRequest::traceRaysIndirect,
                       .feature = &Features::traceRaysIndirect,
                       .prerequisites = {&FeatureRequest::rayTracingPipeline}},
    RequestableFeature{.name = "ray_primitive_culling",
                       .requested = &FeatureRequest::rayPrimitiveCulling,
                       .feature = &Features::rayPrimitiveCulling,
                       .prerequisites = {&FeatureRequest::accelerationStructure}},
    RequestableFeature{.name = "acceleration_structure",
                       .requested = &FeatureRequest::accelerationStructure,
                       .feature = &Features::accelerationStructure,
                       .prerequisites = {&FeatureRequest::bufferAddress}},
    RequestableFeature{.name = "buffer_address",
                       .requested = &FeatureRequest::bufferAddress,
                       .feature = &Features::bufferAddress},
    RequestableFeature{.name = "ray_position_fetch",
                       .requested = &FeatureRequest::rayPositionFetch,
                       .feature = &Features::rayPositionFetch,
                       .prerequisites = {&FeatureRequest::accelerationStructure}},
    RequestableFeature{.name = "acceleration_structure_current_size",
                       .requested = &FeatureRequest::accelerationStructureCurrentSize,
                       .feature = &Features::accelerationStructureCurrentSize,
                       .prerequisites = {&FeatureRequest::accelerationStructure}},
    RequestableFeature{.name = "serialized_bottom_count",
                       .requested = &FeatureRequest::serializedBottomCount,
                       .feature = &Features::serializedBottomCount,
                       .prerequisites = {&FeatureRequest::accelerationStructure}},
    RequestableFeature{.name = "ray_invocation_reorder",
                       .requested = &FeatureRequest::rayInvocationReorder,
                       .feature = &Features::rayInvocationReorder,
                       .prerequisites = {&FeatureRequest::rayTracingPipeline}},
    RequestableFeature{.name = "ray_tracing_pipeline_library",
                       .requested = &FeatureRequest::rayTracingPipelineLibrary,
                       .feature = &Features::rayTracingPipelineLibrary,
                       .prerequisites = {&FeatureRequest::rayTracingPipeline}},
    RequestableFeature{.name = "pipeline_library_identifiers",
                       .requested = &FeatureRequest::pipelineLibraryIdentifiers,
                       .feature = &Features::pipelineLibraryIdentifiers,
                       .prerequisites = {&FeatureRequest::rayTracingPipelineLibrary}},
    RequestableFeature{.name = "ray_tracing_capture_replay",
                       .requested = &FeatureRequest::rayTracingCaptureReplay,
                       .feature = &Features::rayTracingCaptureReplay,
                       .prerequisites = {&FeatureRequest::rayTracingPipeline}},
    RequestableFeature{.name = "ray_tracing_capture_replay_mixed",
                       .requested = &FeatureRequest::rayTracingCaptureReplayMixed,
                       .feature = &Features::rayTracingCaptureReplayMixed,
                       .prerequisites = {&FeatureRequest::rayTracingCaptureReplay}},
    RequestableFeature{.name = "acceleration_structure_indirect_build",
                       .requested = &FeatureRequest::accelerationStructureIndirectBuild,
                       .feature = &Features::accelerationStructureIndirectBuild,
                       .prerequisites = {&FeatureRequest::accelerationStructure}},
    RequestableFeature{.name = "opacity_micromap",
                       .requested = &FeatureRequest::opacityMicromap,
                       .feature = &Features::opacityMicromap,
                       .prerequisites = {&FeatureRequest::accelerationStructure}},
    RequestableFeature{.name = "ray_tracing_motion",
                       .requested = &FeatureRequest::rayTracingMotion,
                       .feature = &Features::rayTracingMotion,
                       .prerequisites = {&FeatureRequest::rayTracingPipeline}},
    RequestableFeature{.name = "ray_tracing_motion_indirect",
                       .requested = &FeatureRequest::rayTracingMotionIndirect,
                       .feature = &Features::rayTracingMotionIndirect,
                       .prerequisites = {&FeatureRequest::rayTracingMotion}},
    RequestableFeature{.name = "partitioned_top",
                       .requested = &FeatureRequest::partitionedTop,
                       .feature = &Features::partitionedTop,
                       .prerequisites = {&FeatureRequest::accelerationStructure}},
    RequestableFeature{.name = "cluster_acceleration_structure",
                       .requested = &FeatureRequest::clusterAccelerationStructure,
                       .feature = &Features::clusterAccelerationStructure,
                       .prerequisites = {&FeatureRequest::accelerationStructure}},
    RequestableFeature{.name = "acceleration_structure_capture_replay",
                       .requested = &FeatureRequest::accelerationStructureCaptureReplay,
                       .feature = &Features::accelerationStructureCaptureReplay,
                       .prerequisites = {&FeatureRequest::accelerationStructure}},
    RequestableFeature{.name = "mesh_shader",
                       .requested = &FeatureRequest::meshShader,
                       .feature = &Features::meshShader},
    RequestableFeature{.name = "task_shader",
                       .requested = &FeatureRequest::taskShader,
                       .feature = &Features::taskShader},
    RequestableFeature{.name = "mesh_multiview",
                       .requested = &FeatureRequest::meshMultiview,
                       .feature = &Features::meshMultiview,
                       .prerequisites = {&FeatureRequest::multiview}},
    RequestableFeature{.name = "mesh_statistics",
                       .requested = &FeatureRequest::meshStatistics,
                       .feature = &Features::meshStatistics},
    RequestableFeature{.name = "mesh_shading_rate",
                       .requested = &FeatureRequest::meshShadingRate,
                       .feature = &Features::meshShadingRate,
                       .noCall = true},
    RequestableFeature{.name = "shader_float16",
                       .requested = &FeatureRequest::shaderFloat16,
                       .feature = &Features::shaderFloat16},
    RequestableFeature{.name = "shader_float64",
                       .requested = &FeatureRequest::shaderFloat64,
                       .feature = &Features::shaderFloat64},
    RequestableFeature{.name = "shader_int8",
                       .requested = &FeatureRequest::shaderInt8,
                       .feature = &Features::shaderInt8},
    RequestableFeature{.name = "shader_int16",
                       .requested = &FeatureRequest::shaderInt16,
                       .feature = &Features::shaderInt16},
    RequestableFeature{.name = "shader_int64",
                       .requested = &FeatureRequest::shaderInt64,
                       .feature = &Features::shaderInt64},
    RequestableFeature{.name = "atomic_int64",
                       .requested = &FeatureRequest::atomicInt64,
                       .feature = &Features::atomicInt64},
    RequestableFeature{.name = "atomic_int64_shared",
                       .requested = &FeatureRequest::atomicInt64Shared,
                       .feature = &Features::atomicInt64Shared},
    RequestableFeature{.name = "atomic_int64_texture",
                       .requested = &FeatureRequest::atomicInt64Texture,
                       .feature = &Features::atomicInt64Texture},
    RequestableFeature{.name = "sparse_atomic_int64_texture",
                       .requested = &FeatureRequest::sparseAtomicInt64Texture,
                       .feature = &Features::sparseAtomicInt64Texture,
                       .prerequisites = {&FeatureRequest::atomicInt64Texture}},
    RequestableFeature{.name = "atomic_float32",
                       .requested = &FeatureRequest::atomicFloat32,
                       .feature = &Features::atomicFloat32},
    RequestableFeature{.name = "atomic_float32_add",
                       .requested = &FeatureRequest::atomicFloat32Add,
                       .feature = &Features::atomicFloat32Add},
    RequestableFeature{.name = "atomic_float64",
                       .requested = &FeatureRequest::atomicFloat64,
                       .feature = &Features::atomicFloat64},
    RequestableFeature{.name = "atomic_float64_add",
                       .requested = &FeatureRequest::atomicFloat64Add,
                       .feature = &Features::atomicFloat64Add},
    RequestableFeature{.name = "atomic_float32_shared",
                       .requested = &FeatureRequest::atomicFloat32Shared,
                       .feature = &Features::atomicFloat32Shared},
    RequestableFeature{.name = "atomic_float32_add_shared",
                       .requested = &FeatureRequest::atomicFloat32AddShared,
                       .feature = &Features::atomicFloat32AddShared},
    RequestableFeature{.name = "atomic_float64_shared",
                       .requested = &FeatureRequest::atomicFloat64Shared,
                       .feature = &Features::atomicFloat64Shared},
    RequestableFeature{.name = "atomic_float64_add_shared",
                       .requested = &FeatureRequest::atomicFloat64AddShared,
                       .feature = &Features::atomicFloat64AddShared},
    RequestableFeature{.name = "atomic_float32_texture",
                       .requested = &FeatureRequest::atomicFloat32Texture,
                       .feature = &Features::atomicFloat32Texture},
    RequestableFeature{.name = "atomic_float32_add_texture",
                       .requested = &FeatureRequest::atomicFloat32AddTexture,
                       .feature = &Features::atomicFloat32AddTexture},
    RequestableFeature{.name = "sparse_atomic_float32_texture",
                       .requested = &FeatureRequest::sparseAtomicFloat32Texture,
                       .feature = &Features::sparseAtomicFloat32Texture,
                       .prerequisites = {&FeatureRequest::atomicFloat32Texture}},
    RequestableFeature{.name = "sparse_atomic_float32_add_texture",
                       .requested = &FeatureRequest::sparseAtomicFloat32AddTexture,
                       .feature = &Features::sparseAtomicFloat32AddTexture,
                       .prerequisites = {&FeatureRequest::atomicFloat32AddTexture}},
    RequestableFeature{.name = "barycentrics",
                       .requested = &FeatureRequest::barycentrics,
                       .feature = &Features::barycentrics},
    RequestableFeature{.name = "geometry_shader",
                       .requested = &FeatureRequest::geometryShader,
                       .feature = &Features::geometryShader},
    RequestableFeature{.name = "tessellation",
                       .requested = &FeatureRequest::tessellation,
                       .feature = &Features::tessellation},
    RequestableFeature{.name = "multiview", .requested = &FeatureRequest::multiview, .feature = &Features::multiview},
    RequestableFeature{.name = "multiview_geometry_shader",
                       .requested = &FeatureRequest::multiviewGeometryShader,
                       .feature = &Features::multiviewGeometryShader,
                       .prerequisites = {&FeatureRequest::multiview}},
    RequestableFeature{.name = "multiview_tessellation_shader",
                       .requested = &FeatureRequest::multiviewTessellationShader,
                       .feature = &Features::multiviewTessellationShader,
                       .prerequisites = {&FeatureRequest::multiview}},
    RequestableFeature{.name = "sample_rate_shading",
                       .requested = &FeatureRequest::sampleRateShading,
                       .feature = &Features::sampleRateShading},
    RequestableFeature{.name = "dual_source_blend",
                       .requested = &FeatureRequest::dualSourceBlend,
                       .feature = &Features::dualSourceBlend},
    RequestableFeature{.name = "independent_blend",
                       .requested = &FeatureRequest::independentBlend,
                       .feature = &Features::independentBlend},
    RequestableFeature{.name = "logic_op", .requested = &FeatureRequest::logicOp, .feature = &Features::logicOp},
    RequestableFeature{.name = "depth_clamp",
                       .requested = &FeatureRequest::depthClamp,
                       .feature = &Features::depthClamp},
    RequestableFeature{.name = "depth_bias_clamp",
                       .requested = &FeatureRequest::depthBiasClamp,
                       .feature = &Features::depthBiasClamp},
    RequestableFeature{.name = "depth_bounds",
                       .requested = &FeatureRequest::depthBounds,
                       .feature = &Features::depthBounds},
    RequestableFeature{.name = "wireframe", .requested = &FeatureRequest::wireframe, .feature = &Features::wireframe},
    RequestableFeature{.name = "wide_lines", .requested = &FeatureRequest::wideLines, .feature = &Features::wideLines},
    RequestableFeature{.name = "sampler_anisotropy",
                       .requested = &FeatureRequest::samplerAnisotropy,
                       .feature = &Features::samplerAnisotropy},
    RequestableFeature{.name = "sampler_minmax",
                       .requested = &FeatureRequest::samplerMinmax,
                       .feature = &Features::samplerMinmax},
    RequestableFeature{.name = "sampler_mirror_clamp_to_edge",
                       .requested = &FeatureRequest::samplerMirrorClampToEdge,
                       .feature = &Features::samplerMirrorClampToEdge},
    RequestableFeature{.name = "sampler_custom_border_color",
                       .requested = &FeatureRequest::samplerCustomBorderColor,
                       .feature = &Features::samplerCustomBorderColor},
    RequestableFeature{.name = "volume_color_targets",
                       .requested = &FeatureRequest::volumeColorTargets,
                       .feature = &Features::volumeColorTargets},
    RequestableFeature{.name = "sampler_lod_bias",
                       .requested = &FeatureRequest::samplerLodBias,
                       .feature = &Features::samplerLodBias},
    RequestableFeature{.name = "image_cube_array",
                       .requested = &FeatureRequest::imageCubeArray,
                       .feature = &Features::imageCubeArray},
    RequestableFeature{.name = "compression_astc_hdr",
                       .requested = &FeatureRequest::compressionAstcHdr,
                       .feature = &Features::compressionAstcHdr},
    RequestableFeature{.name = "format_bgra4_unorm",
                       .requested = &FeatureRequest::formatBgra4Unorm,
                       .feature = &Features::formatBgra4Unorm},
    RequestableFeature{.name = "multi_draw_indirect",
                       .requested = &FeatureRequest::multiDrawIndirect,
                       .feature = &Features::multiDrawIndirect},
    RequestableFeature{.name = "draw_indirect_count",
                       .requested = &FeatureRequest::drawIndirectCount,
                       .feature = &Features::drawIndirectCount},
    RequestableFeature{.name = "draw_indirect_first_instance",
                       .requested = &FeatureRequest::drawIndirectFirstInstance,
                       .feature = &Features::drawIndirectFirstInstance},
    RequestableFeature{.name = "full_draw_index_uint32",
                       .requested = &FeatureRequest::fullDrawIndexUint32,
                       .feature = &Features::fullDrawIndexUint32},
    RequestableFeature{.name = "triangle_fans",
                       .requested = &FeatureRequest::triangleFans,
                       .feature = &Features::triangleFans},
    RequestableFeature{.name = "list_restart",
                       .requested = &FeatureRequest::listRestart,
                       .feature = &Features::listRestart},
    RequestableFeature{.name = "dynamic_topology",
                       .requested = &FeatureRequest::dynamicTopology,
                       .feature = &Features::dynamicTopology},
    RequestableFeature{.name = "alpha_to_one",
                       .requested = &FeatureRequest::alphaToOne,
                       .feature = &Features::alphaToOne},
    RequestableFeature{.name = "storage_read_without_format",
                       .requested = &FeatureRequest::storageReadWithoutFormat,
                       .feature = &Features::storageReadWithoutFormat},
    RequestableFeature{.name = "storage_write_without_format",
                       .requested = &FeatureRequest::storageWriteWithoutFormat,
                       .feature = &Features::storageWriteWithoutFormat},
    RequestableFeature{.name = "storage_texture_multisample",
                       .requested = &FeatureRequest::storageTextureMultisample,
                       .feature = &Features::storageTextureMultisample},
    RequestableFeature{.name = "texture_dynamic_indexing",
                       .requested = &FeatureRequest::textureDynamicIndexing,
                       .feature = &Features::textureDynamicIndexing},
    RequestableFeature{.name = "storage_texture_dynamic_indexing",
                       .requested = &FeatureRequest::storageTextureDynamicIndexing,
                       .feature = &Features::storageTextureDynamicIndexing},
    RequestableFeature{.name = "storage_buffer_dynamic_indexing",
                       .requested = &FeatureRequest::storageBufferDynamicIndexing,
                       .feature = &Features::storageBufferDynamicIndexing},
    RequestableFeature{.name = "uniform_buffer_dynamic_indexing",
                       .requested = &FeatureRequest::uniformBufferDynamicIndexing,
                       .feature = &Features::uniformBufferDynamicIndexing},
    RequestableFeature{.name = "storage_texture_non_uniform_indexing",
                       .requested = &FeatureRequest::storageTextureNonUniformIndexing,
                       .feature = &Features::storageTextureNonUniformIndexing},
    RequestableFeature{.name = "uniform_buffer_non_uniform_indexing",
                       .requested = &FeatureRequest::uniformBufferNonUniformIndexing,
                       .feature = &Features::uniformBufferNonUniformIndexing},
    RequestableFeature{.name = "vertex_stores_and_atomics",
                       .requested = &FeatureRequest::vertexStoresAndAtomics,
                       .feature = &Features::vertexStoresAndAtomics},
    RequestableFeature{.name = "fragment_stores_and_atomics",
                       .requested = &FeatureRequest::fragmentStoresAndAtomics,
                       .feature = &Features::fragmentStoresAndAtomics},
    RequestableFeature{.name = "tessellation_geometry_point_size",
                       .requested = &FeatureRequest::tessellationGeometryPointSize,
                       .feature = &Features::tessellationGeometryPointSize},
    RequestableFeature{.name = "texture_gather_extended",
                       .requested = &FeatureRequest::textureGatherExtended,
                       .feature = &Features::textureGatherExtended},
    RequestableFeature{.name = "clip_distance",
                       .requested = &FeatureRequest::clipDistance,
                       .feature = &Features::clipDistance},
    RequestableFeature{.name = "cull_distance",
                       .requested = &FeatureRequest::cullDistance,
                       .feature = &Features::cullDistance},
    RequestableFeature{.name = "shader_min_lod",
                       .requested = &FeatureRequest::shaderMinLod,
                       .feature = &Features::shaderMinLod},
    RequestableFeature{.name = "storage_buffer_16bit_access",
                       .requested = &FeatureRequest::storageBuffer16bitAccess,
                       .feature = &Features::storageBuffer16bitAccess},
    RequestableFeature{.name = "uniform_and_storage_buffer_16bit_access",
                       .requested = &FeatureRequest::uniformAndStorageBuffer16bitAccess,
                       .feature = &Features::uniformAndStorageBuffer16bitAccess},
    RequestableFeature{.name = "push_constant_16bit",
                       .requested = &FeatureRequest::pushConstant16bit,
                       .feature = &Features::pushConstant16bit},
    RequestableFeature{.name = "input_output_16bit",
                       .requested = &FeatureRequest::inputOutput16bit,
                       .feature = &Features::inputOutput16bit},
    RequestableFeature{.name = "storage_buffer_8bit_access",
                       .requested = &FeatureRequest::storageBuffer8bitAccess,
                       .feature = &Features::storageBuffer8bitAccess},
    RequestableFeature{.name = "uniform_and_storage_buffer_8bit_access",
                       .requested = &FeatureRequest::uniformAndStorageBuffer8bitAccess,
                       .feature = &Features::uniformAndStorageBuffer8bitAccess},
    RequestableFeature{.name = "push_constant_8bit",
                       .requested = &FeatureRequest::pushConstant8bit,
                       .feature = &Features::pushConstant8bit},
    RequestableFeature{.name = "variable_pointers_storage_buffer",
                       .requested = &FeatureRequest::variablePointersStorageBuffer,
                       .feature = &Features::variablePointersStorageBuffer},
    RequestableFeature{.name = "variable_pointers",
                       .requested = &FeatureRequest::variablePointers,
                       .feature = &Features::variablePointers,
                       .prerequisites = {&FeatureRequest::variablePointersStorageBuffer}},
    RequestableFeature{.name = "draw_parameters",
                       .requested = &FeatureRequest::drawParameters,
                       .feature = &Features::drawParameters},
    RequestableFeature{.name = "scalar_block_layout",
                       .requested = &FeatureRequest::scalarBlockLayout,
                       .feature = &Features::scalarBlockLayout},
    RequestableFeature{.name = "uniform_buffer_standard_layout",
                       .requested = &FeatureRequest::uniformBufferStandardLayout,
                       .feature = &Features::uniformBufferStandardLayout},
    RequestableFeature{.name = "subgroup_extended_types",
                       .requested = &FeatureRequest::subgroupExtendedTypes,
                       .feature = &Features::subgroupExtendedTypes},
    RequestableFeature{.name = "memory_model",
                       .requested = &FeatureRequest::memoryModel,
                       .feature = &Features::memoryModel},
    RequestableFeature{.name = "memory_model_device_scope",
                       .requested = &FeatureRequest::memoryModelDeviceScope,
                       .feature = &Features::memoryModelDeviceScope},
    RequestableFeature{.name = "memory_model_visibility_chains",
                       .requested = &FeatureRequest::memoryModelVisibilityChains,
                       .feature = &Features::memoryModelVisibilityChains},
    RequestableFeature{.name = "shader_output_layer",
                       .requested = &FeatureRequest::shaderOutputLayer,
                       .feature = &Features::shaderOutputLayer},
    RequestableFeature{.name = "subgroup_broadcast_dynamic_id",
                       .requested = &FeatureRequest::subgroupBroadcastDynamicId,
                       .feature = &Features::subgroupBroadcastDynamicId},
    RequestableFeature{.name = "demote_to_helper_invocation",
                       .requested = &FeatureRequest::demoteToHelperInvocation,
                       .feature = &Features::demoteToHelperInvocation},
    RequestableFeature{.name = "terminate_invocation",
                       .requested = &FeatureRequest::terminateInvocation,
                       .feature = &Features::terminateInvocation},
    RequestableFeature{.name = "zero_initialize_workgroup_memory",
                       .requested = &FeatureRequest::zeroInitializeWorkgroupMemory,
                       .feature = &Features::zeroInitializeWorkgroupMemory},
    RequestableFeature{.name = "integer_dot_product",
                       .requested = &FeatureRequest::integerDotProduct,
                       .feature = &Features::integerDotProduct},
    RequestableFeature{.name = "local_size_id",
                       .requested = &FeatureRequest::localSizeId,
                       .feature = &Features::localSizeId},
    RequestableFeature{.name = "subgroup_rotate",
                       .requested = &FeatureRequest::subgroupRotate,
                       .feature = &Features::subgroupRotate},
    RequestableFeature{.name = "subgroup_rotate_clustered",
                       .requested = &FeatureRequest::subgroupRotateClustered,
                       .feature = &Features::subgroupRotateClustered},
    RequestableFeature{.name = "float_controls2",
                       .requested = &FeatureRequest::floatControls2,
                       .feature = &Features::floatControls2},
    RequestableFeature{.name = "expect_assume",
                       .requested = &FeatureRequest::expectAssume,
                       .feature = &Features::expectAssume},
    RequestableFeature{.name = "non_semantic_info",
                       .requested = &FeatureRequest::nonSemanticInfo,
                       .feature = &Features::nonSemanticInfo},
    RequestableFeature{.name = "pipeline_statistics",
                       .requested = &FeatureRequest::pipelineStatistics,
                       .feature = &Features::pipelineStatistics},
    RequestableFeature{.name = "occlusion_query_precise",
                       .requested = &FeatureRequest::occlusionQueryPrecise,
                       .feature = &Features::occlusionQueryPrecise},
    RequestableFeature{.name = "sparse_binding",
                       .requested = &FeatureRequest::sparseBinding,
                       .sparse = &Sparse::binding},
    RequestableFeature{.name = "sparse_buffers",
                       .requested = &FeatureRequest::sparseBuffers,
                       .sparse = &Sparse::buffers,
                       .prerequisites = {&FeatureRequest::sparseBinding}},
    RequestableFeature{.name = "sparse_textures_2d",
                       .requested = &FeatureRequest::sparseTextures2d,
                       .sparse = &Sparse::textures2D,
                       .prerequisites = {&FeatureRequest::sparseBinding}},
    RequestableFeature{.name = "sparse_textures_3d",
                       .requested = &FeatureRequest::sparseTextures3d,
                       .sparse = &Sparse::textures3D,
                       .prerequisites = {&FeatureRequest::sparseBinding}},
    RequestableFeature{.name = "sparse_samples_2",
                       .requested = &FeatureRequest::sparseSamples2,
                       .sparse = &Sparse::samples2,
                       .prerequisites = {&FeatureRequest::sparseBinding}},
    RequestableFeature{.name = "sparse_samples_4",
                       .requested = &FeatureRequest::sparseSamples4,
                       .sparse = &Sparse::samples4,
                       .prerequisites = {&FeatureRequest::sparseBinding}},
    RequestableFeature{.name = "sparse_samples_8",
                       .requested = &FeatureRequest::sparseSamples8,
                       .sparse = &Sparse::samples8,
                       .prerequisites = {&FeatureRequest::sparseBinding}},
    RequestableFeature{.name = "sparse_samples_16",
                       .requested = &FeatureRequest::sparseSamples16,
                       .sparse = &Sparse::samples16,
                       .prerequisites = {&FeatureRequest::sparseBinding}},
    RequestableFeature{.name = "sparse_aliased",
                       .requested = &FeatureRequest::sparseAliased,
                       .sparse = &Sparse::aliased,
                       .prerequisites = {&FeatureRequest::sparseBinding}},
    RequestableFeature{.name = "sparse_shader_residency",
                       .requested = &FeatureRequest::sparseShaderResidency,
                       .sparse = &Sparse::shaderResidency},
};

/// The index in kRequestableFeatures of the feature requested names.
[[nodiscard]] constexpr std::size_t requestableIndex(bool FeatureRequest::* requested) {
    for (std::size_t index = 0; index < kRequestableFeatures.size(); ++index) {
        if (kRequestableFeatures.at(index).requested == requested) {
            return index;
        }
    }
    return kRequestableFeatures.size();
}

/// Whether caps offer feature.
[[nodiscard]] inline bool offers(const Caps& caps, const RequestableFeature& feature) {
    if (feature.feature != nullptr) {
        return caps.features.*feature.feature;
    }
    if (feature.sparse != nullptr) {
        return caps.sparse.*feature.sparse;
    }
    const HostClocks& clocks = caps.timestamps.hostClocks;
    return clocks.monotonic || clocks.monotonicRaw || clocks.performanceCounter;
}

/// Fails, naming each feature concerned, with RhiInvalidCall when request
/// asks for a feature without one it needs, and with RhiUnsupported when it
/// asks for one BFR has no call for or one caps, those of the adapter named
/// adapter, don't offer. Each backend's createDevice calls it.
[[nodiscard]] inline core::Result<void> checkFeatures(const FeatureRequest& request, const Caps& caps,
                                                      std::string_view adapter) {
    std::vector<std::string_view> noCall;
    std::vector<std::string_view> lacked;
    std::vector<std::string> needs;
    for (const RequestableFeature& feature : kRequestableFeatures) {
        if (!(request.*feature.requested)) {
            continue;
        }
        for (const auto prerequisite : feature.prerequisites) {
            if (prerequisite != nullptr && !(request.*prerequisite)) {
                needs.push_back(std::format("{} needs {}", feature.name,
                                            kRequestableFeatures.at(requestableIndex(prerequisite)).name));
            }
        }
        if (feature.noCall) {
            noCall.push_back(feature.name);
        }
        else if (!offers(caps, feature)) {
            lacked.push_back(feature.name);
        }
    }
    if (!needs.empty()) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "features asked for without what they need: {}",
                             core::text::join(needs, ", "));
    }
    if (!noCall.empty()) {
        return core::failure(core::ErrorCode::RhiUnsupported, "BFR has no call for: {}",
                             core::text::join(noCall, ", "));
    }
    if (!lacked.empty()) {
        return core::failure(core::ErrorCode::RhiUnsupported, "{} lacks: {}", adapter, core::text::join(lacked, ", "));
    }
    return {};
}

/// Fails with RhiUnsupported when features, a device's, lack swapchain:
/// each backend's createSwapchain calls it first, so a device makes
/// swapchains only where it was asked to, on every backend.
[[nodiscard]] inline core::Result<void> checkSwapchainFeature(const Features& features) {
    if (!features.swapchain) {
        return core::failure(core::ErrorCode::RhiUnsupported, "the device was created without the swapchain feature");
    }
    return {};
}

/// caps as a device created with request has them: each requestable feature
/// true only where asked for, unless facts, in kRequestableFeatures' order,
/// says the device has it asked for or not; what follows one false with it
/// (as Vulkan's caps derive them); and no host clock without
/// timestampCalibration. D3D12's and Metal's device caps, of the adapter's,
/// with no facts; Vulkan's, of what the device enabled, with its version's.
[[nodiscard]] inline Caps requestedOnly(Caps caps, const FeatureRequest& request, std::span<const bool> facts = {}) {
    for (std::size_t index = 0; index < kRequestableFeatures.size(); ++index) {
        const RequestableFeature& feature = kRequestableFeatures.at(index);
        const bool asked = (request.*feature.requested) || (index < facts.size() && facts[index]);
        if (feature.feature != nullptr) {
            caps.features.*feature.feature = asked && caps.features.*feature.feature;
        }
        else if (feature.sparse != nullptr) {
            caps.sparse.*feature.sparse = asked && caps.sparse.*feature.sparse;
        }
        else if (!asked) {
            caps.timestamps.hostClocks = {};
        }
    }
    Features& features = caps.features;
    features.samplerAnisotropyNearestMip = features.samplerAnisotropyNearestMip && features.samplerAnisotropy;
    features.samplerAnisotropyNearestFilter = features.samplerAnisotropyNearestFilter && features.samplerAnisotropy;
    features.accelerationStructureSerialization = features.accelerationStructureSerialization &&
                                                  features.accelerationStructure;
    caps.sparse.mixedResources = caps.sparse.mixedResources && caps.sparse.binding;
    caps.sparse.layerTails = caps.sparse.layerTails && caps.sparse.binding;
    return caps;
}

} // namespace bfr::rhi
