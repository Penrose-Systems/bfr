#pragma once

#include "bfr/rhi/format.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// What an adapter is and can do. bfr.yaml documents each field under the C
// API's name for it.
namespace bfr::rhi {

/// Bytes in a UUID, and in a Windows LUID.
inline constexpr std::size_t kUuidSize = 16;
inline constexpr std::size_t kLuidSize = 8;

struct AdapterInfo {
    std::uint32_t vendorId = 0;
    std::uint32_t deviceId = 0;
    std::string driverName;
    std::string driverInfo;
    std::uint64_t driverVersion = 0;
    std::uint32_t apiMajor = 0;
    std::uint32_t apiMinor = 0;
    std::uint32_t apiPatch = 0;
    std::array<std::uint8_t, kUuidSize> deviceUuid{};
    std::array<std::uint8_t, kUuidSize> driverUuid{};
    std::array<std::uint8_t, kLuidSize> luid{};
    bool luidValid = false;
    std::array<std::uint8_t, kUuidSize> cacheUuid{};
};

struct Limits {
    std::uint32_t maxTexture1d = 0;
    std::uint32_t maxTexture2d = 0;
    std::uint32_t maxTexture3d = 0;
    std::uint32_t maxTextureCube = 0;
    std::uint32_t maxTextureLayers = 0;
    std::uint64_t maxBufferSize = 0;
    std::uint64_t maxAllocationSize = 0;
    std::uint32_t maxAllocations = 0;
    std::uint32_t maxSamplerObjects = 0;
    std::uint32_t maxUniformBufferRange = 0;
    std::uint32_t maxStorageBufferRange = 0;
    std::uint32_t maxTexelBufferElements = 0;
    std::uint32_t maxPushConstantsSize = 0;
    std::uint32_t maxColorAttachments = 0;
    std::uint32_t maxRenderWidth = 0;
    std::uint32_t maxRenderHeight = 0;
    std::uint32_t maxRenderLayers = 0;
    std::uint32_t colorSampleCounts = 0;
    std::uint32_t integerSampleCounts = 0;
    std::uint32_t depthSampleCounts = 0;
    std::uint32_t stencilSampleCounts = 0;
    std::uint32_t noAttachmentSampleCounts = 0;
    std::uint32_t storageTextureSampleCounts = 0;
    std::uint32_t maxViewports = 0;
    std::uint32_t maxViewportWidth = 0;
    std::uint32_t maxViewportHeight = 0;
    std::uint32_t maxDrawIndirectCount = 0;
    std::uint32_t maxClipDistances = 0;
    std::uint32_t maxCullDistances = 0;
    float maxSamplerAnisotropy = 0;
    float maxSamplerLodBias = 0;
    std::uint32_t maxCustomBorderColorSamplers = 0;
    /// Width and height.
    std::array<std::uint32_t, 2> minShadingRateTile{};
    std::array<std::uint32_t, 2> maxShadingRateTile{};
    std::uint64_t minUniformBufferAlignment = 0;
    std::uint64_t minStorageBufferAlignment = 0;
    std::uint64_t minTexelBufferAlignment = 0;
    std::uint64_t copyOffsetAlignment = 0;
    std::uint64_t copyRowPitchAlignment = 0;
    std::uint64_t nonCoherentAtomSize = 0;
    std::uint64_t bufferTextureGranularity = 0;
};

struct BindlessLimits {
    std::uint32_t maxTextures = 0;
    std::uint32_t maxStorageTextures = 0;
    std::uint32_t maxStorageBuffers = 0;
    std::uint32_t maxUniformBuffers = 0;
    std::uint64_t maxSamplers = 0;
    std::uint32_t maxPerStage = 0;
    std::uint32_t maxAccelerationStructures = 0;
};

/// Dimensions of a workgroup.
inline constexpr std::size_t kWorkgroupDimensions = 3;

struct ComputeLimits {
    std::array<std::uint32_t, kWorkgroupDimensions> maxWorkgroupCount{};
    std::array<std::uint64_t, kWorkgroupDimensions> maxWorkgroupSize{};
    std::uint32_t maxWorkgroupInvocations = 0;
    std::uint64_t maxSharedMemory = 0;
    std::uint32_t subgroupSize = 0;
    std::uint32_t minSubgroupSize = 0;
    std::uint32_t maxSubgroupSize = 0;
};

struct MeshLimits {
    std::array<std::uint32_t, kWorkgroupDimensions> maxMeshWorkgroupSize{};
    std::uint32_t maxMeshInvocations = 0;
    std::uint32_t maxMeshOutputVertices = 0;
    std::uint32_t maxMeshOutputPrimitives = 0;
    std::uint32_t maxMeshSharedMemory = 0;
    std::array<std::uint32_t, kWorkgroupDimensions> maxTaskWorkgroupSize{};
    std::uint32_t maxTaskInvocations = 0;
    std::uint32_t maxTaskPayloadSize = 0;
    /// What a drawMesh's counts stay within: of the task shader's workgroups
    /// where the pipeline has one, else of the mesh shader's.
    std::array<std::uint32_t, kWorkgroupDimensions> maxTaskWorkgroupCount{};
    std::uint32_t maxTaskWorkgroupTotalCount = 0;
    std::uint32_t maxTaskSharedMemory = 0;
    std::uint32_t maxTaskPayloadAndSharedMemory = 0;
    std::array<std::uint32_t, kWorkgroupDimensions> maxMeshWorkgroupCount{};
    std::uint32_t maxMeshWorkgroupTotalCount = 0;
    /// The bytes a mesh shader outputs, its vertices' and primitives' sizes
    /// each rounded up to their granularity.
    std::uint32_t maxMeshOutputMemory = 0;
    std::uint32_t meshOutputPerVertexGranularity = 0;
    std::uint32_t meshOutputPerPrimitiveGranularity = 0;
    std::uint32_t maxMeshPayloadAndSharedMemory = 0;
    std::uint32_t maxMeshPayloadAndOutputMemory = 0;
    std::uint32_t maxMeshOutputComponents = 0;
    std::uint32_t maxMeshOutputLayers = 0;
    std::uint32_t maxMeshMultiviewViews = 0;
};

struct RayTracingLimits {
    std::uint64_t maxGeometries = 0;
    std::uint64_t maxInstances = 0;
    std::uint64_t maxPrimitives = 0;
    std::uint32_t scratchAlignment = 0;
    std::uint32_t maxRecursionDepth = 0;
    /// The bytes of a shader group's identifier, which a binding table
    /// record starts with, and the alignment of a record's start.
    std::uint32_t identifierSize = 0;
    std::uint32_t identifierAlignment = 0;
    /// The alignment of a binding table region's start.
    std::uint32_t tableAlignment = 0;
    std::uint32_t maxShaderGroupStride = 0;
    /// The most rays one trace makes.
    std::uint32_t maxTraceRays = 0;
    std::uint32_t maxHitAttributeSize = 0;
    /// The bytes of one entry resolveAccelerationStructureProperties writes.
    std::uint32_t propertyEntryBytes = 0;
    /// Shaders that reorder invocations may see them reordered: the backend
    /// says it reorders (Features::rayInvocationReorder).
    bool reordersInvocations = false;
    /// The highest binding table record index a reordered hit names.
    std::uint32_t maxBindingTableRecordIndex = 0;
    /// The bytes of one record Device::writeInstances writes.
    std::uint32_t instanceSize = 0;
    /// The bytes of one group's identifier Device::captureReplayIdentifiers
    /// writes.
    std::uint32_t captureReplayIdentifierSize = 0;
    /// The highest subdivision level of a two-state and a four-state
    /// opacity micromap.
    std::uint32_t maxOpacity2StateSubdivisionLevel = 0;
    std::uint32_t maxOpacity4StateSubdivisionLevel = 0;
    /// The most keyframes a structure with motion, or a motion instance, has.
    std::uint32_t maxMotionKeyframes = 0;
    /// The bytes from one record Device::writeMotionInstances writes to the
    /// next.
    std::uint32_t motionInstanceSize = 0;
    /// The most partitions, the global partition's instances counted as
    /// partitions, a partitioned top structure holds.
    std::uint32_t maxPartitions = 0;
    /// The most vertices and triangles one cluster holds.
    std::uint32_t maxClusterVertices = 0;
    std::uint32_t maxClusterTriangles = 0;
    /// The alignment in bytes of a cluster operation's scratch, and of the
    /// clusters, templates, cluster bottom structures and template bounds it
    /// writes or reads.
    std::uint32_t clusterScratchAlignment = 0;
    std::uint32_t clusterAlignment = 0;
    std::uint32_t clusterTemplateAlignment = 0;
    std::uint32_t clusterBottomAlignment = 0;
    std::uint32_t clusterTemplateBoundsAlignment = 0;
    /// The geometry indices of clusters are below it.
    std::uint32_t maxClusterGeometryIndex = 0;
};

struct Features {
    /// Rays traced inline from compute shaders.
    bool rayQuery = false;
    /// Rays traced inline from vertex, fragment, task and mesh shaders.
    bool rayQueryGraphics = false;
    bool rayTracingPipeline = false;
    /// Device::traceRaysIndirect.
    bool traceRaysIndirect = false;
    /// Rays that skip triangles or boxes.
    bool rayPrimitiveCulling = false;
    bool accelerationStructure = false;
    /// Device::bufferAddress.
    bool bufferAddress = false;
    /// Shaders read a hit triangle's positions, of a structure built with
    /// allowDataAccess.
    bool rayPositionFetch = false;
    /// Device::serializeAccelerationStructure and its kin.
    bool accelerationStructureSerialization = false;
    /// The CurrentSize and SerializedBottomCount properties.
    bool accelerationStructureCurrentSize = false;
    bool serializedBottomCount = false;
    /// Shaders reorder ray tracing invocations.
    bool rayInvocationReorder = false;
    /// Ray tracing pipeline libraries, RayTracingPipelineDesc::library.
    bool rayTracingPipelineLibrary = false;
    /// Device::shaderIdentifiers of a library, the same as a pipeline that
    /// links it gives for its groups.
    bool pipelineLibraryIdentifiers = false;
    /// RayTracingPipelineDesc::captureReplay.
    bool rayTracingCaptureReplay = false;
    /// A device makes pipelines with replayed identifiers and pipelines
    /// without them in any order, at once.
    bool rayTracingCaptureReplayMixed = false;
    /// Device::buildAccelerationStructureIndirect.
    bool accelerationStructureIndirectBuild = false;
    /// AccelerationStructureType::OpacityMicromap: structures of micromaps,
    /// which bottom structures' triangles take their opacity from.
    bool opacityMicromap = false;
    /// Structures with motion, AccelerationStructureDesc::motionKeyframes,
    /// traced at a time by pipelines made with allowMotion.
    bool rayTracingMotion = false;
    /// Device::traceRaysIndirect with a pipeline made with allowMotion.
    bool rayTracingMotionIndirect = false;
    /// AccelerationStructureType::PartitionedTop, and its table
    /// (DeviceDesc::maxBindlessPartitionedTops).
    bool partitionedTop = false;
    /// Device::buildClusters, and traces of the cluster bottom structures it
    /// builds by pipelines made with allowClusters.
    bool clusterAccelerationStructure = false;
    /// AccelerationStructureDesc::captureReplay.
    bool accelerationStructureCaptureReplay = false;
    bool meshShader = false;
    bool taskShader = false;
    bool shadingRate = false;
    bool shadingRateTexture = false;
    bool shaderFloat16 = false;
    bool shaderFloat64 = false;
    bool shaderInt8 = false;
    bool shaderInt16 = false;
    bool shaderInt64 = false;
    bool atomicInt64 = false;
    bool atomicInt64Texture = false;
    bool atomicFloat32Add = false;
    bool barycentrics = false;
    bool conservativeRaster = false;
    bool geometryShader = false;
    bool tessellation = false;
    bool multiview = false;
    bool sampleRateShading = false;
    bool dualSourceBlend = false;
    bool independentBlend = false;
    bool logicOp = false;
    bool depthClamp = false;
    bool depthBounds = false;
    bool wireframe = false;
    bool wideLines = false;
    bool samplerAnisotropy = false;
    bool samplerMinmax = false;
    bool samplerMirrorClampToEdge = false;
    bool samplerAnisotropyNearestMip = false;
    bool samplerAnisotropyNearestFilter = false;
    bool samplerCustomBorderColor = false;
    bool samplerIntegerBorderColor = false;
    bool samplerUnnormalizedCoordinates = false;
    /// A 3D texture with colorAttachment usage is drawn to by depth slice:
    /// a ColorAttachment's layer and layerCount name its slices. Not for a
    /// sparse one on Vulkan, which draws to slices only through 2D views of
    /// an image made for them, which sparse images may not be.
    bool volumeColorTargets = false;
    bool samplerLodBias = false;
    bool samplerLodAverage = false;
    bool compressionBc = false;
    bool compressionEtc2 = false;
    bool compressionAstc = false;
    bool compressionAstcHdr = false;
    bool multiDrawIndirect = false;
    bool drawIndirectCount = false;
    bool drawIndirectFirstInstance = false;
    /// Topology::TriangleFan draws.
    bool triangleFans = false;
    /// PrimitiveRestart on list topologies, not only strips.
    bool listRestart = false;
    /// Device::setTopology, which fails with RhiUnsupported where this is false.
    bool dynamicTopology = false;
    /// Mesh pipelines in multiview renderings, where multiview.
    bool meshMultiview = false;
    /// Pipeline statistics and primitive queries count task and mesh work.
    bool meshStatistics = false;
    /// GraphicsPipelineDesc::alphaToOne.
    bool alphaToOne = false;
    /// A GraphicsPipelineDesc::sampleMask that leaves samples out.
    bool sampleMask = false;
    /// Device::createPipelineCache makes caches.
    bool pipelineCache = false;
    bool storageReadWithoutFormat = false;
    bool storageWriteWithoutFormat = false;
    /// A storage texture table indexed by a value that differs within a
    /// shader's invocation group.
    bool storageTextureNonUniformIndexing = false;
    /// The uniform buffer table indexed by a value that differs within a
    /// shader's invocation group.
    bool uniformBufferNonUniformIndexing = false;
    /// Every queue whose family runs graphics or compute writes timestamps.
    bool timestampsOnAllQueues = false;
    /// Device::createQueryPool makes Occlusion pools, and PipelineStatistics
    /// and MeshPrimitivesGenerated ones where pipelineStatistics and
    /// meshStatistics say.
    bool queries = false;
    bool pipelineStatistics = false;
    bool occlusionQueryPrecise = false;
    bool swapchain = false;
    bool robustBufferAccess = false;
    bool robustImageAccess = false;
    bool meshShadingRate = false;
    bool atomicInt64Shared = false;
    bool sparseAtomicInt64Texture = false;
    bool atomicFloat32 = false;
    bool atomicFloat64 = false;
    bool atomicFloat64Add = false;
    bool atomicFloat32Shared = false;
    bool atomicFloat32AddShared = false;
    bool atomicFloat64Shared = false;
    bool atomicFloat64AddShared = false;
    bool atomicFloat32Texture = false;
    bool atomicFloat32AddTexture = false;
    bool sparseAtomicFloat32Texture = false;
    bool sparseAtomicFloat32AddTexture = false;
    bool multiviewGeometryShader = false;
    bool multiviewTessellationShader = false;
    bool depthBiasClamp = false;
    bool imageCubeArray = false;
    bool formatBgra4Unorm = false;
    bool fullDrawIndexUint32 = false;
    bool storageTextureMultisample = false;
    bool textureDynamicIndexing = false;
    bool storageTextureDynamicIndexing = false;
    bool storageBufferDynamicIndexing = false;
    bool uniformBufferDynamicIndexing = false;
    bool vertexStoresAndAtomics = false;
    bool fragmentStoresAndAtomics = false;
    bool tessellationGeometryPointSize = false;
    bool textureGatherExtended = false;
    bool clipDistance = false;
    bool cullDistance = false;
    bool shaderMinLod = false;
    bool storageBuffer16bitAccess = false;
    bool uniformAndStorageBuffer16bitAccess = false;
    bool pushConstant16bit = false;
    bool inputOutput16bit = false;
    bool storageBuffer8bitAccess = false;
    bool uniformAndStorageBuffer8bitAccess = false;
    bool pushConstant8bit = false;
    bool variablePointersStorageBuffer = false;
    bool variablePointers = false;
    bool drawParameters = false;
    bool scalarBlockLayout = false;
    bool uniformBufferStandardLayout = false;
    bool subgroupExtendedTypes = false;
    bool memoryModel = false;
    bool memoryModelDeviceScope = false;
    bool memoryModelVisibilityChains = false;
    bool shaderOutputLayer = false;
    bool shaderOutputViewportIndex = false;
    bool multiViewport = false;
    bool largePoints = false;
    bool subgroupBroadcastDynamicId = false;
    bool demoteToHelperInvocation = false;
    bool terminateInvocation = false;
    bool zeroInitializeWorkgroupMemory = false;
    bool integerDotProduct = false;
    bool localSizeId = false;
    bool subgroupRotate = false;
    bool subgroupRotateClustered = false;
    bool floatControls2 = false;
    bool expectAssume = false;
    bool nonSemanticInfo = false;
    /// Device::beginDebugLabel, endDebugLabel and insertDebugLabel.
    bool debugLabels = false;
    /// Device::beginQueueDebugLabel, endQueueDebugLabel and
    /// insertQueueDebugLabel.
    bool queueDebugLabels = false;
    /// The name field of the descs of buffers, textures, pipelines, pipeline
    /// caches, timestamp and query pools and acceleration structures; where
    /// false, a create given a name fails with RhiUnsupported.
    bool objectNames = false;
};

/// The optional features a device is created with, each named as its
/// Features member (sparse ones as Sparse's, with a sparse prefix), and
/// timestampCalibration for calibrateTimestamps; all false enables none of
/// them. bfr.yaml's bfr_features documents each.
struct FeatureRequest {
    bool swapchain = false;
    bool timestampCalibration = false;
    bool robustBufferAccess = false;
    bool robustImageAccess = false;
    bool rayQuery = false;
    bool rayQueryGraphics = false;
    bool rayTracingPipeline = false;
    bool traceRaysIndirect = false;
    bool rayPrimitiveCulling = false;
    bool accelerationStructure = false;
    bool bufferAddress = false;
    bool rayPositionFetch = false;
    bool accelerationStructureCurrentSize = false;
    bool serializedBottomCount = false;
    bool rayInvocationReorder = false;
    bool rayTracingPipelineLibrary = false;
    bool pipelineLibraryIdentifiers = false;
    bool rayTracingCaptureReplay = false;
    bool rayTracingCaptureReplayMixed = false;
    bool accelerationStructureIndirectBuild = false;
    bool opacityMicromap = false;
    bool rayTracingMotion = false;
    bool rayTracingMotionIndirect = false;
    bool partitionedTop = false;
    bool clusterAccelerationStructure = false;
    bool accelerationStructureCaptureReplay = false;
    bool meshShader = false;
    bool taskShader = false;
    bool meshMultiview = false;
    bool meshStatistics = false;
    bool meshShadingRate = false;
    bool shaderFloat16 = false;
    bool shaderFloat64 = false;
    bool shaderInt8 = false;
    bool shaderInt16 = false;
    bool shaderInt64 = false;
    bool atomicInt64 = false;
    bool atomicInt64Shared = false;
    bool atomicInt64Texture = false;
    bool sparseAtomicInt64Texture = false;
    bool atomicFloat32 = false;
    bool atomicFloat32Add = false;
    bool atomicFloat64 = false;
    bool atomicFloat64Add = false;
    bool atomicFloat32Shared = false;
    bool atomicFloat32AddShared = false;
    bool atomicFloat64Shared = false;
    bool atomicFloat64AddShared = false;
    bool atomicFloat32Texture = false;
    bool atomicFloat32AddTexture = false;
    bool sparseAtomicFloat32Texture = false;
    bool sparseAtomicFloat32AddTexture = false;
    bool barycentrics = false;
    bool geometryShader = false;
    bool tessellation = false;
    bool multiview = false;
    bool multiviewGeometryShader = false;
    bool multiviewTessellationShader = false;
    bool sampleRateShading = false;
    bool dualSourceBlend = false;
    bool independentBlend = false;
    bool logicOp = false;
    bool depthClamp = false;
    bool depthBiasClamp = false;
    bool depthBounds = false;
    bool wireframe = false;
    bool wideLines = false;
    bool samplerAnisotropy = false;
    bool samplerMinmax = false;
    bool samplerMirrorClampToEdge = false;
    bool samplerCustomBorderColor = false;
    bool volumeColorTargets = false;
    bool samplerLodBias = false;
    bool imageCubeArray = false;
    bool compressionAstcHdr = false;
    bool formatBgra4Unorm = false;
    bool multiDrawIndirect = false;
    bool drawIndirectCount = false;
    bool drawIndirectFirstInstance = false;
    bool fullDrawIndexUint32 = false;
    bool triangleFans = false;
    bool listRestart = false;
    bool dynamicTopology = false;
    bool alphaToOne = false;
    bool storageReadWithoutFormat = false;
    bool storageWriteWithoutFormat = false;
    bool storageTextureMultisample = false;
    bool textureDynamicIndexing = false;
    bool storageTextureDynamicIndexing = false;
    bool storageBufferDynamicIndexing = false;
    bool uniformBufferDynamicIndexing = false;
    bool storageTextureNonUniformIndexing = false;
    bool uniformBufferNonUniformIndexing = false;
    bool vertexStoresAndAtomics = false;
    bool fragmentStoresAndAtomics = false;
    bool tessellationGeometryPointSize = false;
    bool textureGatherExtended = false;
    bool clipDistance = false;
    bool cullDistance = false;
    bool shaderMinLod = false;
    bool storageBuffer16bitAccess = false;
    bool uniformAndStorageBuffer16bitAccess = false;
    bool pushConstant16bit = false;
    bool inputOutput16bit = false;
    bool storageBuffer8bitAccess = false;
    bool uniformAndStorageBuffer8bitAccess = false;
    bool pushConstant8bit = false;
    bool variablePointersStorageBuffer = false;
    bool variablePointers = false;
    bool drawParameters = false;
    bool scalarBlockLayout = false;
    bool uniformBufferStandardLayout = false;
    bool subgroupExtendedTypes = false;
    bool memoryModel = false;
    bool memoryModelDeviceScope = false;
    bool memoryModelVisibilityChains = false;
    bool shaderOutputLayer = false;
    bool shaderOutputViewportIndex = false;
    bool multiViewport = false;
    bool largePoints = false;
    bool subgroupBroadcastDynamicId = false;
    bool demoteToHelperInvocation = false;
    bool terminateInvocation = false;
    bool zeroInitializeWorkgroupMemory = false;
    bool integerDotProduct = false;
    bool localSizeId = false;
    bool subgroupRotate = false;
    bool subgroupRotateClustered = false;
    bool floatControls2 = false;
    bool expectAssume = false;
    bool nonSemanticInfo = false;
    bool pipelineStatistics = false;
    bool occlusionQueryPrecise = false;
    bool sparseBinding = false;
    bool sparseBuffers = false;
    bool sparseTextures2d = false;
    bool sparseTextures3d = false;
    bool sparseSamples2 = false;
    bool sparseSamples4 = false;
    bool sparseSamples8 = false;
    bool sparseSamples16 = false;
    bool sparseAliased = false;
    bool sparseShaderResidency = false;
};

struct MemoryHeap {
    std::uint64_t size = 0;
    bool deviceLocal = false;
    bool hostVisible = false;
};

struct MemoryType {
    std::uint32_t heap = 0;
    bool deviceLocal = false;
    bool hostVisible = false;
    bool hostCoherent = false;
    bool hostCached = false;
    bool lazilyAllocated = false;
};

/// At most 16 heaps and 32 types, as every backend has.
struct Memory {
    std::vector<MemoryHeap> heaps;
    std::vector<MemoryType> types;
    bool unified = false;
};

/// The QueueLevel values a family's queues take; each true where the backend
/// says so, none where it has no setting.
struct QueueLevels {
    bool normal = false;
    bool low = false;
    bool high = false;
    bool realtime = false;
};

/// How a copy between a buffer and a texture, or between two textures, lines
/// up in the lists of a family's queues, in texels, or texel blocks for a compressed
/// format: its offsets are multiples of width, height and depth, and each
/// size is a multiple too or reaches the edge of the mip level. With
/// wholeLevels it copies whole mip levels only, from 0. All 0 and false
/// where the backend states none.
struct CopyGranularity {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t depth = 0;
    bool wholeLevels = false;
};

/// The TimestampStage values a timestamp write takes; each true where the
/// backend says so.
struct TimestampStages {
    bool end = false;
    bool start = false;
    bool vertex = false;
    bool fragment = false;
    bool compute = false;
    bool copy = false;
    bool task = false;
    bool mesh = false;
    bool rayTracing = false;
    bool accelerationStructureBuild = false;
};

/// The HostClock values calibrateTimestamps reads beside a GPU timestamp.
struct HostClocks {
    bool monotonic = false;
    bool monotonicRaw = false;
    bool performanceCounter = false;
};

/// One of an adapter's queue families, as the backend gives it: what its
/// queues run, how many it has (0 where the backend has no count, and
/// createDevice then takes any number), and their timestamps, levels and
/// copy granularity. A timestamp write on one of its queues outside rendering
/// takes the stages in timestampStages.
struct QueueFamily {
    bool graphics = false;
    bool compute = false;
    bool copy = false;
    /// Its queues run Device::updateTileMappings.
    bool sparse = false;
    std::uint32_t count = 0;
    std::uint32_t timestampBits = 0;
    bool timestamps = false;
    TimestampStages timestampStages;
    /// Its queues record resetTimestamps. A pool written on a queue whose
    /// family can't is reset on one whose family can, ordered before the
    /// writes by a submit's wait.
    bool timestampResets = false;
    QueueLevels levels;
    CopyGranularity granularity;
};

/// What the adapter's GPU timestamps are on every family alike. A write
/// between beginRendering and endRendering takes the stages in
/// insideRendering on every family that runs graphics. granularity says the
/// backend takes a TimestampGranularity; where it is false the value is
/// unused. entryBytes is what one timestamp takes in a buffer
/// resolveTimestamps writes, and hostClocks the clocks calibrateTimestamps
/// reads.
struct Timestamps {
    TimestampStages insideRendering;
    bool granularity = false;
    std::uint64_t entryBytes = 0;
    HostClocks hostClocks;
};

/// What the adapter's sparse resources are; each true where the backend says
/// so, or its documentation where it has no query, as each backend names.
struct Sparse {
    /// Resources made with sparse take tiles at all.
    bool binding = false;
    bool buffers = false;
    bool textures2D = false;
    bool textures3D = false;
    /// Sparse 2D textures with 2, 4, 8 and 16 samples.
    bool samples2 = false;
    bool samples4 = false;
    bool samples8 = false;
    bool samples16 = false;
    /// Resources made with sparseAliased share memory's tiles.
    bool aliased = false;
    /// Tiles have the standard shapes, for 2D, multisampled 2D and 3D textures.
    bool standard2DShape = false;
    bool standard2DMultisampleShape = false;
    bool standard3DShape = false;
    /// Every mip level not a whole number of tiles is in the tail.
    bool alignedMipSize = false;
    /// Reads of an unmapped tile give 0 and writes to one are dropped.
    bool nonResidentStrict = false;
    /// Shaders ask whether what they read was mapped.
    bool shaderResidency = false;
    /// One SparseMemory holds textures' and buffers' tiles together.
    bool mixedResources = false;
    /// Tail mappings take a layer other than 0.
    bool layerTails = false;
    /// Bytes of sparse resources a device holds, where the backend counts them so.
    std::uint64_t addressSpaceBytes = 0;
    /// Bits of a resource's and of the process's GPU addresses, where the
    /// backend counts them so.
    std::uint32_t addressBitsPerResource = 0;
    std::uint32_t addressBitsPerProcess = 0;
    /// The tile sizes, in bytes, a sparse resource is made with; empty where
    /// the backend sets them itself, and TextureTiling and BufferTiling say
    /// what they are. At most kMaxSparseTileSizes.
    std::vector<std::uint32_t> tileBytes;
};

/// The most tile sizes a backend offers: Metal's three sparse page sizes.
inline constexpr std::size_t kMaxSparseTileSizes = 3;

/// Everything about an adapter that stays the same while it runs.
/// How a rendering's depth-stencil attachment resolves. depthModes and
/// stencilModes have bit 1 << mode set for each ResolveMode but None the
/// device takes; independent: depth and stencil may take different modes;
/// independentNone: they may differ where one is None.
struct ResolveCaps {
    /// Device::resolveTexture resolves outside a rendering; false on Metal,
    /// which resolves only as a render pass ends.
    bool texture = false;
    std::uint32_t depthModes = 0;
    std::uint32_t stencilModes = 0;
    bool independent = false;
    bool independentNone = false;
};

struct Caps {
    AdapterInfo info;
    Limits limits;
    BindlessLimits bindless;
    ComputeLimits compute;
    MeshLimits mesh;
    RayTracingLimits rayTracing;
    Features features;
    Memory memory;
    /// Every queue family the backend has, in its order; a QueueDesc names
    /// one by its index here.
    std::vector<QueueFamily> queueFamilies;
    /// How many distinct priorities the backend gives a device's queues by
    /// their QueueDesc::rank; 0 where it doesn't rank them.
    std::uint32_t queueRanks = 0;
    Timestamps timestamps;
    Sparse sparse;
    ResolveCaps resolve;
};

struct FormatCaps {
    Format format = Format::Undefined;
    bool sampled = false;
    bool filterLinear = false;
    bool filterMinmax = false;
    bool storage = false;
    bool storageAtomic = false;
    bool colorAttachment = false;
    bool blend = false;
    bool depthStencil = false;
    bool copySrc = false;
    bool copyDst = false;
    bool texelBuffer = false;
    bool storageTexelBuffer = false;
    bool storageTexelBufferAtomic = false;
    std::uint32_t sampleCounts = 0;
    /// Multisampled textures of it resolve into single-sample ones: color
    /// formats in a rendering and by Device::resolveTexture, depth and stencil
    /// formats in a rendering, by the modes ResolveCaps names.
    bool resolve = false;
    /// TriangleGeometry::vertexFormat takes it.
    bool accelerationStructureVertices = false;
};

struct SwapchainCaps {
    std::uint32_t minImageCount = 0;
    std::uint32_t maxImageCount = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t minWidth = 0;
    std::uint32_t minHeight = 0;
    std::uint32_t maxWidth = 0;
    std::uint32_t maxHeight = 0;
    DisplayRotation rotation = DisplayRotation::None;
    bool fifo = false;
    bool fifoRelaxed = false;
    bool mailbox = false;
    bool immediate = false;
    bool alphaOpaque = false;
    bool alphaPremultiplied = false;
    bool alphaPostmultiplied = false;
    bool alphaInherit = false;
};

struct SwapchainFormat {
    Format format = Format::Undefined;
    ColorSpace colorSpace{};
};

struct HeapBudget {
    std::uint64_t budget = 0;
    std::uint64_t usage = 0;
};

/// One entry per heap, in Memory's order.
struct MemoryBudget {
    std::vector<HeapBudget> heaps;
};

} // namespace bfr::rhi
