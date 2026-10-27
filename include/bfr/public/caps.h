#pragma once

#include "format.h"

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
    std::uint32_t minShadingRateTile = 0;
    std::uint32_t maxShadingRateTile = 0;
    std::uint64_t minUniformBufferAlignment = 0;
    std::uint64_t minStorageBufferAlignment = 0;
    std::uint64_t minTexelBufferAlignment = 0;
    std::uint64_t copyOffsetAlignment = 0;
    std::uint64_t copyRowPitchAlignment = 0;
    std::uint64_t nonCoherentAtomSize = 0;
    std::uint64_t bufferTextureGranularity = 0;
    float timestampPeriod = 0;
};

struct BindlessLimits {
    std::uint32_t maxTextures = 0;
    std::uint32_t maxStorageTextures = 0;
    std::uint32_t maxStorageBuffers = 0;
    std::uint32_t maxUniformBuffers = 0;
    std::uint32_t maxSamplers = 0;
    std::uint32_t maxPerStage = 0;
};

struct ComputeLimits {
    std::array<std::uint32_t, 3> maxWorkgroupCount{};
    std::array<std::uint32_t, 3> maxWorkgroupSize{};
    std::uint32_t maxWorkgroupInvocations = 0;
    std::uint32_t maxSharedMemory = 0;
    std::uint32_t subgroupSize = 0;
    std::uint32_t minSubgroupSize = 0;
    std::uint32_t maxSubgroupSize = 0;
};

struct MeshLimits {
    std::array<std::uint32_t, 3> maxMeshWorkgroupSize{};
    std::uint32_t maxMeshInvocations = 0;
    std::uint32_t maxMeshOutputVertices = 0;
    std::uint32_t maxMeshOutputPrimitives = 0;
    std::uint32_t maxMeshSharedMemory = 0;
    std::array<std::uint32_t, 3> maxTaskWorkgroupSize{};
    std::uint32_t maxTaskInvocations = 0;
    std::uint32_t maxTaskPayloadSize = 0;
};

struct RayTracingLimits {
    std::uint64_t maxGeometries = 0;
    std::uint64_t maxInstances = 0;
    std::uint64_t maxPrimitives = 0;
    std::uint32_t scratchAlignment = 0;
    std::uint32_t maxRecursionDepth = 0;
    std::uint32_t handleSize = 0;
    std::uint32_t handleAlignment = 0;
    std::uint32_t tableAlignment = 0;
};

struct Features {
    bool rayQuery = false;
    bool rayTracingPipeline = false;
    bool accelerationStructure = false;
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
    bool compressionBc = false;
    bool compressionEtc2 = false;
    bool compressionAstc = false;
    bool compressionAstcHdr = false;
    bool multiDrawIndirect = false;
    bool drawIndirectCount = false;
    bool drawIndirectFirstInstance = false;
    bool storageReadWithoutFormat = false;
    bool storageWriteWithoutFormat = false;
    bool timestampsOnAllQueues = false;
    bool pipelineStatistics = false;
    bool occlusionQueryPrecise = false;
    bool sparseBinding = false;
    bool sparseTextures = false;
    bool sparseBuffers = false;
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

struct Queues {
    std::uint32_t graphics = 0;
    std::uint32_t compute = 0;
    std::uint32_t copy = 0;
    std::uint32_t graphicsTimestampBits = 0;
    std::uint32_t computeTimestampBits = 0;
    std::uint32_t copyTimestampBits = 0;
};

/// Everything about an adapter that stays the same while it runs.
struct Caps {
    AdapterInfo info;
    Limits limits;
    BindlessLimits bindless;
    ComputeLimits compute;
    MeshLimits mesh;
    RayTracingLimits rayTracing;
    Features features;
    Memory memory;
    Queues queues;
};

struct FormatCaps {
    Format format = Format::R8Unorm;
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
};

struct SurfaceCaps {
    std::uint32_t minImageCount = 0;
    std::uint32_t maxImageCount = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t minWidth = 0;
    std::uint32_t minHeight = 0;
    std::uint32_t maxWidth = 0;
    std::uint32_t maxHeight = 0;
    SurfaceRotation rotation = SurfaceRotation::None;
    bool fifo = false;
    bool fifoRelaxed = false;
    bool mailbox = false;
    bool immediate = false;
    bool alphaOpaque = false;
    bool alphaPremultiplied = false;
    bool alphaPostmultiplied = false;
    bool alphaInherit = false;
};

struct SurfaceFormat {
    Format format = Format::R8Unorm;
    ColorSpace colorSpace = ColorSpace::Srgb;
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
