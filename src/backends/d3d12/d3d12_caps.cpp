#include "d3d12/d3d12_caps.h"

#include "bfr/rhi/caps.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/instance.h"
#include "d3d12/d3d12_formats.h"
#include "d3d12/d3d12_pipeline.h"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <windows.h>
#include <wrl/client.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace bfr::rhi::d3d12 {

namespace {

using Microsoft::WRL::ComPtr;

// A feature level holds its major from bit 12 and its minor in the 4 bits from
// bit 8: 0xc100 is 12_1.
constexpr std::uint32_t kMajorShift = 12;
constexpr std::uint32_t kMinorShift = 8;
constexpr std::uint32_t kMinorMask = 0xF;

// Asks device for feature; false when it won't say, leaving data as it was.
template <class Data>
bool query(ID3D12Device* device, D3D12_FEATURE feature, Data& data) {
    return SUCCEEDED(device->CheckFeatureSupport(feature, &data, sizeof(data)));
}

// Whether the adapter's memory is the machine's, as D3D12 answers it; false
// when it won't say.
bool unifiedMemory(ID3D12Device* device) {
    D3D12_FEATURE_DATA_ARCHITECTURE architecture{.NodeIndex = 0,
                                                 .TileBasedRenderer = FALSE,
                                                 .UMA = FALSE,
                                                 .CacheCoherentUMA = FALSE};
    return query(device, D3D12_FEATURE_ARCHITECTURE, architecture) && architecture.UMA != FALSE;
}

// The highest D3D_FEATURE_LEVEL the device runs, 0 when it won't say.
std::uint32_t featureLevelOf(ID3D12Device* device) {
    constexpr std::array kLevels{D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_12_0,
                                 D3D_FEATURE_LEVEL_12_1, D3D_FEATURE_LEVEL_12_2};
    D3D12_FEATURE_DATA_FEATURE_LEVELS data{.NumFeatureLevels = static_cast<UINT>(kLevels.size()),
                                           .pFeatureLevelsRequested = kLevels.data(),
                                           .MaxSupportedFeatureLevel = D3D_FEATURE_LEVEL_11_0};
    if (!query(device, D3D12_FEATURE_FEATURE_LEVELS, data)) {
        return 0;
    }
    return static_cast<std::uint32_t>(data.MaxSupportedFeatureLevel);
}

AdapterInfo infoOf(const DXGI_ADAPTER_DESC1& desc, ID3D12Device* device) {
    AdapterInfo info;
    info.vendorId = desc.VendorId;
    info.deviceId = desc.DeviceId;
    // The API version is the feature level.
    const std::uint32_t level = featureLevelOf(device);
    info.apiMajor = level >> kMajorShift;
    info.apiMinor = (level >> kMinorShift) & kMinorMask;
    static_assert(sizeof(LUID) == kLuidSize);
    std::memcpy(info.luid.data(), &desc.AdapterLuid, sizeof(LUID));
    info.luidValid = true;
    return info;
}

// The limits d3d12.h defines are each the macro whose name says what it limits.
Limits limitsOf(const D3D12_FEATURE_DATA_D3D12_OPTIONS6& options6) {
    Limits limits;
    limits.maxTexture1d = D3D12_REQ_TEXTURE1D_U_DIMENSION;
    limits.maxTexture2d = D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION;
    limits.maxTexture3d = D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION;
    limits.maxTextureCube = D3D12_REQ_TEXTURECUBE_DIMENSION;
    limits.maxTextureLayers = D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION;
    limits.maxPushConstantsSize = kPushConstantsSize;
    limits.maxColorAttachments = D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT;
    limits.maxViewports = kViewportCount;
    limits.maxSamplerAnisotropy = D3D12_REQ_MAXANISOTROPY;
    limits.maxSamplerLodBias = D3D12_MIP_LOD_BIAS_MAX;
    limits.minUniformBufferAlignment = D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;
    limits.copyOffsetAlignment = D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT;
    limits.copyRowPitchAlignment = D3D12_TEXTURE_DATA_PITCH_ALIGNMENT;
    // D3D12 answers sample counts per format only, in formatCaps, so the
    // device-wide counts stay 0.
    if (options6.VariableShadingRateTier >= D3D12_VARIABLE_SHADING_RATE_TIER_2) {
        // One size, for a square tile.
        const UINT tile = options6.ShadingRateImageTileSize;
        limits.minShadingRateTile = {tile, tile};
        limits.maxShadingRateTile = {tile, tile};
    }
    return limits;
}

// Resource binding tier 3, which the floor asks for, reads any descriptor in
// a full shader-visible heap from any stage. Each heap is as OPTIONS19 says
// when the device answers it, else as d3d12.h defines it for the tiers. The
// three resource tables are regions of one heap (createD3D12BindlessLayout),
// so each may take the whole heap, but together they take no more than it.
// BFR makes no uniform buffer table, so maxUniformBuffers stays 0.
BindlessLimits bindlessOf(const SamplerFeatures& sampler) {
    BindlessLimits bindless;
    const std::uint32_t views = sampler.maxViewHeapSize != 0 ? sampler.maxViewHeapSize
                                                             : D3D12_MAX_SHADER_VISIBLE_DESCRIPTOR_HEAP_SIZE_TIER_2;
    bindless.maxTextures = views;
    bindless.maxStorageTextures = views;
    bindless.maxStorageBuffers = views;
    bindless.maxSamplers = sampler.maxSamplerHeapSize != 0 ? sampler.maxSamplerHeapSize
                                                           : D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE;
    bindless.maxPerStage = views;
    return bindless;
}

ComputeLimits computeOf(ID3D12Device* device) {
    ComputeLimits compute;
    constexpr std::uint32_t kGroups = D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION;
    compute.maxWorkgroupCount = {kGroups, kGroups, kGroups};
    compute.maxWorkgroupSize = {D3D12_CS_THREAD_GROUP_MAX_X, D3D12_CS_THREAD_GROUP_MAX_Y, D3D12_CS_THREAD_GROUP_MAX_Z};
    compute.maxWorkgroupInvocations = D3D12_CS_THREAD_GROUP_MAX_THREADS_PER_GROUP;
    D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1{};
    if (query(device, D3D12_FEATURE_D3D12_OPTIONS1, options1) && options1.WaveOps != FALSE) {
        compute.subgroupSize = options1.WaveLaneCountMin;
        compute.minSubgroupSize = options1.WaveLaneCountMin;
        compute.maxSubgroupSize = options1.WaveLaneCountMax;
    }
    return compute;
}

// What the first D3D12 options answer.
void addBasicFeatures(ID3D12Device* device, Features& features) {
    D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
    if (query(device, D3D12_FEATURE_D3D12_OPTIONS, options)) {
        features.shaderFloat64 = options.DoublePrecisionFloatShaderOps != FALSE;
        features.logicOp = options.OutputMergerLogicOp != FALSE;
        features.conservativeRaster = options.ConservativeRasterizationTier !=
                                      D3D12_CONSERVATIVE_RASTERIZATION_TIER_NOT_SUPPORTED;
        const bool tiled = options.TiledResourcesTier != D3D12_TILED_RESOURCES_TIER_NOT_SUPPORTED;
        features.sparseBinding = tiled;
        features.sparseTextures = tiled;
        features.sparseBuffers = tiled;
    }
    D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1{};
    if (query(device, D3D12_FEATURE_D3D12_OPTIONS1, options1)) {
        features.shaderInt64 = options1.Int64ShaderOps != FALSE;
    }
    D3D12_FEATURE_DATA_D3D12_OPTIONS2 options2{};
    if (query(device, D3D12_FEATURE_D3D12_OPTIONS2, options2)) {
        features.depthBounds = options2.DepthBoundsTestSupported != FALSE;
    }
    D3D12_FEATURE_DATA_D3D12_OPTIONS3 options3{};
    if (query(device, D3D12_FEATURE_D3D12_OPTIONS3, options3)) {
        features.multiview = options3.ViewInstancingTier != D3D12_VIEW_INSTANCING_TIER_NOT_SUPPORTED;
        features.barycentrics = options3.BarycentricsSupported != FALSE;
    }
    D3D12_FEATURE_DATA_D3D12_OPTIONS4 options4{};
    if (query(device, D3D12_FEATURE_D3D12_OPTIONS4, options4)) {
        // One switch for 16-bit floats and integers in shaders.
        features.shaderFloat16 = options4.Native16BitShaderOpsSupported != FALSE;
        features.shaderInt16 = options4.Native16BitShaderOpsSupported != FALSE;
    }
}

// What the options for ray tracing, shading rate, mesh shaders and 64-bit
// atomics answer.
void addAdvancedFeatures(ID3D12Device* device, Features& features) {
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 options5{};
    if (query(device, D3D12_FEATURE_D3D12_OPTIONS5, options5)) {
        features.rayTracingPipeline = options5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_0;
        features.accelerationStructure = options5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_0;
        features.rayQuery = options5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1;
    }
    D3D12_FEATURE_DATA_D3D12_OPTIONS6 options6{};
    if (query(device, D3D12_FEATURE_D3D12_OPTIONS6, options6)) {
        features.shadingRate = options6.VariableShadingRateTier >= D3D12_VARIABLE_SHADING_RATE_TIER_1;
        features.shadingRateTexture = options6.VariableShadingRateTier >= D3D12_VARIABLE_SHADING_RATE_TIER_2;
    }
    D3D12_FEATURE_DATA_D3D12_OPTIONS7 options7{};
    if (query(device, D3D12_FEATURE_D3D12_OPTIONS7, options7)) {
        // A task shader is an amplification shader.
        features.meshShader = options7.MeshShaderTier >= D3D12_MESH_SHADER_TIER_1;
        features.taskShader = options7.MeshShaderTier >= D3D12_MESH_SHADER_TIER_1;
    }
    D3D12_FEATURE_DATA_D3D12_OPTIONS9 options9{};
    if (query(device, D3D12_FEATURE_D3D12_OPTIONS9, options9)) {
        features.atomicInt64Texture = options9.AtomicInt64OnTypedResourceSupported != FALSE;
    }
}

// What samplers may be. D3D12 has no anisotropic filter with a nearest min or
// mag filter (d3d12.h's D3D12_FILTER), and no LOD averaging. Where no query
// answers, Microsoft's documentation does:
// - learn.microsoft.com/windows/win32/direct3d12/hardware-feature-levels:
//   anisotropy (16) and Mirror Once at every feature level; Min/Max Filters
//   "Yes | Yes | Optional | No" for 12_1, 12_0, 11_1, 11_0, and no D3D12
//   query for the optional one.
// - D3D12_SAMPLER_DESC: BorderColor is any RGBA in 0 to 1, MipLODBias any
//   offset in D3D12_MIP_LOD_BIAS_MIN to MAX.
void addSamplerFeatures(ID3D12Device* device, const SamplerFeatures& sampler, Features& features) {
    features.samplerAnisotropy = true;
    features.samplerMinmax = featureLevelOf(device) >= D3D_FEATURE_LEVEL_12_0;
    features.samplerMirrorClampToEdge = true;
    features.samplerCustomBorderColor = true;
    features.samplerLodBias = true;
    features.samplerAnisotropyNearestMip = sampler.anisotropyNearestMip;
    features.samplerIntegerBorderColor = sampler.integerBorderColor;
    features.samplerUnnormalizedCoordinates = sampler.unnormalizedCoordinates;
}

// What every device BFR opens has, which no D3D12 query answers: d3d12.h
// has no caps bit for D3D12_BLEND_DESC.IndependentBlendEnable,
// D3D12_FILL_MODE_WIREFRAME or D3D12_RASTERIZER_DESC.DepthClipEnable, nor for
// the BC formats as a set. Devices are made at feature level 11_0 or above
// (D3D12CreateDevice in d3d12_instance.cpp), where the Direct3D functional
// spec requires all four; no header states that requirement.
void addFloorFeatures(Features& features) {
    features.independentBlend = true;
    features.wireframe = true;
    features.depthClamp = true;
    features.compressionBc = true;
}

Features featuresOf(ID3D12Device* device, const SamplerFeatures& sampler) {
    Features features;
    addBasicFeatures(device, features);
    addAdvancedFeatures(device, features);
    addSamplerFeatures(device, sampler, features);
    addFloorFeatures(features);
    return features;
}

// The heaps, in Memory's order: the adapter's own memory, D3D12's pool L1,
// and the machine's memory it may use, pool L0.
constexpr std::uint32_t kDedicatedHeap = 0;
constexpr std::uint32_t kSharedHeap = 1;

// The heap types buffers are placed in (d3d12_buffer.cpp placementFor), in
// order, as GetCustomHeapProperties says this adapter backs each: its pool
// names the heap, and its CPU page property whether and how the CPU maps it.
// A UMA adapter's one pool is DXGI's local segment group, the first heap's
// budget, which D3D12MA charges every allocation on it to
// (HeapPropertiesToMemorySegmentGroup). D3D12 has no call to flush or
// invalidate mapped memory: Map and Unmap are ID3D12Resource's only mapping
// calls and neither flushes, so what the CPU maps is coherent.
constexpr std::array kBufferHeapTypes{D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_TYPE_UPLOAD, D3D12_HEAP_TYPE_READBACK};

MemoryType memoryTypeOf(ID3D12Device* device, D3D12_HEAP_TYPE type, const Memory& memory) {
    // The adapter's one GPU, as a single-GPU node mask.
    constexpr UINT kNodeMask = 0;
    const D3D12_HEAP_PROPERTIES properties = device->GetCustomHeapProperties(kNodeMask, type);
    const bool local = memory.unified || properties.MemoryPoolPreference == D3D12_MEMORY_POOL_L1;
    const std::uint32_t heap = local ? kDedicatedHeap : kSharedHeap;
    const bool mapped = properties.CPUPageProperty != D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE;
    return {.heap = heap,
            .deviceLocal = memory.heaps[heap].deviceLocal,
            .hostVisible = mapped,
            .hostCoherent = mapped,
            .hostCached = properties.CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_WRITE_BACK};
}

// The sizes of the adapter's own memory and of the machine's memory it may
// use, as DXGI describes them, whether the two are the same memory, and the
// memory types buffers are made in. A heap is host visible when a type in it
// is.
Memory memoryOf(const DXGI_ADAPTER_DESC1& desc, ID3D12Device* device) {
    Memory memory;
    memory.unified = unifiedMemory(device);
    memory.heaps.push_back({.size = desc.DedicatedVideoMemory, .deviceLocal = true});
    memory.heaps.push_back({.size = desc.SharedSystemMemory});
    for (const D3D12_HEAP_TYPE type : kBufferHeapTypes) {
        const MemoryType made = memoryTypeOf(device, type, memory);
        memory.heaps[made.heap].hostVisible = memory.heaps[made.heap].hostVisible || made.hostVisible;
        memory.types.push_back(made);
    }
    return memory;
}

// The levels device says a queue of type takes, as
// D3D12_FEATURE_COMMAND_QUEUE_PRIORITY answers for each priority D3D12 has.
QueueLevels levelsOf(ID3D12Device* device, D3D12_COMMAND_LIST_TYPE type) {
    const auto takes = [device, type](D3D12_COMMAND_QUEUE_PRIORITY priority) {
        D3D12_FEATURE_DATA_COMMAND_QUEUE_PRIORITY data{.CommandListType = type,
                                                       .Priority = static_cast<UINT>(priority),
                                                       .PriorityForTypeIsSupported = FALSE};
        return query(device, D3D12_FEATURE_COMMAND_QUEUE_PRIORITY, data) && data.PriorityForTypeIsSupported != FALSE;
    };
    return {.normal = takes(D3D12_COMMAND_QUEUE_PRIORITY_NORMAL),
            .low = false,
            .high = takes(D3D12_COMMAND_QUEUE_PRIORITY_HIGH),
            .realtime = takes(D3D12_COMMAND_QUEUE_PRIORITY_GLOBAL_REALTIME)};
}

// D3D12 has no count of queues: CreateCommandQueue makes as many of each type
// as asked. D3D12Device::open makes as many graphics, compute and copy queues
// as DeviceDesc asks, so any count it holds is taken. It doesn't rank a
// device's queues.
Queues queuesOf(ID3D12Device* device) {
    return {.graphics = 0,
            .compute = 0,
            .copy = 0,
            .ranks = 0,
            .graphicsLevels = levelsOf(device, D3D12_COMMAND_LIST_TYPE_DIRECT),
            .computeLevels = levelsOf(device, D3D12_COMMAND_LIST_TYPE_COMPUTE),
            .copyLevels = levelsOf(device, D3D12_COMMAND_LIST_TYPE_COPY)};
}

} // namespace

SamplerFeatures samplerFeaturesOf(ID3D12Device* device) {
    SamplerFeatures sampler;
    D3D12_FEATURE_DATA_D3D12_OPTIONS19 options19{};
    if (query(device, D3D12_FEATURE_D3D12_OPTIONS19, options19)) {
        sampler.anisotropyNearestMip = options19.AnisoFilterWithPointMipSupported != FALSE;
        sampler.maxSamplerHeapSize = options19.MaxSamplerDescriptorHeapSize;
        sampler.maxViewHeapSize = options19.MaxViewDescriptorHeapSize;
    }
    ComPtr<ID3D12Device11> device11;
    sampler.device11 = SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device11)));
    D3D12_FEATURE_DATA_D3D12_OPTIONS17 options17{};
    sampler.unnormalizedCoordinates = sampler.device11 && query(device, D3D12_FEATURE_D3D12_OPTIONS17, options17) &&
                                      options17.NonNormalizedCoordinateSamplersSupported != FALSE;
    D3D12_FEATURE_DATA_D3D12_OPTIONS14 options14{};
    sampler.integerBorderColor = sampler.device11 && query(device, D3D12_FEATURE_D3D12_OPTIONS14, options14) &&
                                 options14.AdvancedTextureOpsSupported != FALSE;
    return sampler;
}

std::string belowFloor(ID3D12Device* device) {
    std::string reasons;
    const auto add = [&reasons](const char* reason) {
        if (!reasons.empty()) {
            reasons += ", ";
        }
        reasons += reason;
    };
    D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
    if (!query(device, D3D12_FEATURE_D3D12_OPTIONS, options) ||
        options.ResourceBindingTier != D3D12_RESOURCE_BINDING_TIER_3) {
        add("not resource binding tier 3");
    }
    D3D12_FEATURE_DATA_SHADER_MODEL model{.HighestShaderModel = D3D_SHADER_MODEL_6_6};
    // A runtime that doesn't know shader model 6.6 refuses to say.
    if (!query(device, D3D12_FEATURE_SHADER_MODEL, model) || model.HighestShaderModel < D3D_SHADER_MODEL_6_6) {
        add("not shader model 6.6");
    }
    D3D12_FEATURE_DATA_D3D12_OPTIONS12 options12{};
    if (!query(device, D3D12_FEATURE_D3D12_OPTIONS12, options12) || options12.EnhancedBarriersSupported == FALSE) {
        add("no enhanced barriers");
    }
    return reasons;
}

AdapterType typeOf(const DXGI_ADAPTER_DESC1& desc, ID3D12Device* device) {
    if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) {
        return AdapterType::Cpu;
    }
    // D3D12's own answer first: unified memory is an integrated GPU, whatever
    // memory DXGI says is dedicated to it.
    if (unifiedMemory(device)) {
        return AdapterType::Integrated;
    }
    return desc.DedicatedVideoMemory > 0 ? AdapterType::Discrete : AdapterType::Integrated;
}

std::string nameOf(const DXGI_ADAPTER_DESC1& desc) {
    const int bytes = WideCharToMultiByte(CP_UTF8, 0, static_cast<const WCHAR*>(desc.Description), -1, nullptr, 0,
                                          nullptr, nullptr);
    if (bytes <= 1) {
        return {};
    }
    std::string name(static_cast<std::size_t>(bytes), '\0');
    WideCharToMultiByte(CP_UTF8, 0, static_cast<const WCHAR*>(desc.Description), -1, name.data(), bytes, nullptr,
                        nullptr);
    // The count includes the terminating null.
    name.resize(static_cast<std::size_t>(bytes) - 1);
    return name;
}

namespace {

// D3D12 writes a timestamp with EndQuery, which takes no stage or
// granularity and marks the end of the work before it, in rendering and out.
// Direct and compute queues write them; a copy queue where OPTIONS3 says so,
// into a COPY_QUEUE_TIMESTAMP heap (d3d12.h, D3D12_QUERY_HEAP_TYPE). Each
// resolves as a UINT64. GetClockCalibration reads QueryPerformanceCounter.
Timestamps timestampsOf(ID3D12Device* device) {
    D3D12_FEATURE_DATA_D3D12_OPTIONS3 options3{};
    const bool copy = query(device, D3D12_FEATURE_D3D12_OPTIONS3, options3) &&
                      options3.CopyQueueTimestampQueriesSupported != FALSE;
    constexpr TimestampStages kEnd{.end = true};
    return {.graphics = true,
            .compute = true,
            .copy = copy,
            .outsideRendering = kEnd,
            .insideRendering = kEnd,
            .granularity = false,
            .entryBytes = sizeof(UINT64),
            .hostClocks = {.performanceCounter = true}};
}

} // namespace

Caps readCaps(const DXGI_ADAPTER_DESC1& desc, ID3D12Device* device) {
    Caps caps;
    D3D12_FEATURE_DATA_D3D12_OPTIONS6 options6{};
    static_cast<void>(query(device, D3D12_FEATURE_D3D12_OPTIONS6, options6));
    caps.info = infoOf(desc, device);
    caps.limits = limitsOf(options6);
    const SamplerFeatures sampler = samplerFeaturesOf(device);
    caps.bindless = bindlessOf(sampler);
    // D3D12 states no limit on samplers with a custom border colour, so any
    // sampler the table holds may have one.
    caps.limits.maxCustomBorderColorSamplers = caps.bindless.maxSamplers;
    // A sampler is a descriptor in the sampler table's heap, so the heap holds
    // every sampler the device has.
    caps.limits.maxSamplerObjects = caps.bindless.maxSamplers;
    caps.compute = computeOf(device);
    caps.features = featuresOf(device, sampler);
    caps.memory = memoryOf(desc, device);
    caps.queues = queuesOf(device);
    caps.timestamps = timestampsOf(device);
    return caps;
}

} // namespace bfr::rhi::d3d12
