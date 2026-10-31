#include "d3d12/d3d12_caps.h"

#include "bfr/rhi/caps.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/instance.h"
#include "d3d12/d3d12_formats.h"

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

constexpr double kNanosecondsPerSecond = 1e9;
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

// Nanoseconds a timestamp tick lasts, 0 when the device won't say. The
// frequency belongs to a queue, so one is made for the asking.
float timestampPeriodOf(ID3D12Device* device) {
    const D3D12_COMMAND_QUEUE_DESC desc{.Type = D3D12_COMMAND_LIST_TYPE_DIRECT,
                                        .Priority = 0,
                                        .Flags = D3D12_COMMAND_QUEUE_FLAG_NONE,
                                        .NodeMask = 0};
    ComPtr<ID3D12CommandQueue> queue;
    UINT64 frequency = 0;
    if (FAILED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue))) ||
        FAILED(queue->GetTimestampFrequency(&frequency)) || frequency == 0) {
        return 0;
    }
    return static_cast<float>(kNanosecondsPerSecond / static_cast<double>(frequency));
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

// The sample counts a format renders at, as caps lists them.
std::uint32_t sampleCountsOf(ID3D12Device* device, Format format) {
    const auto caps = formatCapsFor(device, format);
    return caps ? caps->sampleCounts : 0;
}

// The limits d3d12.h defines are each the macro whose name says what it limits.
Limits limitsOf(ID3D12Device* device, const D3D12_FEATURE_DATA_D3D12_OPTIONS6& options6) {
    Limits limits;
    limits.maxTexture1d = D3D12_REQ_TEXTURE1D_U_DIMENSION;
    limits.maxTexture2d = D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION;
    limits.maxTexture3d = D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION;
    limits.maxTextureCube = D3D12_REQ_TEXTURECUBE_DIMENSION;
    limits.maxTextureLayers = D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION;
    limits.maxSamplerObjects = D3D12_REQ_SAMPLER_OBJECT_COUNT_PER_DEVICE;
    limits.maxColorAttachments = D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT;
    limits.maxViewports = D3D12_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
    limits.maxSamplerAnisotropy = D3D12_REQ_MAXANISOTROPY;
    limits.maxSamplerLodBias = D3D12_MIP_LOD_BIAS_MAX;
    limits.minUniformBufferAlignment = D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;
    limits.copyOffsetAlignment = D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT;
    limits.copyRowPitchAlignment = D3D12_TEXTURE_DATA_PITCH_ALIGNMENT;
    limits.colorSampleCounts = sampleCountsOf(device, Format::Rgba8Unorm);
    limits.integerSampleCounts = sampleCountsOf(device, Format::Rgba8Uint);
    limits.depthSampleCounts = sampleCountsOf(device, Format::D32Float);
    limits.stencilSampleCounts = sampleCountsOf(device, Format::D24UnormS8Uint);
    if (options6.VariableShadingRateTier >= D3D12_VARIABLE_SHADING_RATE_TIER_2) {
        limits.minShadingRateTile = options6.ShadingRateImageTileSize;
        limits.maxShadingRateTile = options6.ShadingRateImageTileSize;
    }
    limits.timestampPeriod = timestampPeriodOf(device);
    return limits;
}

BindlessLimits bindlessOf() {
    BindlessLimits bindless;
    bindless.maxSamplers = D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE;
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

Features featuresOf(ID3D12Device* device) {
    Features features;
    addBasicFeatures(device, features);
    addAdvancedFeatures(device, features);
    return features;
}

// The sizes of the adapter's own memory and of the machine's memory it may
// use, as DXGI describes them, and whether the two are the same memory.
Memory memoryOf(const DXGI_ADAPTER_DESC1& desc, ID3D12Device* device) {
    Memory memory;
    D3D12_FEATURE_DATA_ARCHITECTURE architecture{.NodeIndex = 0,
                                                 .TileBasedRenderer = FALSE,
                                                 .UMA = FALSE,
                                                 .CacheCoherentUMA = FALSE};
    memory.unified = query(device, D3D12_FEATURE_ARCHITECTURE, architecture) && architecture.UMA != FALSE;
    memory.heaps.push_back({.size = desc.DedicatedVideoMemory, .deviceLocal = true});
    memory.heaps.push_back({.size = desc.SharedSystemMemory, .hostVisible = true});
    return memory;
}

} // namespace

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

AdapterType typeOf(const DXGI_ADAPTER_DESC1& desc) {
    if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) {
        return AdapterType::Cpu;
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

Caps readCaps(const DXGI_ADAPTER_DESC1& desc, ID3D12Device* device) {
    Caps caps;
    D3D12_FEATURE_DATA_D3D12_OPTIONS6 options6{};
    static_cast<void>(query(device, D3D12_FEATURE_D3D12_OPTIONS6, options6));
    caps.info = infoOf(desc, device);
    caps.limits = limitsOf(device, options6);
    caps.bindless = bindlessOf();
    caps.compute = computeOf(device);
    caps.features = featuresOf(device);
    caps.memory = memoryOf(desc, device);
    return caps;
}

} // namespace bfr::rhi::d3d12
