#include "metal/metal_caps.h"

#include "bfr/rhi/caps.h"
#include "metal/metal_pipeline.h"
#include "metal/metal_util.h"

#include <Foundation/NSError.hpp>
#include <Metal/MTL4Counters.hpp>
#include <Metal/MTLDevice.hpp>
#include <Metal/MTLTypes.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

namespace bfr::rhi::metal {

namespace {

constexpr std::uint32_t kMetal4 = 4;

std::uint32_t clampTo32(std::uint64_t value) {
    return static_cast<std::uint32_t>(std::min<std::uint64_t>(value, std::numeric_limits<std::uint32_t>::max()));
}

// Metal defines no most sample count, so every power of two up to 64 is
// asked, as Godot asks (drivers/metal/metal_device_properties.cpp).
constexpr std::uint32_t kMostSampleCountAsked = 64;

// Metal gives no PCI IDs, driver name or version, UUID or LUID. It says
// whether the device runs Metal 4.
AdapterInfo infoOf(MTL::Device* device) {
    AdapterInfo info;
    if (device->supportsFamily(MTL::GPUFamilyMetal4)) {
        info.apiMajor = kMetal4;
    }
    return info;
}

// MTLSamplerDescriptor's documentation: maxAnisotropy "between 1 and 16,
// inclusive"; lodBias "the range is [-16.0, 15.999]".
constexpr float kMaxSamplerAnisotropy = 16;
constexpr float kMaxSamplerLodBias = 15.999F;

// Metal has no query for it; the Metal feature set tables give 8 color render
// targets per render pass descriptor.
constexpr std::uint32_t kMaxColorAttachments = 8;

// setViewport sets one viewport on the encoder, so one is what BFR takes.
constexpr std::uint32_t kMaxViewports = 1;

Limits limitsOf(MTL::Device* device) {
    Limits limits;
    limits.maxBufferSize = device->maxBufferLength();
    limits.maxPushConstantsSize = kPushConstantsSize;
    limits.maxColorAttachments = kMaxColorAttachments;
    limits.maxViewports = kMaxViewports;
    limits.colorSampleCounts = sampleCountsOf(device);
    limits.maxSamplerAnisotropy = kMaxSamplerAnisotropy;
    if (device->supportsFamily(MTL::GPUFamilyApple10)) {
        limits.maxSamplerLodBias = kMaxSamplerLodBias;
    }
    return limits;
}

ComputeLimits computeOf(MTL::Device* device) {
    ComputeLimits compute;
    const MTL::Size threads = device->maxThreadsPerThreadgroup();
    compute.maxWorkgroupSize = {clampTo32(threads.width), clampTo32(threads.height), clampTo32(threads.depth)};
    compute.maxSharedMemory = clampTo32(device->maxThreadgroupMemoryLength());
    return compute;
}

// Metal says how many samplers an argument buffer holds. For textures and
// buffers it has no call; the Metal Feature Set Tables give 500,000 each for
// the Apple7 and later families Metal 4 runs on, with tier 2 argument
// buffers. BFR builds no uniform buffer table, so it holds none.
constexpr std::uint32_t kArgumentBufferResources = 500000;

BindlessLimits bindlessOf(MTL::Device* device) {
    BindlessLimits bindless;
    bindless.maxTextures = kArgumentBufferResources;
    bindless.maxStorageTextures = kArgumentBufferResources;
    bindless.maxStorageBuffers = kArgumentBufferResources;
    bindless.maxSamplers = clampTo32(device->maxArgumentBufferSamplerCount());
    bindless.maxPerStage = kArgumentBufferResources;
    return bindless;
}

// What samplers may be, from the Metal Feature Set Tables (Oct 2026) and
// MTLSamplerDescriptor's documentation, for the Apple7 and later families
// Metal 4 runs on: anisotropy from Apple2; minimum and maximum reduction and
// LOD bias from Apple10; mirror clamp to edge, unnormalized coordinates and
// LOD averaging with no family named. Border colours are the three presets
// alone, and nothing says how they apply to integer formats.
void addSamplerFeatures(MTL::Device* device, Features& features) {
    const bool apple10 = device->supportsFamily(MTL::GPUFamilyApple10);
    features.samplerAnisotropy = true;
    features.samplerMinmax = apple10;
    features.samplerLodBias = apple10;
    features.samplerMirrorClampToEdge = true;
    features.samplerUnnormalizedCoordinates = true;
    features.samplerLodAverage = true;
}

// Which compressed pixel formats the device samples. Metal answers BC; for the
// rest the Metal Feature Set Tables name the families: ETC2, EAC and ASTC LDR
// from Apple2, ASTC HDR from Apple6 (Godot's Metal driver, pixel_formats.cpp,
// gates HDR on Apple6 too).
void addCompressionFeatures(MTL::Device* device, Features& features) {
    features.compressionBc = device->supportsBCTextureCompression();
    features.compressionEtc2 = device->supportsFamily(MTL::GPUFamilyApple2);
    features.compressionAstc = device->supportsFamily(MTL::GPUFamilyApple2);
    features.compressionAstcHdr = device->supportsFamily(MTL::GPUFamilyApple6);
}

// Render state Metal has no call for: a triangle fill mode and depth clip
// mode on every render encoder, and blending set per color attachment, on the
// Apple7 and later families Metal 4 runs on.
void addRenderFeatures(Features& features) {
    features.wireframe = true;
    features.depthClamp = true;
    features.independentBlend = true;
}

Features featuresOf(MTL::Device* device) {
    Features features;
    features.rayQuery = device->supportsRaytracing();
    features.accelerationStructure = device->supportsRaytracing();
    features.barycentrics = device->supportsShaderBarycentricCoordinates();
    features.atomicInt64 = has64BitAtomics(device);
    features.atomicInt64Texture = has64BitAtomics(device);
    addCompressionFeatures(device, features);
    addRenderFeatures(features);
    addSamplerFeatures(device, features);
    return features;
}

// BFR opens as many graphics, compute and copy queues as DeviceDesc asks,
// each an MTL4CommandQueue. Metal has no call for how many a
// device has and states no limit, so createDevice takes any count. Metal has
// no queue priority, so there are no ranks or levels.
Queues queuesOf() {
    return {.graphics = 0,
            .compute = 0,
            .copy = 0,
            .graphicsTimestampBits = 0,
            .computeTimestampBits = 0,
            .copyTimestampBits = 0,
            .ranks = 0,
            .graphicsLevels = {},
            .computeLevels = {},
            .copyLevels = {}};
}

// The one pool of memory Metal has for a device to use.
Memory memoryOf(MTL::Device* device) {
    Memory memory;
    memory.heaps.push_back({.size = device->recommendedMaxWorkingSetSize()});
    memory.unified = device->hasUnifiedMemory();
    return memory;
}

} // namespace

std::uint32_t sampleCountsOf(MTL::Device* device) {
    std::uint32_t counts = 0;
    for (std::uint32_t count = 1; count <= kMostSampleCountAsked; count *= 2) {
        if (device->supportsTextureSampleCount(count)) {
            counts |= count;
        }
    }
    return counts;
}

// Godot's pixel_formats.cpp:938-943 and The Forge's MetalRenderer.mm:2229-2235.
bool has64BitAtomics(MTL::Device* device) {
    return device->supportsFamily(MTL::GPUFamilyApple9) ||
           (device->supportsFamily(MTL::GPUFamilyApple8) && device->supportsFamily(MTL::GPUFamilyMac2));
}

std::string belowFloor(MTL::Device* device) {
    std::string reasons;
    const auto add = [&reasons](const char* reason) {
        if (!reasons.empty()) {
            reasons += ", ";
        }
        reasons += reason;
    };
    if (!device->supportsFamily(MTL::GPUFamilyMetal4)) {
        add("no Metal 4");
    }
    return reasons;
}

namespace {

// Metal 4 writes timestamps into a counter heap: between encoders through the
// command buffer, which takes no stage and marks the end of the work before
// it, and in rendering through the render encoder, after its vertex or
// fragment stage and at a granularity. Any queue writes them, where the
// device makes a timestamp heap. sampleTimestamps (MTLDevice.hpp) reads a CPU
// clock that metal-cpp doesn't name, so no host clock is listed.
Timestamps timestampsOf(MTL::Device* device) {
    const AutoreleasePool pool;
    const Ref<MTL4::CounterHeapDescriptor> descriptor = adopt(MTL4::CounterHeapDescriptor::alloc()->init());
    descriptor->setType(MTL4::CounterHeapTypeTimestamp);
    descriptor->setCount(1);
    NS::Error* error = nullptr;
    const Ref<MTL4::CounterHeap> heap = adopt(device->newCounterHeap(descriptor.get(), &error));
    if (!heap) {
        return {};
    }
    return {.graphics = true,
            .compute = true,
            .copy = true,
            .outsideRendering = {.end = true},
            .insideRendering = {.vertex = true, .fragment = true},
            .granularity = true,
            .entryBytes = clampTo32(device->sizeOfCounterHeapEntry(MTL4::CounterHeapTypeTimestamp)),
            .hostClocks = {}};
}

} // namespace

Caps readCaps(MTL::Device* device) {
    Caps caps;
    caps.info = infoOf(device);
    caps.limits = limitsOf(device);
    caps.compute = computeOf(device);
    caps.bindless = bindlessOf(device);
    caps.features = featuresOf(device);
    caps.memory = memoryOf(device);
    caps.queues = queuesOf();
    caps.timestamps = timestampsOf(device);
    return caps;
}

} // namespace bfr::rhi::metal
