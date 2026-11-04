#include "metal/metal_caps.h"

#include "bfr/rhi/caps.h"

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

// Each sample count the device renders at, as bits whose values are the counts.
std::uint32_t sampleCountsOf(MTL::Device* device) {
    std::uint32_t counts = 1;
    for (const std::uint32_t count : {2U, 4U, 8U}) {
        if (device->supportsTextureSampleCount(count)) {
            counts |= count;
        }
    }
    return counts;
}

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

Limits limitsOf(MTL::Device* device) {
    Limits limits;
    limits.maxBufferSize = device->maxBufferLength();
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
// the Apple7 and later families the floor asks for, with tier 2 argument
// buffers.
constexpr std::uint32_t kArgumentBufferResources = 500000;

BindlessLimits bindlessOf(MTL::Device* device) {
    BindlessLimits bindless;
    bindless.maxTextures = kArgumentBufferResources;
    bindless.maxStorageTextures = kArgumentBufferResources;
    bindless.maxStorageBuffers = kArgumentBufferResources;
    bindless.maxUniformBuffers = kArgumentBufferResources;
    bindless.maxSamplers = clampTo32(device->maxArgumentBufferSamplerCount());
    bindless.maxPerStage = kArgumentBufferResources;
    return bindless;
}

// What samplers may be, from the Metal Feature Set Tables (Oct 2026) and
// MTLSamplerDescriptor's documentation, for the Apple7 and later families the
// floor asks for: anisotropy from Apple2; minimum and maximum reduction and
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

Features featuresOf(MTL::Device* device) {
    Features features;
    features.rayQuery = device->supportsRaytracing();
    features.accelerationStructure = device->supportsRaytracing();
    features.barycentrics = device->supportsShaderBarycentricCoordinates();
    features.compressionBc = device->supportsBCTextureCompression();
    addSamplerFeatures(device, features);
    return features;
}

// The one pool of memory Metal has for a device to use.
Memory memoryOf(MTL::Device* device) {
    Memory memory;
    memory.heaps.push_back({.size = device->recommendedMaxWorkingSetSize()});
    memory.unified = device->hasUnifiedMemory();
    return memory;
}

} // namespace

std::string belowFloor(MTL::Device* device) {
    std::string reasons;
    const auto add = [&reasons](const char* reason) {
        if (!reasons.empty()) {
            reasons += ", ";
        }
        reasons += reason;
    };
    if (!device->supportsFamily(MTL::GPUFamilyApple7)) {
        add("not Apple GPU family 7 or later");
    }
    if (!device->supportsFamily(MTL::GPUFamilyMetal4)) {
        add("no Metal 4");
    }
    return reasons;
}

Caps readCaps(MTL::Device* device) {
    Caps caps;
    caps.info = infoOf(device);
    caps.limits = limitsOf(device);
    caps.compute = computeOf(device);
    caps.bindless = bindlessOf(device);
    caps.features = featuresOf(device);
    caps.memory = memoryOf(device);
    return caps;
}

} // namespace bfr::rhi::metal
