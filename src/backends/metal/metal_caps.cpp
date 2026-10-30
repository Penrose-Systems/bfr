#include "metal/metal_caps.h"

#include "bfr/rhi/caps.h"

#include <Metal/MTLDevice.hpp>
#include <Metal/MTLTypes.hpp>
#include <TargetConditionals.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

namespace bfr::rhi::metal {

namespace {

// The values below that are not asked of the device are the ones Godot's
// drivers/metal/metal_device_properties.cpp (init_limits, init_features) lists.
constexpr std::uint32_t kMaxTexture2d = 16384;
constexpr std::uint32_t kMaxTexture3d = 2048;
constexpr std::uint32_t kMaxTextureLayers = 2048;
constexpr std::uint32_t kMaxColorAttachments = 8;
constexpr std::uint32_t kMaxViewports = 16;
constexpr std::uint32_t kThreadgroupMemory = 32768;
constexpr std::uint32_t kMinSubgroupSize = 4;
constexpr std::uint32_t kMaxSubgroupSize = 32;
// 16 on a Mac, 64 elsewhere.
constexpr std::uint64_t kUniformAlignment = TARGET_OS_OSX != 0 ? 16 : 64;

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

// Metal gives no PCI IDs, driver name or version, UUID or LUID. The floor is
// Metal 4.
AdapterInfo infoOf() {
    AdapterInfo info;
    info.apiMajor = 4;
    return info;
}

Limits limitsOf(MTL::Device* device) {
    Limits limits;
    const std::uint32_t counts = sampleCountsOf(device);
    limits.maxTexture1d = kMaxTexture2d;
    limits.maxTexture2d = kMaxTexture2d;
    limits.maxTexture3d = kMaxTexture3d;
    limits.maxTextureCube = kMaxTexture2d;
    limits.maxTextureLayers = kMaxTextureLayers;
    limits.maxBufferSize = device->maxBufferLength();
    limits.maxColorAttachments = kMaxColorAttachments;
    limits.maxRenderWidth = kMaxTexture2d;
    limits.maxRenderHeight = kMaxTexture2d;
    limits.colorSampleCounts = counts;
    limits.integerSampleCounts = counts;
    limits.depthSampleCounts = counts;
    limits.stencilSampleCounts = counts;
    limits.maxViewports = kMaxViewports;
    limits.maxViewportWidth = kMaxTexture2d;
    limits.maxViewportHeight = kMaxTexture2d;
    limits.minUniformBufferAlignment = kUniformAlignment;
    return limits;
}

ComputeLimits computeOf(MTL::Device* device) {
    ComputeLimits compute;
    const MTL::Size threads = device->maxThreadsPerThreadgroup();
    constexpr std::uint32_t kUnlimited = std::numeric_limits<std::uint32_t>::max();
    compute.maxWorkgroupCount = {kUnlimited, kUnlimited, kUnlimited};
    compute.maxWorkgroupSize = {clampTo32(threads.width), clampTo32(threads.height), clampTo32(threads.depth)};
    compute.maxSharedMemory = kThreadgroupMemory;
    compute.subgroupSize = kMaxSubgroupSize;
    compute.minSubgroupSize = kMinSubgroupSize;
    compute.maxSubgroupSize = kMaxSubgroupSize;
    return compute;
}

Features featuresOf(MTL::Device* device) {
    Features features;
    // What the device answers.
    features.rayQuery = device->supportsRaytracing();
    features.accelerationStructure = device->supportsRaytracing();
    features.barycentrics = device->supportsShaderBarycentricCoordinates();
    features.compressionBc = device->supportsBCTextureCompression();
    // What Godot's init_features decides from the GPU family, and its format
    // table gives every Apple GPU for ETC2 and LDR ASTC.
    features.tessellation = device->supportsFamily(MTL::GPUFamilyApple3);
    features.compressionAstcHdr = device->supportsFamily(MTL::GPUFamilyApple6);
    features.compressionEtc2 = true;
    features.compressionAstc = true;
    features.atomicInt64Texture = device->supportsFamily(MTL::GPUFamilyApple9) ||
                                  (device->supportsFamily(MTL::GPUFamilyApple8) &&
                                   device->supportsFamily(MTL::GPUFamilyMac2));
    return features;
}

Memory memoryOf(MTL::Device* device) {
    Memory memory;
    const std::uint64_t working = device->recommendedMaxWorkingSetSize();
    const bool unified = device->hasUnifiedMemory();
    // Metal has one pool of memory for a device to use; a unified device
    // takes it from the machine's memory, which is all it has.
    const std::uint64_t size = unified
                                   ? std::max<std::uint64_t>(NS::ProcessInfo::processInfo()->physicalMemory(), working)
                                   : working;
    memory.heaps.push_back({.size = size, .deviceLocal = true, .hostVisible = unified});
    // Private and shared storage.
    memory.types.push_back({.heap = 0, .deviceLocal = true});
    memory.types.push_back({.heap = 0, .deviceLocal = unified, .hostVisible = true, .hostCoherent = true});
    memory.unified = unified;
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
    caps.info = infoOf();
    caps.limits = limitsOf(device);
    caps.compute = computeOf(device);
    caps.features = featuresOf(device);
    caps.memory = memoryOf(device);
    // BFR makes one queue, which runs graphics.
    caps.queues.graphics = 1;
    return caps;
}

} // namespace bfr::rhi::metal
