// Prints what each adapter is and can do, as BFR reports it.
//
//     bfr_caps             a summary of each adapter
//     bfr_caps --formats   a table of the formats each adapter supports
//     bfr_caps --window    what a swapchain for a headless window may be
//     bfr_caps --all       every field, as the API names it
#include "bfr.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <format>
#include <initializer_list>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kLabelWidth = 13;
constexpr std::size_t kLineWidth = 100;
constexpr std::string_view kIndent = "  ";
constexpr std::string_view kSeparator = " · ";
constexpr std::uint64_t kKiB = 1024;
constexpr std::uint32_t kMaxSamples = 64;

struct Options {
    bool formats = false;
    bool window = false;
    bool all = false;
};

void print(std::string_view text) {
    std::cout << text;
}

// label, then items joined by kSeparator, wrapped under the first item.
void row(std::string_view label, const std::vector<std::string>& items) {
    const std::string margin(kIndent.size() + kLabelWidth, ' ');
    std::string line = std::format("{}{:<{}}", kIndent, label, kLabelWidth);
    bool first = true;
    for (const std::string& item : items) {
        const std::string_view separator = first ? std::string_view{} : kSeparator;
        if (!first && line.size() + separator.size() + item.size() > kLineWidth) {
            print(line + "\n");
            line = margin + item;
        }
        else {
            line += std::string{separator} + item;
        }
        first = false;
    }
    print(line + "\n");
}

void row(std::string_view label, const std::string& text) {
    row(label, std::vector{text});
}

// 1048576 as "1,048,576".
std::string count(std::uint64_t value) {
    std::string digits = std::to_string(value);
    for (std::size_t at = digits.size(); at > 3; at -= 3) {
        digits.insert(at - 3, ",");
    }
    return digits;
}

// 1536 as "1.5 KiB".
std::string bytes(std::uint64_t value) {
    constexpr std::array kUnits{"B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"};
    auto size = static_cast<double>(value);
    std::size_t unit = 0;
    while (size >= static_cast<double>(kKiB) && unit + 1 < kUnits.size()) {
        size /= static_cast<double>(kKiB);
        ++unit;
    }
    std::string number = std::format("{:.1f}", size);
    if (number.ends_with(".0")) {
        number.resize(number.size() - 2);
    }
    return std::format("{} {}", number, kUnits.at(unit));
}

// A sample count mask, 0b1111, as "1, 2, 4, 8".
std::string samples(std::uint32_t mask) {
    std::string out;
    for (std::uint32_t samples = 1; samples <= kMaxSamples; samples <<= 1U) {
        if ((mask & samples) != 0) {
            out += (out.empty() ? "" : ", ") + std::to_string(samples);
        }
    }
    return out.empty() ? "none" : out;
}

std::string_view typeName(bfr::AdapterType type) {
    switch (type) {
        case bfr::AdapterType::Discrete:
            return "discrete GPU";
        case bfr::AdapterType::Integrated:
            return "integrated GPU";
        case bfr::AdapterType::Virtual:
            return "virtual GPU";
        case bfr::AdapterType::Cpu:
            return "CPU";
        case bfr::AdapterType::Other:
            break;
    }
    return "other";
}

std::string_view backendName(bfr::Backend backend) {
    switch (backend) {
        case bfr::Backend::Vulkan:
            return "Vulkan";
        case bfr::Backend::D3d12:
            return "D3D12";
        case bfr::Backend::Metal:
            return "Metal";
    }
    return "unknown";
}

void printHeading(const bfr::CapsInfo& info) {
    print(std::format("{} — {}, {} {}.{}.{}\n", info.name, typeName(info.type), backendName(info.backend),
                      info.apiMajor, info.apiMinor, info.apiPatch));
    const std::string driver = info.driverName.empty() ? "unnamed driver" : info.driverName;
    row("Driver",
        std::format("{} {} (vendor 0x{:04X}, device 0x{:04X})", driver, info.driverInfo, info.vendorId, info.deviceId));
}

// A heap by what it is: VRAM, VRAM the CPU can map, system memory, or the one
// memory a unified adapter shares.
std::string_view heapKind(const bfr::CapsMemory& memory, std::uint32_t heap) {
    const bfr::MemoryHeap& at = memory.heaps.at(heap);
    if (memory.unified) {
        return "shared";
    }
    if (at.deviceLocal) {
        return at.hostVisible ? "VRAM, CPU-visible" : "VRAM";
    }
    return "system";
}

void printMemory(const bfr::CapsMemory& memory) {
    std::vector<std::string> heaps;
    heaps.reserve(memory.heapCount);
    for (std::uint32_t heap = 0; heap < memory.heapCount; ++heap) {
        heaps.push_back(std::format("{} {}", bytes(memory.heaps.at(heap).size), heapKind(memory, heap)));
    }
    row("Memory", heaps);
}

void printLimits(const bfr::Caps& caps) {
    const bfr::CapsLimits& l = caps.limits;
    const bfr::CapsQueues& q = caps.queues;
    row("Queues",
        {std::format("{} graphics", q.graphics), std::format("{} compute", q.compute), std::format("{} copy", q.copy)});
    print("\n");
    row("Textures", {std::format("2D {}", l.maxTexture2d), std::format("3D {}", l.maxTexture3d),
                     std::format("cube {}", l.maxTextureCube), std::format("{} layers", l.maxTextureLayers)});
    row("Buffers",
        {std::format("up to {}", bytes(l.maxBufferSize)), std::format("{} per allocation", bytes(l.maxAllocationSize)),
         std::format("{} push constants", bytes(l.maxPushConstantsSize))});
    row("Rendering",
        {std::format("{} attachments", l.maxColorAttachments), std::format("MSAA {}", samples(l.colorSampleCounts)),
         std::format("{}x anisotropy", l.maxSamplerAnisotropy)});
    const bfr::CapsBindless& b = caps.bindless;
    row("Bindless", {std::format("{} textures", count(b.maxTextures)),
                     std::format("{} storage textures", count(b.maxStorageTextures)),
                     std::format("{} storage buffers", count(b.maxStorageBuffers)),
                     std::format("{} samplers", count(b.maxSamplers))});
}

void printShaders(const bfr::Caps& caps) {
    const bfr::CapsCompute& c = caps.compute;
    std::string subgroup = std::format("subgroup {}", c.subgroupSize);
    if (c.minSubgroupSize != c.maxSubgroupSize) {
        subgroup += std::format(" ({} to {})", c.minSubgroupSize, c.maxSubgroupSize);
    }
    row("Compute", {std::format("{} threads a group", c.maxWorkgroupInvocations),
                    std::format("{} shared", bytes(c.maxSharedMemory)), subgroup});
    const bfr::CapsMesh& m = caps.mesh;
    if (caps.features.meshShader) {
        row("Mesh", {std::format("{} vertices", m.maxMeshOutputVertices),
                     std::format("{} primitives", m.maxMeshOutputPrimitives),
                     std::format("{} task payload", bytes(m.maxTaskPayloadSize))});
    }
    else {
        row("Mesh", "none");
    }
    const bfr::CapsRayTracing& r = caps.rayTracing;
    if (caps.features.accelerationStructure) {
        std::vector<std::string> items{std::format("{} instances", count(r.maxInstances)),
                                       std::format("{} primitives", count(r.maxPrimitives))};
        if (caps.features.rayTracingPipeline) {
            items.push_back(std::format("recursion {}", r.maxRecursionDepth));
        }
        row("Ray tracing", items);
    }
    else {
        row("Ray tracing", "none");
    }
}

// The optional features, split by whether the adapter has them.
void printFeatures(const bfr::CapsFeatures& f) {
    struct Named {
        bool has;
        std::string_view name;
    };
    const std::initializer_list<Named> features{
        {.has = f.rayQuery, .name = "ray query"},
        {.has = f.rayTracingPipeline, .name = "ray tracing pipelines"},
        {.has = f.meshShader, .name = "mesh shaders"},
        {.has = f.taskShader, .name = "task shaders"},
        {.has = f.shadingRate, .name = "variable rate shading"},
        {.has = f.shadingRateTexture, .name = "shading rate textures"},
        {.has = f.shaderFloat16, .name = "fp16"},
        {.has = f.shaderFloat64, .name = "fp64"},
        {.has = f.shaderInt8, .name = "int8"},
        {.has = f.shaderInt16, .name = "int16"},
        {.has = f.shaderInt64, .name = "int64"},
        {.has = f.atomicInt64, .name = "64-bit atomics"},
        {.has = f.atomicInt64Texture, .name = "64-bit texture atomics"},
        {.has = f.atomicFloat32Add, .name = "float atomic add"},
        {.has = f.barycentrics, .name = "barycentrics"},
        {.has = f.conservativeRaster, .name = "conservative raster"},
        {.has = f.geometryShader, .name = "geometry shaders"},
        {.has = f.tessellation, .name = "tessellation"},
        {.has = f.multiview, .name = "multiview"},
        {.has = f.sampleRateShading, .name = "sample rate shading"},
        {.has = f.dualSourceBlend, .name = "dual source blend"},
        {.has = f.independentBlend, .name = "independent blend"},
        {.has = f.logicOp, .name = "logic ops"},
        {.has = f.depthClamp, .name = "depth clamp"},
        {.has = f.depthBounds, .name = "depth bounds"},
        {.has = f.wireframe, .name = "wireframe"},
        {.has = f.wideLines, .name = "wide lines"},
        {.has = f.samplerAnisotropy, .name = "anisotropy"},
        {.has = f.samplerMinmax, .name = "min/max sampling"},
        {.has = f.compressionBc, .name = "BC"},
        {.has = f.compressionEtc2, .name = "ETC2"},
        {.has = f.compressionAstc, .name = "ASTC"},
        {.has = f.compressionAstcHdr, .name = "ASTC HDR"},
        {.has = f.multiDrawIndirect, .name = "multi-draw indirect"},
        {.has = f.drawIndirectCount, .name = "indirect count"},
        {.has = f.drawIndirectFirstInstance, .name = "indirect first instance"},
        {.has = f.storageReadWithoutFormat, .name = "typeless storage reads"},
        {.has = f.storageWriteWithoutFormat, .name = "typeless storage writes"},
        {.has = f.timestampsOnAllQueues, .name = "timestamps on every queue"},
        {.has = f.pipelineStatistics, .name = "pipeline statistics"},
        {.has = f.occlusionQueryPrecise, .name = "precise occlusion"},
        {.has = f.sparseBinding, .name = "sparse binding"},
        {.has = f.sparseTextures, .name = "sparse textures"},
        {.has = f.sparseBuffers, .name = "sparse buffers"},
    };
    std::vector<std::string> has;
    std::vector<std::string> lacks;
    for (const Named& feature : features) {
        (feature.has ? has : lacks).emplace_back(feature.name);
    }
    row("Has", has.empty() ? std::vector<std::string>{"none of these"} : has);
    row("Lacks", lacks.empty() ? std::vector<std::string>{"nothing"} : lacks);
}

void printFormatCounts(const std::vector<bfr::FormatCaps>& formats) {
    std::size_t sampled = 0;
    std::size_t renderable = 0;
    std::size_t storage = 0;
    for (const bfr::FormatCaps& format : formats) {
        sampled += format.sampled ? 1 : 0;
        renderable += format.colorAttachment || format.depthStencil ? 1 : 0;
        storage += format.storage ? 1 : 0;
    }
    row("Formats", {std::format("{} sampled", sampled), std::format("{} renderable", renderable),
                    std::format("{} storage", storage)});
}

// One row per format the adapter supports, then the names of those it doesn't.
void printFormatTable(const std::vector<bfr::FormatCaps>& formats) {
    constexpr std::size_t kNameWidth = 26;
    const auto mark = [](bool has) {
        return has ? "✓" : "·";
    };
    print(std::format("\n{}{:<{}} sample linear minmax storage atomic color blend depth texel  MSAA\n", kIndent,
                      "Format", kNameWidth));
    std::vector<std::string> unsupported;
    for (const bfr::FormatCaps& f : formats) {
        if (!f.sampled && !f.storage && !f.colorAttachment && !f.depthStencil && !f.texelBuffer && !f.copySrc) {
            unsupported.emplace_back(bfr::name(f.format));
            continue;
        }
        print(std::format("{}{:<{}}   {}      {}      {}      {}       {}     {}     {}     {}     {}    {}\n", kIndent,
                          bfr::name(f.format), kNameWidth, mark(f.sampled), mark(f.filterLinear), mark(f.filterMinmax),
                          mark(f.storage), mark(f.storageAtomic), mark(f.colorAttachment), mark(f.blend),
                          mark(f.depthStencil), mark(f.texelBuffer),
                          f.sampleCounts == 0 ? "·" : samples(f.sampleCounts)));
    }
    if (!unsupported.empty()) {
        print("\n");
        row("Unsupported", unsupported);
    }
}

void printFormats(const bfr::Instance& instance, std::uint32_t adapter, const Options& options) {
    const auto formats = instance.formats(adapter);
    if (!formats) {
        row("Formats", "can't list them: " + formats.error().message);
        return;
    }
    printFormatCounts(*formats);
    if (options.formats) {
        printFormatTable(*formats);
    }
}

void printSurface(const bfr::Instance& instance, std::uint32_t adapter) {
    const auto caps = instance.surfaceCaps(adapter, {.window = {.system = bfr::WindowSystem::Headless}});
    if (!caps) {
        row("Window", "can't ask: " + caps.error().message);
        return;
    }
    std::vector<std::string> modes;
    for (const auto& [has, mode] : {std::pair{caps->fifo, "fifo"}, std::pair{caps->fifoRelaxed, "fifo relaxed"},
                                    std::pair{caps->mailbox, "mailbox"}, std::pair{caps->immediate, "immediate"}}) {
        if (has) {
            modes.emplace_back(mode);
        }
    }
    const std::string most = caps->maxImageCount == 0 ? "any" : std::to_string(caps->maxImageCount);
    row("Window", {std::format("{} to {} images", caps->minImageCount, most),
                   std::format("{}x{} to {}x{}", caps->minWidth, caps->minHeight, caps->maxWidth, caps->maxHeight)});
    row("Present", modes);
    const auto formats = instance.surfaceFormats(adapter, {.window = {.system = bfr::WindowSystem::Headless}});
    if (!formats) {
        row("Swap formats", "can't list them: " + formats.error().message);
        return;
    }
    std::vector<std::string> names;
    for (const bfr::SurfaceFormat& format : *formats) {
        names.push_back(std::format("{} {}", bfr::name(format.format), bfr::name(format.colorSpace)));
    }
    row("Swap formats", names);
}

// The budget needs a device, so this opens one.
void printBudget(const bfr::Instance& instance, std::uint32_t adapter, const bfr::CapsMemory& memory) {
    const auto device = instance.createDevice({.adapter = adapter});
    if (!device) {
        row("Budget", "can't open a device: " + device.error().message);
        return;
    }
    const auto budget = device->budget();
    if (!budget) {
        row("Budget", budget.error().message);
        return;
    }
    std::vector<std::string> heaps;
    for (std::uint32_t heap = 0; heap < budget->heapCount; ++heap) {
        const bfr::MemoryBudgetHeap& at = budget->heaps.at(heap);
        heaps.push_back(std::format("{} used of {} {}", bytes(at.usage), bytes(at.budget), heapKind(memory, heap)));
    }
    row("Budget", heaps);
}

// --all: every field under its own name, a group to a line.
std::string text(bool value) {
    return value ? "true" : "false";
}
std::string text(std::uint32_t value) {
    return std::to_string(value);
}
std::string text(std::uint64_t value) {
    return std::to_string(value);
}
std::string text(float value) {
    return std::format("{}", value);
}
std::string text(const std::array<std::uint32_t, 3>& value) {
    return std::format("{} {} {}", value[0], value[1], value[2]);
}
std::string text(const std::string& value) {
    return value;
}
template <class Enum>
    requires std::is_enum_v<Enum>
std::string text(Enum value) {
    return std::string{bfr::name(value)};
}
template <std::size_t Size>
std::string text(const std::array<std::uint8_t, Size>& value) {
    std::string out;
    for (const std::uint8_t byte : value) {
        out += std::format("{:02x}", byte);
    }
    return out;
}

template <class Value>
std::string field(std::string_view name, const Value& value) {
    return std::format("{} {}", name, text(value));
}

void group(std::string_view name, std::initializer_list<std::string> fields) {
    row(name, std::vector(fields));
}

void printAllInfo(const bfr::Caps& caps) {
    const bfr::CapsInfo& i = caps.info;
    group("info", {field("name", i.name), field("type", i.type), field("backend", i.backend),
                   field("vendorId", i.vendorId), field("deviceId", i.deviceId), field("driverName", i.driverName),
                   field("driverInfo", i.driverInfo), field("driverVersion", i.driverVersion),
                   field("api", std::format("{}.{}.{}", i.apiMajor, i.apiMinor, i.apiPatch)),
                   field("deviceUuid", i.deviceUuid), field("driverUuid", i.driverUuid), field("luid", i.luid),
                   field("luidValid", i.luidValid), field("cacheUuid", i.cacheUuid)});
}

void printAllLimits(const bfr::CapsLimits& l) {
    group("limits", {field("maxTexture1d", l.maxTexture1d),
                     field("maxTexture2d", l.maxTexture2d),
                     field("maxTexture3d", l.maxTexture3d),
                     field("maxTextureCube", l.maxTextureCube),
                     field("maxTextureLayers", l.maxTextureLayers),
                     field("maxBufferSize", l.maxBufferSize),
                     field("maxAllocationSize", l.maxAllocationSize),
                     field("maxAllocations", l.maxAllocations),
                     field("maxSamplerObjects", l.maxSamplerObjects),
                     field("maxUniformBufferRange", l.maxUniformBufferRange),
                     field("maxStorageBufferRange", l.maxStorageBufferRange),
                     field("maxTexelBufferElements", l.maxTexelBufferElements),
                     field("maxPushConstantsSize", l.maxPushConstantsSize),
                     field("maxColorAttachments", l.maxColorAttachments),
                     field("maxRenderWidth", l.maxRenderWidth),
                     field("maxRenderHeight", l.maxRenderHeight),
                     field("maxRenderLayers", l.maxRenderLayers),
                     field("colorSampleCounts", l.colorSampleCounts),
                     field("integerSampleCounts", l.integerSampleCounts),
                     field("depthSampleCounts", l.depthSampleCounts),
                     field("stencilSampleCounts", l.stencilSampleCounts),
                     field("noAttachmentSampleCounts", l.noAttachmentSampleCounts),
                     field("storageTextureSampleCounts", l.storageTextureSampleCounts),
                     field("maxViewports", l.maxViewports),
                     field("maxViewportWidth", l.maxViewportWidth),
                     field("maxViewportHeight", l.maxViewportHeight),
                     field("maxDrawIndirectCount", l.maxDrawIndirectCount),
                     field("maxClipDistances", l.maxClipDistances),
                     field("maxCullDistances", l.maxCullDistances),
                     field("maxSamplerAnisotropy", l.maxSamplerAnisotropy),
                     field("maxSamplerLodBias", l.maxSamplerLodBias),
                     field("minShadingRateTile", l.minShadingRateTile),
                     field("maxShadingRateTile", l.maxShadingRateTile),
                     field("minUniformBufferAlignment", l.minUniformBufferAlignment),
                     field("minStorageBufferAlignment", l.minStorageBufferAlignment),
                     field("minTexelBufferAlignment", l.minTexelBufferAlignment),
                     field("copyOffsetAlignment", l.copyOffsetAlignment),
                     field("copyRowPitchAlignment", l.copyRowPitchAlignment),
                     field("nonCoherentAtomSize", l.nonCoherentAtomSize),
                     field("bufferTextureGranularity", l.bufferTextureGranularity),
                     field("timestampPeriod", l.timestampPeriod)});
}

void printAllShaders(const bfr::Caps& caps) {
    const bfr::CapsBindless& b = caps.bindless;
    group("bindless", {field("maxTextures", b.maxTextures), field("maxStorageTextures", b.maxStorageTextures),
                       field("maxStorageBuffers", b.maxStorageBuffers), field("maxUniformBuffers", b.maxUniformBuffers),
                       field("maxSamplers", b.maxSamplers), field("maxPerStage", b.maxPerStage)});
    const bfr::CapsCompute& c = caps.compute;
    group("compute", {field("maxWorkgroupCount", c.maxWorkgroupCount), field("maxWorkgroupSize", c.maxWorkgroupSize),
                      field("maxWorkgroupInvocations", c.maxWorkgroupInvocations),
                      field("maxSharedMemory", c.maxSharedMemory), field("subgroupSize", c.subgroupSize),
                      field("minSubgroupSize", c.minSubgroupSize), field("maxSubgroupSize", c.maxSubgroupSize)});
    const bfr::CapsMesh& m = caps.mesh;
    group("mesh",
          {field("maxMeshWorkgroupSize", m.maxMeshWorkgroupSize), field("maxMeshInvocations", m.maxMeshInvocations),
           field("maxMeshOutputVertices", m.maxMeshOutputVertices),
           field("maxMeshOutputPrimitives", m.maxMeshOutputPrimitives),
           field("maxMeshSharedMemory", m.maxMeshSharedMemory), field("maxTaskWorkgroupSize", m.maxTaskWorkgroupSize),
           field("maxTaskInvocations", m.maxTaskInvocations), field("maxTaskPayloadSize", m.maxTaskPayloadSize)});
    const bfr::CapsRayTracing& r = caps.rayTracing;
    group("rayTracing", {field("maxGeometries", r.maxGeometries), field("maxInstances", r.maxInstances),
                         field("maxPrimitives", r.maxPrimitives), field("scratchAlignment", r.scratchAlignment),
                         field("maxRecursionDepth", r.maxRecursionDepth), field("handleSize", r.handleSize),
                         field("handleAlignment", r.handleAlignment), field("tableAlignment", r.tableAlignment)});
}

void printAllFeatures(const bfr::CapsFeatures& f) {
    group("features", {field("rayQuery", f.rayQuery),
                       field("rayTracingPipeline", f.rayTracingPipeline),
                       field("accelerationStructure", f.accelerationStructure),
                       field("meshShader", f.meshShader),
                       field("taskShader", f.taskShader),
                       field("shadingRate", f.shadingRate),
                       field("shadingRateTexture", f.shadingRateTexture),
                       field("shaderFloat16", f.shaderFloat16),
                       field("shaderFloat64", f.shaderFloat64),
                       field("shaderInt8", f.shaderInt8),
                       field("shaderInt16", f.shaderInt16),
                       field("shaderInt64", f.shaderInt64),
                       field("atomicInt64", f.atomicInt64),
                       field("atomicInt64Texture", f.atomicInt64Texture),
                       field("atomicFloat32Add", f.atomicFloat32Add),
                       field("barycentrics", f.barycentrics),
                       field("conservativeRaster", f.conservativeRaster),
                       field("geometryShader", f.geometryShader),
                       field("tessellation", f.tessellation),
                       field("multiview", f.multiview),
                       field("sampleRateShading", f.sampleRateShading),
                       field("dualSourceBlend", f.dualSourceBlend),
                       field("independentBlend", f.independentBlend),
                       field("logicOp", f.logicOp),
                       field("depthClamp", f.depthClamp),
                       field("depthBounds", f.depthBounds),
                       field("wireframe", f.wireframe),
                       field("wideLines", f.wideLines),
                       field("samplerAnisotropy", f.samplerAnisotropy),
                       field("samplerMinmax", f.samplerMinmax),
                       field("compressionBc", f.compressionBc),
                       field("compressionEtc2", f.compressionEtc2),
                       field("compressionAstc", f.compressionAstc),
                       field("compressionAstcHdr", f.compressionAstcHdr),
                       field("multiDrawIndirect", f.multiDrawIndirect),
                       field("drawIndirectCount", f.drawIndirectCount),
                       field("drawIndirectFirstInstance", f.drawIndirectFirstInstance),
                       field("storageReadWithoutFormat", f.storageReadWithoutFormat),
                       field("storageWriteWithoutFormat", f.storageWriteWithoutFormat),
                       field("timestampsOnAllQueues", f.timestampsOnAllQueues),
                       field("pipelineStatistics", f.pipelineStatistics),
                       field("occlusionQueryPrecise", f.occlusionQueryPrecise),
                       field("sparseBinding", f.sparseBinding),
                       field("sparseTextures", f.sparseTextures),
                       field("sparseBuffers", f.sparseBuffers)});
}

void printAllMemory(const bfr::Caps& caps) {
    const bfr::CapsMemory& memory = caps.memory;
    group("memory", {field("heapCount", memory.heapCount), field("typeCount", memory.typeCount),
                     field("unified", memory.unified)});
    for (std::uint32_t heap = 0; heap < memory.heapCount; ++heap) {
        const bfr::MemoryHeap& at = memory.heaps.at(heap);
        group(std::format("heaps[{}]", heap),
              {field("size", at.size), field("deviceLocal", at.deviceLocal), field("hostVisible", at.hostVisible)});
    }
    for (std::uint32_t type = 0; type < memory.typeCount; ++type) {
        const bfr::MemoryType& at = memory.types.at(type);
        group(std::format("types[{}]", type),
              {field("heap", at.heap), field("deviceLocal", at.deviceLocal), field("hostVisible", at.hostVisible),
               field("hostCoherent", at.hostCoherent), field("hostCached", at.hostCached),
               field("lazilyAllocated", at.lazilyAllocated)});
    }
    const bfr::CapsQueues& q = caps.queues;
    group("queues",
          {field("graphics", q.graphics), field("compute", q.compute), field("copy", q.copy),
           field("graphicsTimestampBits", q.graphicsTimestampBits),
           field("computeTimestampBits", q.computeTimestampBits), field("copyTimestampBits", q.copyTimestampBits)});
}

void printAll(const bfr::Caps& caps) {
    print("\n");
    printAllInfo(caps);
    printAllLimits(caps.limits);
    printAllShaders(caps);
    printAllFeatures(caps.features);
    printAllMemory(caps);
}

void printAdapter(const bfr::Instance& instance, std::uint32_t adapter, const Options& options) {
    const auto caps = instance.caps(adapter);
    if (!caps) {
        print(std::format("Adapter {}: can't read its caps: {}\n", adapter, caps.error().message));
        return;
    }
    printHeading(caps->info);
    printMemory(caps->memory);
    printLimits(*caps);
    printShaders(*caps);
    print("\n");
    printFeatures(caps->features);
    print("\n");
    printFormats(instance, adapter, options);
    if (options.window) {
        printSurface(instance, adapter);
    }
    printBudget(instance, adapter, caps->memory);
    if (options.all) {
        printAll(*caps);
    }
}

int run(std::span<char*> args) {
    Options options;
    for (const std::string_view arg : args.subspan(1)) {
        if (arg == "--formats") {
            options.formats = true;
        }
        else if (arg == "--window") {
            options.window = true;
        }
        else if (arg == "--all") {
            options.all = true;
        }
        else {
            std::cerr << "usage: bfr_caps [--formats] [--window] [--all]\n";
            return 2;
        }
    }
    const auto instance = bfr::Instance::create({.backend = bfr::Backend::Vulkan});
    if (!instance) {
        std::cerr << "can't create an instance: " << instance.error().message << "\n";
        return 1;
    }
    const auto adapters = instance->adapters();
    if (!adapters) {
        std::cerr << "can't list adapters: " << adapters.error().message << "\n";
        return 1;
    }
    for (std::uint32_t adapter = 0; adapter < adapters->size(); ++adapter) {
        if (adapter != 0) {
            print("\n");
        }
        printAdapter(*instance, adapter, options);
    }
    return 0;
}

} // namespace

// BFR doesn't throw; formatting the output can run out of memory.
int main(int argc, char** argv) {
    try {
        return run(std::span(argv, static_cast<std::size_t>(argc)));
    } catch (const std::exception& error) {
        std::cerr << "bfr_caps: " << error.what() << "\n";
        return 1;
    }
}
