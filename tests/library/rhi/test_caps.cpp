// What every backend reports about its adapters: caps that agree with
// themselves, 0 for each limit of a feature the adapter lacks, an answer for
// every format. Each case runs on every backend the build has.

#include "bfr.hpp"
#include "bfr/bfr.h"
#include "bfr_test_support.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_container_properties.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <format>
#include <initializer_list>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace bfr;
using namespace bfr::test;

namespace {

// The last Format bfr.yaml declares. A format declared after it is listed past
// it, which formatProblems reports, until this names it.
constexpr Format kLastFormat = Format::Astc12x12Float;

// Each name in checks whose condition is false.
std::vector<std::string> failing(std::initializer_list<std::pair<bool, std::string>> checks) {
    std::vector<std::string> failed;
    for (const auto& [held, name] : checks) {
        if (!held) {
            failed.push_back(name);
        }
    }
    return failed;
}

// A granularity that is whole levels with sizes too, or has some sizes and
// not others.
bool granularityHolds(const CopyGranularity& granularity) {
    const bool none = granularity.width == 0 && granularity.height == 0 && granularity.depth == 0;
    const bool sized = granularity.width != 0 && granularity.height != 0 && granularity.depth != 0;
    return granularity.wholeLevels ? none : none || sized;
}

// Whether stages lists any stage.
bool anyStage(const TimestampStages& stages) {
    return stages.end || stages.start || stages.vertex || stages.fragment || stages.compute || stages.copy ||
           stages.task || stages.mesh || stages.rayTracing || stages.accelerationStructureBuild;
}

// Each way caps and the adapter's queue families disagree with themselves.
std::vector<std::string> inconsistencies(const Caps& caps, const std::vector<QueueFamily>& families) {
    const CapsMemory& memory = caps.memory;
    return failing({
        // A range of 0, 0 is none reported.
        {(caps.compute.minSubgroupSize == 0 && caps.compute.maxSubgroupSize == 0) ||
             (caps.compute.minSubgroupSize <= caps.compute.subgroupSize &&
              caps.compute.subgroupSize <= caps.compute.maxSubgroupSize),
         "a subgroup size within its range, where one is reported"},
        {std::ranges::all_of(std::span{memory.types}.first(memory.typeCount),
                             [&memory](const MemoryType& type) { return type.heap < memory.heapCount; }),
         "types in listed heaps"},
        {std::ranges::all_of(families, [](const QueueFamily& family) { return granularityHolds(family.granularity); }),
         "copy granularities of whole levels or of every size"},
        {std::ranges::all_of(
             families, [](const QueueFamily& family) { return (!family.graphics && !family.compute) || family.copy; }),
         "families that run graphics or compute copy too"},
        {std::ranges::all_of(families,
                             [](const QueueFamily& family) {
                                 return family.timestamps || !anyStage(family.timestampStages);
                             }),
         "no timestamp stages on a family that writes no timestamps"},
    });
}

// Each limit caps reports for a feature it lacks that is not 0.
std::vector<std::string> limitsWithoutFeatures(const Caps& caps) {
    const CapsFeatures& features = caps.features;
    return failing({
        {features.meshShader || caps.mesh.maxMeshOutputVertices == 0, "mesh"},
        {features.taskShader || caps.mesh.maxTaskPayloadSize == 0, "task"},
        {features.accelerationStructure || caps.rayTracing.maxInstances == 0, "acceleration structure"},
        {features.accelerationStructure || caps.rayTracing.instanceSize == 0, "instance records"},
        {features.accelerationStructure || caps.rayTracing.propertyEntryBytes == 0, "property entries"},
        {features.accelerationStructure ||
             !(features.accelerationStructureSerialization || features.accelerationStructureCurrentSize ||
               features.serializedBottomCount || features.rayPositionFetch),
         "acceleration structure features"},
        {features.rayInvocationReorder ||
             (!caps.rayTracing.reordersInvocations && caps.rayTracing.maxBindingTableRecordIndex == 0),
         "invocation reorder"},
        {features.rayTracingPipeline || !(features.rayTracingPipelineLibrary || features.pipelineLibraryIdentifiers ||
                                          features.rayTracingCaptureReplay),
         "ray tracing pipeline features"},
        {features.rayTracingPipelineLibrary || !features.pipelineLibraryIdentifiers, "library identifiers"},
        {features.rayTracingCaptureReplay || !features.rayTracingCaptureReplayMixed, "mixed capture replay"},
        {features.rayTracingCaptureReplay || caps.rayTracing.captureReplayIdentifierSize == 0,
         "capture-replay identifiers"},
        {features.accelerationStructure || !features.accelerationStructureIndirectBuild, "indirect builds"},
        {features.accelerationStructure || !features.opacityMicromap, "opacity micromaps"},
        {features.rayTracingPipeline || !features.rayTracingMotion, "motion"},
        {features.rayTracingMotion || !features.rayTracingMotionIndirect, "indirect motion"},
        {features.rayTracingMotion ||
             (caps.rayTracing.maxMotionKeyframes == 0 && caps.rayTracing.motionInstanceSize == 0),
         "motion limits"},
        {features.opacityMicromap || (caps.rayTracing.maxOpacity2StateSubdivisionLevel == 0 &&
                                      caps.rayTracing.maxOpacity4StateSubdivisionLevel == 0),
         "opacity micromap levels"},
        {features.accelerationStructure || caps.bindless.maxAccelerationStructures == 0,
         "acceleration structure table"},
        {features.accelerationStructure || !(features.rayQuery || features.rayQueryGraphics), "ray queries"},
        {features.rayTracingPipeline || caps.rayTracing.identifierSize == 0, "ray tracing pipeline"},
        {features.rayTracingPipeline || !features.traceRaysIndirect, "indirect traces"},
        {features.shadingRateTexture || caps.limits.maxShadingRateTile == std::array<std::uint32_t, 2>{},
         "shading rate texture"},
    });
}

// Each format the adapter's list fails on, lacks or holds out of place: it
// lists every Format but Undefined, in the order Format declares them.
std::vector<std::string> formatProblems(const Instance& instance, std::uint32_t adapter) {
    const auto found = instance.formats(adapter);
    if (!found) {
        return {failureOf(found)};
    }
    std::vector<std::string> failed;
    auto next = found->begin();
    for (auto value = static_cast<std::int32_t>(Format::Undefined) + 1; value <= static_cast<std::int32_t>(kLastFormat);
         ++value) {
        const auto format = static_cast<Format>(value);
        if (name(format).empty()) {
            continue;
        }
        if (next == found->end() || next->format != format) {
            failed.push_back(std::format("{} {} is not listed in its place", value, name(format)));
            continue;
        }
        ++next;
    }
    for (; next != found->end(); ++next) {
        failed.push_back(std::format("{} is listed past the last format", static_cast<std::int32_t>(next->format)));
    }
    return failed;
}

// What is wrong with device's budget for memory: a heap whose budget is 0 or
// more than the heap holds, or a failure other than the driver not saying.
std::vector<std::string> budgetProblems(const Device& device, const CapsMemory& memory) {
    const auto budget = device.budget();
    if (!budget) {
        return budget.error().code == Result::ErrorUnsupported ? std::vector<std::string>{}
                                                               : std::vector<std::string>{failureOf(budget)};
    }
    if (budget->heapCount != memory.heapCount) {
        return {std::format("{} heaps, not {}", budget->heapCount, memory.heapCount)};
    }
    std::vector<std::string> problems;
    for (std::uint32_t heap = 0; heap < memory.heapCount; ++heap) {
        const std::uint64_t bytes = budget->heaps.at(heap).budget;
        if (bytes == 0 || bytes > memory.heaps.at(heap).size) {
            problems.push_back(std::format("heap {}: budget {} of {}", heap, bytes, memory.heaps.at(heap).size));
        }
    }
    return problems;
}

} // namespace

TEST_CASE("Instance reads each adapter's caps, consistent and 0 for each feature it lacks", "[rhi][caps][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    const auto rhi = createInstance(recordingInto(backend, messages), kValidation);
    INFO(name(backend) << ' ' << failureOf(rhi));
    REQUIRE(rhi.has_value());

    const std::vector<AdapterInfo> adapters = adaptersOf(*rhi);
    for (std::uint32_t adapter = 0; adapter < adapters.size(); ++adapter) {
        INFO(adapters[adapter].name);
        const Caps caps = capsOf(*rhi, adapter);
        CHECK_THAT(inconsistencies(caps, queueFamiliesOf(*rhi, adapter)), Catch::Matchers::IsEmpty());
        CHECK_THAT(limitsWithoutFeatures(caps), Catch::Matchers::IsEmpty());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Instance reports what every adapter does with each format", "[rhi][caps][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    const auto rhi = createInstance(recordingInto(backend, messages), kValidation);
    INFO(name(backend) << ' ' << failureOf(rhi));
    REQUIRE(rhi.has_value());

    const std::vector<AdapterInfo> adapters = adaptersOf(*rhi);
    for (std::uint32_t adapter = 0; adapter < adapters.size(); ++adapter) {
        INFO(adapters[adapter].name);
        CHECK_THAT(formatProblems(*rhi, adapter), Catch::Matchers::IsEmpty());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Instance reports what a swapchain for a headless window may be", "[rhi][caps][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    const auto rhi = createInstance(recordingInto(backend, messages), kValidation);
    INFO(name(backend) << ' ' << failureOf(rhi));
    REQUIRE(rhi.has_value());
    const std::uint32_t adapter = headlessAdapter(*rhi);

    const auto caps = rhi->swapchainCaps(adapter, {.window = kHeadless});
    const auto formats = rhi->swapchainFormats(adapter, {.window = kHeadless});

    INFO(failureOf(caps) << ' ' << failureOf(formats));
    REQUIRE(caps.has_value());
    REQUIRE(formats.has_value());
    CHECK(caps->minImageCount >= 1);
    CHECK((caps->maxImageCount == 0 || caps->maxImageCount >= caps->minImageCount));
    CHECK(caps->fifo);
    CHECK((caps->alphaOpaque || caps->alphaPremultiplied || caps->alphaPostmultiplied || caps->alphaInherit));
    CHECK_FALSE(formats->empty());
    CHECK(std::ranges::any_of(*formats,
                              [](const SwapchainFormat& format) { return format.colorSpace == ColorSpace::Srgb; }));
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Instance refuses queries about an adapter it does not list", "[rhi][caps][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    const auto rhi = Instance::create(describe(backend));
    INFO(name(backend) << ' ' << failureOf(rhi));
    REQUIRE(rhi.has_value());
    const auto missing = static_cast<std::uint32_t>(adaptersOf(*rhi).size());

    CHECK(rhi->formats(missing).error().code == Result::ErrorInvalidArgument);
    CHECK(rhi->queueFamilies(missing).error().code == Result::ErrorInvalidArgument);
    CHECK(rhi->queueFamilyPresents(missing, 0, {.window = kHeadless}).error().code == Result::ErrorInvalidArgument);
    CHECK(rhi->findQueueFamily(missing, {}).error().code == Result::ErrorInvalidArgument);
    CHECK(rhi->swapchainCaps(missing, {.window = kHeadless}).error().code == Result::ErrorInvalidArgument);
    CHECK(rhi->swapchainFormats(missing, {.window = kHeadless}).error().code == Result::ErrorInvalidArgument);
}

TEST_CASE("Device reports its memory budget heap by heap", "[rhi][caps][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    const auto rhi = createInstance(recordingInto(backend, messages), kValidation);
    INFO(name(backend) << ' ' << failureOf(rhi));
    REQUIRE(rhi.has_value());

    const std::vector<AdapterInfo> adapters = adaptersOf(*rhi);
    for (std::uint32_t adapter = 0; adapter < adapters.size(); ++adapter) {
        INFO(adapters[adapter].name);
        const CapsMemory memory = capsOf(*rhi, adapter).memory;
        const auto device = rhi->createDevice(testDeviceOn(*rhi, adapter, kNoWindow));
        INFO(failureOf(device));
        REQUIRE(device.has_value());

        CHECK_THAT(budgetProblems(*device, memory), Catch::Matchers::IsEmpty());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device refuses a feature asked for without one it needs, naming both, and one BFR has no call for",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        auto created = createInstance(recordingInto(backend, messages), kValidation);
        INFO(failureOf(created));
        REQUIRE(created.has_value());
        const Instance instance = std::move(*created);

        const auto unmet = instance.createDevice(
            testDeviceWith(instance, kFirstAdapter, kNoWindow, [](DeviceDesc& desc) {
                desc.features.rayQuery = true;
                desc.features.rayQueryGraphics = true;
            }));
        REQUIRE_FALSE(unmet.has_value());
        CHECK(unmet.error().code == Result::ErrorInvalidArgument);
        CHECK_THAT(unmet.error().message.text(),
                   Catch::Matchers::ContainsSubstring("ray_query needs acceleration_structure") &&
                       !Catch::Matchers::ContainsSubstring("ray_query_graphics needs"));

        const auto unbound = instance.createDevice(
            testDeviceWith(instance, kFirstAdapter, kNoWindow,
                           [](DeviceDesc& desc) { desc.features.sparseBuffers = true; }));
        REQUIRE_FALSE(unbound.has_value());
        CHECK(unbound.error().code == Result::ErrorInvalidArgument);
        CHECK_THAT(unbound.error().message.text(),
                   Catch::Matchers::ContainsSubstring("sparse_buffers needs sparse_binding"));

        const auto noCall = instance.createDevice(
            testDeviceWith(instance, kFirstAdapter, kNoWindow,
                           [](DeviceDesc& desc) { desc.features.meshShadingRate = true; }));
        REQUIRE_FALSE(noCall.has_value());
        CHECK(noCall.error().code == Result::ErrorUnsupported);
        CHECK_THAT(noCall.error().message.text(), Catch::Matchers::ContainsSubstring("BFR has no call for: "
                                                                                     "mesh_shading_rate"));
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device has the features it was created with and no others, and the limits that follow them",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages), [](DeviceDesc& desc, const Caps& caps) {
                                          desc.features.samplerAnisotropy = caps.features.samplerAnisotropy;
                                          desc.features.dynamicTopology = caps.features.dynamicTopology;
                                          desc.features.subgroupRotate = caps.features.subgroupRotate;
                                          desc.features.floatControls2 = caps.features.floatControls2;
                                          desc.features.expectAssume = caps.features.expectAssume;
                                      }};
        const Caps adapter = capsOf(made.instance, kFirstAdapter);
        const auto device = made.device.deviceCaps();
        INFO(failureOf(device));
        REQUIRE(device.has_value());
        CHECK(device->features.samplerAnisotropy == adapter.features.samplerAnisotropy);
        CHECK(device->features.subgroupRotate == adapter.features.subgroupRotate);
        CHECK(device->features.floatControls2 == adapter.features.floatControls2);
        CHECK(device->features.expectAssume == adapter.features.expectAssume);
        CHECK(device->features.samplerAnisotropyNearestMip ==
              (adapter.features.samplerAnisotropy && adapter.features.samplerAnisotropyNearestMip));
        CHECK(device->features.dynamicTopology == adapter.features.dynamicTopology);
        // Asked for nothing else, it has none of the rest.
        CHECK_THAT(failing({{!device->features.swapchain, "swapchain"},
                            {!device->features.geometryShader, "geometryShader"},
                            {!device->features.meshShader, "meshShader"},
                            {!device->features.accelerationStructure, "accelerationStructure"},
                            {!device->features.bufferAddress, "bufferAddress"},
                            {!device->features.rayQuery, "rayQuery"},
                            {!device->features.multiview, "multiview"},
                            {!device->features.imageCubeArray, "imageCubeArray"},
                            {!device->features.robustBufferAccess, "robustBufferAccess"},
                            {!device->sparse.binding, "sparse.binding"},
                            {!device->timestamps.hostClocks.monotonic, "hostClocks.monotonic"},
                            {device->mesh.maxMeshOutputVertices == 0, "mesh.maxMeshOutputVertices"},
                            {device->rayTracing.maxGeometries == 0, "rayTracing.maxGeometries"},
                            {device->bindless.maxAccelerationStructures == 0, "bindless.maxAccelerationStructures"}}),
                   Catch::Matchers::IsEmpty());
        // What no request turns on stays as the adapter says.
        CHECK(device->features.sampleMask == adapter.features.sampleMask);
        CHECK(device->features.compressionBc == adapter.features.compressionBc);
        CHECK(device->limits.maxTexture2d == adapter.limits.maxTexture2d);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device refuses a call of a feature it was not created with, though the adapter has it", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        const Caps adapter = capsOf(made.instance, kFirstAdapter);
        if (adapter.features.samplerAnisotropy) {
            constexpr std::uint32_t kAnisotropy = 4;
            CHECK(resultOf(device.createSampler({.maxAnisotropy = kAnisotropy})) == Result::ErrorUnsupported);
        }
        // Where the backend's version has it with nothing to enable (Vulkan
        // 1.3's), the device has it unasked, and the call follows its caps.
        const auto has = device.deviceCaps();
        INFO(failureOf(has));
        REQUIRE(has.has_value());
        if (adapter.features.dynamicTopology) {
            CHECK(inList(device, 0, [](const Device& on, CommandList list) {
                      return bfr_set_topology(on.get(), list.get(), BFR_TOPOLOGY_TRIANGLE_STRIP);
                  }) == (has->features.dynamicTopology ? Result::Success : Result::ErrorUnsupported));
        }
        if (adapter.features.formatBgra4Unorm) {
            constexpr std::uint32_t kSide = 4;
            CHECK(resultOf(device.createTexture({.format = Format::Bgra4Unorm,
                                                 .width = kSide,
                                                 .height = kSide,
                                                 .depth = 1,
                                                 .layers = 1,
                                                 .mipLevels = 1,
                                                 .samples = 1,
                                                 .usage = {.sampled = true}})) ==
                  (has->features.formatBgra4Unorm ? Result::Success : Result::ErrorUnsupported));
        }
        if (adapter.features.imageCubeArray) {
            constexpr std::uint32_t kSide = 4;
            constexpr std::uint32_t kTwoCubes = 12;
            CHECK(resultOf(device.createTexture({.dimension = TextureDimension::Cube,
                                                 .format = Format::Rgba8Unorm,
                                                 .width = kSide,
                                                 .height = kSide,
                                                 .depth = 1,
                                                 .layers = kTwoCubes,
                                                 .mipLevels = 1,
                                                 .samples = 1,
                                                 .usage = {.sampled = true}})) == Result::ErrorUnsupported);
        }
        if (adapter.sparse.buffers) {
            constexpr std::uint64_t kSparseBytes = std::uint64_t{1} << 20U;
            const std::uint32_t tile = adapter.sparse.tileByteCount != 0 ? adapter.sparse.tileBytes.front() : 0;
            CHECK(resultOf(device.createBuffer(
                      {.size = kSparseBytes, .usage = {.copyDst = true}, .sparse = true, .tileBytes = tile})) ==
                  Result::ErrorUnsupported);
        }
        if (adapter.features.pipelineStatistics) {
            CHECK(
                resultOf(device.createQueryPool(
                    {.type = QueryType::PipelineStatistics, .count = 1, .statistics = {.vertexInvocations = true}})) ==
                Result::ErrorUnsupported);
        }
        if (adapter.features.occlusionQueryPrecise) {
            auto pool = device.createQueryPool({.type = QueryType::Occlusion, .count = 1});
            INFO(failureOf(pool));
            REQUIRE(pool.has_value());
            CHECK(inList(device, 0, [&pool](const Device& on, CommandList list) {
                      return on.beginQuery(list, *pool, 0, true);
                  }) == Result::ErrorUnsupported);
        }
        constexpr std::uint64_t kDrawBytes = 16;
        constexpr std::uint32_t kTwoDraws = 2;
        const Buffer arguments = created(device,
                                         BufferDesc{.size = kDrawBytes * kTwoDraws, .usage = {.indirect = true}});
        if (adapter.features.multiDrawIndirect) {
            CHECK(inList(device, 0, [&arguments](const Device& on, CommandList list) {
                      return on.drawIndirect(list, arguments, 0, kTwoDraws, kDrawBytes);
                  }) == Result::ErrorUnsupported);
        }
        if (adapter.features.drawIndirectCount) {
            CHECK(inList(device, 0, [&arguments](const Device& on, CommandList list) {
                      return on.drawIndirectCount(list, arguments, 0, arguments, kDrawBytes, 1, kDrawBytes);
                  }) == Result::ErrorUnsupported);
        }
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}
