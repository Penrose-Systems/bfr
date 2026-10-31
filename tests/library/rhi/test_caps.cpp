// What every backend reports about its adapters: caps that agree with
// themselves, 0 for each limit of a feature the adapter lacks, an answer for
// every format. Each case runs on every backend the build has.

#include "bfr/core/error.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/instance.h"
#include "rhi_test_support.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_container_properties.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

using namespace bfr::rhi;
using namespace bfr::rhi::test;

namespace {

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

// Each way caps disagree with themselves.
std::vector<std::string> inconsistencies(const Caps& caps) {
    const auto& heaps = caps.memory.heaps;
    return failing({
        {caps.compute.minSubgroupSize <= caps.compute.subgroupSize &&
             caps.compute.subgroupSize <= caps.compute.maxSubgroupSize,
         "a subgroup size within its range"},
        {std::ranges::all_of(caps.memory.types, [&heaps](const MemoryType& type) { return type.heap < heaps.size(); }),
         "types in listed heaps"},
    });
}

// Each limit caps reports for a feature it lacks that is not 0.
std::vector<std::string> limitsWithoutFeatures(const Caps& caps) {
    const Features& features = caps.features;
    return failing({
        {features.meshShader || caps.mesh.maxMeshOutputVertices == 0, "mesh"},
        {features.taskShader || caps.mesh.maxTaskPayloadSize == 0, "task"},
        {features.accelerationStructure || caps.rayTracing.maxInstances == 0, "acceleration structure"},
        {features.rayTracingPipeline || caps.rayTracing.handleSize == 0, "ray tracing pipeline"},
        {features.shadingRateTexture || caps.limits.maxShadingRateTile == 0, "shading rate texture"},
    });
}

// Each format whose query fails or answers about another format.
std::vector<std::string> formatProblems(const Instance& rhi, std::size_t adapter) {
    std::vector<std::string> failed;
    for (const Format format : kFormats) {
        const auto found = rhi.formatCaps(adapter, format);
        if (!found || found->format != format) {
            failed.push_back(std::to_string(static_cast<int>(format)) + " " + failureOf(found));
        }
    }
    return failed;
}

// What is wrong with device's budget for memory: a heap whose budget is 0 or
// more than the heap holds, or a failure other than the driver not saying.
std::vector<std::string> budgetProblems(const Device& device, const Memory& memory) {
    const auto budget = device.budget();
    if (!budget) {
        return budget.error().code == bfr::core::ErrorCode::RhiUnsupported
                   ? std::vector<std::string>{}
                   : std::vector<std::string>{failureOf(budget)};
    }
    if (budget->heaps.size() != memory.heaps.size()) {
        return {std::format("{} heaps, not {}", budget->heaps.size(), memory.heaps.size())};
    }
    std::vector<std::string> problems;
    for (std::size_t heap = 0; heap < memory.heaps.size(); ++heap) {
        const std::uint64_t bytes = budget->heaps[heap].budget;
        if (bytes == 0 || bytes > memory.heaps[heap].size) {
            problems.push_back(std::format("heap {}: budget {} of {}", heap, bytes, memory.heaps[heap].size));
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

    for (const Adapter& adapter : (*rhi)->adapters()) {
        INFO(adapter.name);
        CHECK_THAT(inconsistencies(adapter.caps), Catch::Matchers::IsEmpty());
        CHECK_THAT(limitsWithoutFeatures(adapter.caps), Catch::Matchers::IsEmpty());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Instance reports what every adapter does with each format", "[rhi][caps][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    const auto rhi = createInstance(recordingInto(backend, messages), kValidation);
    INFO(name(backend) << ' ' << failureOf(rhi));
    REQUIRE(rhi.has_value());

    for (std::size_t adapter = 0; adapter < (*rhi)->adapters().size(); ++adapter) {
        INFO((*rhi)->adapters()[adapter].name);
        CHECK_THAT(formatProblems(**rhi, adapter), Catch::Matchers::IsEmpty());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Instance reports what a swapchain for a headless window may be", "[rhi][caps][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    const auto rhi = createInstance(recordingInto(backend, messages), kValidation);
    INFO(name(backend) << ' ' << failureOf(rhi));
    REQUIRE(rhi.has_value());
    const std::size_t adapter = headlessAdapter(**rhi);

    const auto caps = (*rhi)->swapchainCaps(adapter, bfr::core::window::Headless{});
    const auto formats = (*rhi)->swapchainFormats(adapter, bfr::core::window::Headless{});

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
    const auto rhi = createInstance(describe(backend), {});
    INFO(name(backend) << ' ' << failureOf(rhi));
    REQUIRE(rhi.has_value());
    const std::size_t missing = (*rhi)->adapters().size();

    CHECK((*rhi)->formatCaps(missing, Format::Rgba8Unorm).error().code == bfr::core::ErrorCode::RhiAdapterOutOfRange);
    CHECK((*rhi)->swapchainCaps(missing, bfr::core::window::Headless{}).error().code ==
          bfr::core::ErrorCode::RhiAdapterOutOfRange);
    CHECK((*rhi)->swapchainFormats(missing, bfr::core::window::Headless{}).error().code ==
          bfr::core::ErrorCode::RhiAdapterOutOfRange);
}

TEST_CASE("Device reports its memory budget heap by heap", "[rhi][caps][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    const auto rhi = createInstance(recordingInto(backend, messages), kValidation);
    INFO(name(backend) << ' ' << failureOf(rhi));
    REQUIRE(rhi.has_value());

    for (std::size_t adapter = 0; adapter < (*rhi)->adapters().size(); ++adapter) {
        const Adapter& listed = (*rhi)->adapters()[adapter];
        INFO(listed.name);
        const auto device = (*rhi)->createDevice(adapter, kNoWindow, {});
        INFO(failureOf(device));
        REQUIRE(device.has_value());

        CHECK_THAT(budgetProblems(**device, listed.caps.memory), Catch::Matchers::IsEmpty());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}
