#include "caps.h"
#include "device.h"
#include "error.h"
#include "format.h"
#include "rhi.h"
#include "rhi_test_support.h"
#include "scoped_env.h"
#include "window.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_container_properties.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <initializer_list>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace bfr::rhi;
using namespace bfr::rhi::test;

namespace {

// The least Vulkan 1.3, with the floor's descriptor indexing, lets an adapter report.
constexpr std::uint32_t kMinSamples = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT;
constexpr std::uint32_t kMinTexture2d = 4096;
constexpr std::uint32_t kMinTextureLayers = 256;
constexpr std::uint64_t kMinBufferSize = std::uint64_t{1} << 30U;
constexpr std::uint32_t kMinPushConstants = 128;
constexpr std::uint32_t kMinColorAttachments = 4;
constexpr std::uint32_t kMinBindlessTextures = 500'000;
constexpr std::uint32_t kMinWorkgroupInvocations = 128;

// Each name in checks whose condition is false.
std::vector<std::string> failing(std::initializer_list<std::pair<bool, std::string_view>> checks) {
    std::vector<std::string> failed;
    for (const auto& [held, name] : checks) {
        if (!held) {
            failed.emplace_back(name);
        }
    }
    return failed;
}

// Each guarantee of Vulkan 1.3 and the floor that caps break.
std::vector<std::string> brokenGuarantees(const Caps& caps) {
    const auto& heaps = caps.memory.heaps;
    return failing({
        {caps.info.apiMajor == 1 && caps.info.apiMinor >= 3, "Vulkan 1.3"},
        {caps.info.vendorId != 0, "a vendor"},
        {!caps.info.driverName.empty(), "a driver name"},
        {std::ranges::any_of(caps.info.deviceUuid, [](std::uint8_t byte) { return byte != 0; }), "a device UUID"},
        {caps.limits.maxTexture2d >= kMinTexture2d, "4096 wide 2D textures"},
        {caps.limits.maxTextureLayers >= kMinTextureLayers, "256 layers"},
        {caps.limits.maxBufferSize >= kMinBufferSize, "1 GiB buffers"},
        {caps.limits.maxPushConstantsSize >= kMinPushConstants, "128 bytes of push constants"},
        {caps.limits.maxColorAttachments >= kMinColorAttachments, "4 color attachments"},
        {(caps.limits.colorSampleCounts & kMinSamples) == kMinSamples, "1 and 4 color samples"},
        {caps.limits.timestampPeriod > 0, "a timestamp period"},
        {caps.bindless.maxTextures >= kMinBindlessTextures, "500000 bindless textures"},
        {caps.compute.maxWorkgroupInvocations >= kMinWorkgroupInvocations, "128 invocations a workgroup"},
        {caps.compute.minSubgroupSize <= caps.compute.subgroupSize &&
             caps.compute.subgroupSize <= caps.compute.maxSubgroupSize,
         "a subgroup size within its range"},
        {caps.queues.graphics >= 1, "a graphics queue"},
        {std::ranges::any_of(heaps, &MemoryHeap::deviceLocal), "a device-local heap"},
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

// Each use of rgba8_unorm Vulkan requires that caps lacks.
std::vector<std::string> missingRgba8Uses(const FormatCaps& caps) {
    return failing({
        {caps.format == Format::Rgba8Unorm, "the format asked about"},
        {caps.sampled && caps.filterLinear, "linear sampling"},
        {caps.storage, "storage"},
        {caps.colorAttachment && caps.blend, "blended rendering"},
        {caps.copySrc && caps.copyDst, "copies"},
        {(caps.sampleCounts & kMinSamples) == kMinSamples, "1 and 4 samples"},
    });
}

// Each use of d16_unorm Vulkan requires that caps lacks, or has but shouldn't.
std::vector<std::string> missingD16Uses(const FormatCaps& caps) {
    return failing({
        {caps.depthStencil, "depth"},
        {!caps.colorAttachment, "no color"},
        {(caps.sampleCounts & kMinSamples) == kMinSamples, "1 and 4 samples"},
    });
}

// Each use of rgba8_unorm and d16_unorm Vulkan requires that adapter lacks,
// and each format its query fails for or answers about another format.
std::vector<std::string> formatProblems(const Rhi& rhi, std::size_t adapter) {
    const auto rgba8 = rhi.formatCaps(adapter, Format::Rgba8Unorm);
    const auto d16 = rhi.formatCaps(adapter, Format::D16Unorm);
    if (!rgba8 || !d16) {
        return {failureOf(rgba8) + failureOf(d16)};
    }
    std::vector<std::string> failed = missingRgba8Uses(*rgba8);
    std::ranges::move(missingD16Uses(*d16), std::back_inserter(failed));
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

TEST_CASE("Rhi reads each adapter's caps, within what Vulkan 1.3 guarantees", "[rhi][caps][gpu]") {
    std::vector<std::string> messages;
    const auto rhi = Rhi::create(recordingInto(messages), kValidation);
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());

    for (const Adapter& adapter : (*rhi)->adapters()) {
        INFO(adapter.name);
        CHECK_THAT(brokenGuarantees(adapter.caps), Catch::Matchers::IsEmpty());
        CHECK_THAT(limitsWithoutFeatures(adapter.caps), Catch::Matchers::IsEmpty());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Rhi reports what the driver says about an adapter", "[rhi][caps][gpu]") {
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv vendor("BFR_ADAPTER_VENDOR_ID", "0x1234");
    const ScopedEnv noFloat16("BFR_ADAPTER_NO_FEATURE", "shaderFloat16");

    const auto rhi = Rhi::create({}, {});

    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());
    for (const Adapter& adapter : (*rhi)->adapters()) {
        INFO(adapter.name);
        CHECK(adapter.caps.info.vendorId == 0x1234);
        CHECK_FALSE(adapter.caps.features.shaderFloat16);
    }
}

TEST_CASE("Rhi reports what every adapter does with each format", "[rhi][caps][gpu]") {
    std::vector<std::string> messages;
    const auto rhi = Rhi::create(recordingInto(messages), kValidation);
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());

    for (std::size_t adapter = 0; adapter < (*rhi)->adapters().size(); ++adapter) {
        INFO((*rhi)->adapters()[adapter].name);
        CHECK_THAT(formatProblems(**rhi, adapter), Catch::Matchers::IsEmpty());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Rhi reports no sample counts for a format the API says it can't render to", "[rhi][caps][gpu]") {
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv failing("BFR_FAIL", "vkGetPhysicalDeviceImageFormatProperties2");
    const ScopedEnv unsupported("BFR_FAIL_RESULT", "VK_ERROR_FORMAT_NOT_SUPPORTED");
    const auto rhi = Rhi::create({}, {});
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());

    const auto rgba8 = (*rhi)->formatCaps(0, Format::Rgba8Unorm);

    INFO(failureOf(rgba8));
    REQUIRE(rgba8.has_value());
    CHECK(rgba8->colorAttachment);
    CHECK(rgba8->sampleCounts == 0);
}

TEST_CASE("Rhi reports a failed format query as the failure it is", "[rhi][caps][gpu]") {
    const auto [name, code] = GENERATE(std::pair{"VK_ERROR_UNKNOWN", bfr::core::ErrorCode::RhiQueryFailed},
                                       std::pair{"VK_ERROR_OUT_OF_HOST_MEMORY", bfr::core::ErrorCode::RhiOutOfMemory});
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv failing("BFR_FAIL", "vkGetPhysicalDeviceImageFormatProperties2");
    const ScopedEnv result("BFR_FAIL_RESULT", name);
    const auto rhi = Rhi::create({}, {});
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());

    const auto rgba8 = (*rhi)->formatCaps(0, Format::Rgba8Unorm);

    INFO(name);
    REQUIRE_FALSE(rgba8.has_value());
    CHECK(rgba8.error().code == code);
    CHECK_THAT(rgba8.error().message,
               Catch::Matchers::Equals(std::string{"vkGetPhysicalDeviceImageFormatProperties2: "} + name));
}

TEST_CASE("Rhi reports what a swapchain for a headless window may be", "[rhi][caps][gpu]") {
    std::vector<std::string> messages;
    const auto rhi = Rhi::create(recordingInto(messages), kValidation);
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());
    const std::size_t adapter = headlessAdapter(**rhi);

    const auto caps = (*rhi)->surfaceCaps(adapter, bfr::core::window::Headless{});
    const auto formats = (*rhi)->surfaceFormats(adapter, bfr::core::window::Headless{});

    INFO(failureOf(caps) << ' ' << failureOf(formats));
    REQUIRE(caps.has_value());
    REQUIRE(formats.has_value());
    CHECK(caps->minImageCount >= 1);
    CHECK((caps->maxImageCount == 0 || caps->maxImageCount >= caps->minImageCount));
    CHECK(caps->fifo);
    CHECK((caps->alphaOpaque || caps->alphaPremultiplied || caps->alphaPostmultiplied || caps->alphaInherit));
    CHECK_FALSE(formats->empty());
    CHECK(std::ranges::any_of(*formats,
                              [](const SurfaceFormat& format) { return format.colorSpace == ColorSpace::Srgb; }));
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Rhi reports a failed surface query as the failure it is", "[rhi][caps][gpu]") {
    const std::string function = GENERATE("vkGetPhysicalDeviceSurfaceCapabilitiesKHR",
                                          "vkGetPhysicalDeviceSurfaceFormatsKHR");
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv failing("BFR_FAIL", function);
    const auto rhi = Rhi::create({}, {});
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());
    const std::size_t adapter = headlessAdapter(**rhi);

    const auto caps = (*rhi)->surfaceCaps(adapter, bfr::core::window::Headless{});
    const auto formats = (*rhi)->surfaceFormats(adapter, bfr::core::window::Headless{});

    INFO(function);
    const bool capsFail = function == "vkGetPhysicalDeviceSurfaceCapabilitiesKHR";
    REQUIRE(caps.has_value() != capsFail);
    REQUIRE(formats.has_value() == capsFail);
    const bfr::core::Error& failed = capsFail ? caps.error() : formats.error();
    CHECK(failed.code == bfr::core::ErrorCode::RhiSurfaceFailed);
    CHECK_THAT(failed.message, Catch::Matchers::Equals(function + ": VK_ERROR_UNKNOWN"));
}

TEST_CASE("Rhi doesn't ask NVIDIA's driver about a headless window's swapchain on a loader that lets it crash",
          "[rhi][caps][gpu]") {
    const LoaderVersion loader{"1.4.318"};
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv nvidia("BFR_ADAPTER_DRIVER_ID", std::to_string(VK_DRIVER_ID_NVIDIA_PROPRIETARY));
    const auto rhi = Rhi::create({}, {});
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());

    const auto caps = (*rhi)->surfaceCaps(headlessAdapter(**rhi), bfr::core::window::Headless{});

    REQUIRE_FALSE(caps.has_value());
    CHECK(caps.error().code == bfr::core::ErrorCode::RhiPresentUnsupported);
    CHECK_THAT(caps.error().message, Catch::Matchers::ContainsSubstring("can't present to this window"));
    CHECK_THAT(caps.error().message, !Catch::Matchers::ContainsSubstring("loader"));
}

TEST_CASE("Rhi refuses queries about an adapter it does not list", "[rhi][caps][gpu]") {
    const auto rhi = Rhi::create({}, {});
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());
    const std::size_t missing = (*rhi)->adapters().size();

    CHECK((*rhi)->formatCaps(missing, Format::Rgba8Unorm).error().code == bfr::core::ErrorCode::RhiAdapterOutOfRange);
    CHECK((*rhi)->surfaceCaps(missing, bfr::core::window::Headless{}).error().code ==
          bfr::core::ErrorCode::RhiAdapterOutOfRange);
    CHECK((*rhi)->surfaceFormats(missing, bfr::core::window::Headless{}).error().code ==
          bfr::core::ErrorCode::RhiAdapterOutOfRange);
}

TEST_CASE("Device reports its memory budget heap by heap", "[rhi][caps][gpu]") {
    std::vector<std::string> messages;
    const auto rhi = Rhi::create(recordingInto(messages), kValidation);
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());

    for (std::size_t adapter = 0; adapter < (*rhi)->adapters().size(); ++adapter) {
        const Adapter& listed = (*rhi)->adapters()[adapter];
        INFO(listed.name);
        const auto device = (*rhi)->createDevice(adapter, kNoWindow);
        INFO(failureOf(device));
        REQUIRE(device.has_value());

        CHECK_THAT(budgetProblems(**device, listed.caps.memory), Catch::Matchers::IsEmpty());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device without VK_EXT_memory_budget fails with RhiUnsupported", "[rhi][caps][gpu]") {
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv hidden("BFR_ADAPTER_HIDE", "VK_EXT_memory_budget");
    const auto rhi = Rhi::create({}, {});
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());
    const auto device = (*rhi)->createDevice(0, kNoWindow);
    INFO(failureOf(device));
    REQUIRE(device.has_value());

    const auto budget = (*device)->budget();

    REQUIRE_FALSE(budget.has_value());
    CHECK(budget.error().code == bfr::core::ErrorCode::RhiUnsupported);
    CHECK_THAT(budget.error().message, Catch::Matchers::ContainsSubstring("VK_EXT_memory_budget"));
}
