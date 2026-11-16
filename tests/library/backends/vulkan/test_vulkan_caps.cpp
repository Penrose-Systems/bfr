// What the Vulkan backend reports about its adapters: what the driver says,
// and how each failed query comes back.

#include "bfr.hpp"
#include "rhi/bfr_test_support.h"
#include "scoped_env.h"
#include "vulkan_test_support.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_container_properties.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using namespace bfr;
using namespace bfr::test;
using namespace bfr::test::vulkan;

namespace {

// Each thing Vulkan promises about an adapter's info that info lacks.
std::vector<std::string> missingVulkanInfo(const CapsInfo& info) {
    std::vector<std::string> missing;
    if (info.apiMajor != 1 || info.apiMinor < 2) {
        missing.emplace_back("Vulkan 1.2");
    }
    if (info.vendorId == 0) {
        missing.emplace_back("a vendor");
    }
    if (info.driverName.empty()) {
        missing.emplace_back("a driver name");
    }
    if (std::ranges::none_of(info.deviceUuid, [](std::uint8_t byte) { return byte != 0; })) {
        missing.emplace_back("a device UUID");
    }
    return missing;
}

} // namespace

TEST_CASE("Vulkan lists only adapters at 1.2 or newer, each with a vendor, a driver name and a device UUID",
          "[vulkan][caps][gpu]") {
    std::vector<std::string> messages;
    const auto rhi = createInstance(recordingInto(messages), kValidation);
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());

    const std::vector<AdapterInfo> adapters = adaptersOf(*rhi);
    for (std::uint32_t adapter = 0; adapter < adapters.size(); ++adapter) {
        INFO(adapters[adapter].name);
        CHECK_THAT(missingVulkanInfo(capsOf(*rhi, adapter).info), Catch::Matchers::IsEmpty());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Vulkan reports what the driver says about an adapter", "[vulkan][caps][gpu]") {
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv vendor("BFR_ADAPTER_VENDOR_ID", "0x1234");
    const ScopedEnv noFloat16("BFR_ADAPTER_NO_FEATURE", "shaderFloat16");

    const auto rhi = Instance::create(vulkanDesc());

    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());
    const std::vector<AdapterInfo> adapters = adaptersOf(*rhi);
    for (std::uint32_t adapter = 0; adapter < adapters.size(); ++adapter) {
        INFO(adapters[adapter].name);
        const Caps caps = capsOf(*rhi, adapter);
        CHECK(caps.info.vendorId == 0x1234);
        CHECK_FALSE(caps.features.shaderFloat16);
    }
}

TEST_CASE("Vulkan reports LOD bias on a portability driver as the driver answers it", "[vulkan][caps][gpu]") {
    const AdapterOverrideLayer adapterOverride{"VK_LAYER_KHRONOS_validation"};
    const ScopedEnv added("BFR_ADAPTER_ADD", "VK_KHR_portability_subset");
    const bool answer = GENERATE(false, true);
    const ScopedEnv lodBias("BFR_ADAPTER_LOD_BIAS", answer ? "1" : "0");
    std::vector<std::string> errors;
    const auto rhi = createInstance(recordingErrorsInto(errors), kValidation);
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());
    const std::uint32_t adapter = cpuAdapter(*rhi);

    CHECK(capsOf(*rhi, adapter).features.samplerLodBias == answer);
    if (answer) {
        const auto device = rhi->createDevice(testDeviceOn(adapter, kNoWindow));
        INFO(failureOf(device));
        REQUIRE(device.has_value());
        constexpr float kLodBias = 1;
        CHECK(device->createSampler({.lodBias = kLodBias}).has_value());
        CHECK_THAT(errors, Catch::Matchers::IsEmpty());
    }
}

TEST_CASE("Vulkan reports no sample counts for a format the API says it can't render to", "[vulkan][caps][gpu]") {
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv failing("BFR_FAIL", "vkGetPhysicalDeviceImageFormatProperties2");
    const ScopedEnv unsupported("BFR_FAIL_RESULT", "VK_ERROR_FORMAT_NOT_SUPPORTED");
    const auto rhi = Instance::create(vulkanDesc());
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());

    const auto rgba8 = formatCapsOf(*rhi, kFirstAdapter, Format::Rgba8Unorm);

    INFO(failureOf(rgba8));
    REQUIRE(rgba8.has_value());
    CHECK(rgba8->colorAttachment);
    CHECK(rgba8->sampleCounts == 0);
}

TEST_CASE("Vulkan reports a failed format query as the failure it is", "[vulkan][caps][gpu]") {
    const auto [name, code] = GENERATE(std::pair{"VK_ERROR_UNKNOWN", Result::ErrorQueryFailed},
                                       std::pair{"VK_ERROR_OUT_OF_HOST_MEMORY", Result::ErrorOutOfMemory});
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv failing("BFR_FAIL", "vkGetPhysicalDeviceImageFormatProperties2");
    const ScopedEnv result("BFR_FAIL_RESULT", name);
    const auto rhi = Instance::create(vulkanDesc());
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());

    const auto rgba8 = formatCapsOf(*rhi, kFirstAdapter, Format::Rgba8Unorm);

    INFO(name);
    REQUIRE_FALSE(rgba8.has_value());
    CHECK(rgba8.error().code == code);
    CHECK_THAT(rgba8.error().message.text(),
               Catch::Matchers::Equals(std::string{"vkGetPhysicalDeviceImageFormatProperties2: "} + name));
}

TEST_CASE("Vulkan reports a failed surface query as the failure it is", "[vulkan][caps][gpu]") {
    const std::string function = GENERATE("vkGetPhysicalDeviceSurfaceCapabilitiesKHR",
                                          "vkGetPhysicalDeviceSurfaceFormatsKHR");
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv failing("BFR_FAIL", function);
    const auto rhi = Instance::create(vulkanDesc());
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());
    const std::uint32_t adapter = cpuAdapter(*rhi);

    const auto caps = rhi->swapchainCaps(adapter, {.window = kHeadless});
    const auto formats = rhi->swapchainFormats(adapter, {.window = kHeadless});

    INFO(function);
    const bool capsFail = function == "vkGetPhysicalDeviceSurfaceCapabilitiesKHR";
    REQUIRE(caps.has_value() != capsFail);
    REQUIRE(formats.has_value() == capsFail);
    const Error& failed = capsFail ? caps.error() : formats.error();
    CHECK(failed.code == Result::ErrorWindowFailed);
    CHECK_THAT(failed.message.text(), Catch::Matchers::Equals(function + ": VK_ERROR_UNKNOWN"));
}

TEST_CASE("Vulkan doesn't ask NVIDIA's driver about a headless window's swapchain on a loader that lets it crash",
          "[vulkan][caps][gpu]") {
    const LoaderVersion loader{"1.4.318"};
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv nvidia("BFR_ADAPTER_DRIVER_ID", std::to_string(VK_DRIVER_ID_NVIDIA_PROPRIETARY));
    const auto rhi = Instance::create(vulkanDesc());
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());

    const auto caps = rhi->swapchainCaps(cpuAdapter(*rhi), {.window = kHeadless});

    REQUIRE_FALSE(caps.has_value());
    CHECK(caps.error().code == Result::ErrorPresentUnsupported);
    CHECK_THAT(caps.error().message.text(), Catch::Matchers::ContainsSubstring("can't present to this window"));
    CHECK_THAT(caps.error().message.text(), !Catch::Matchers::ContainsSubstring("loader"));
}

TEST_CASE("Vulkan device without VK_EXT_memory_budget fails with RhiUnsupported", "[vulkan][caps][gpu]") {
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv hidden("BFR_ADAPTER_HIDE", "VK_EXT_memory_budget");
    const auto rhi = Instance::create(vulkanDesc());
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());
    const auto device = rhi->createDevice(testDeviceOn(kFirstAdapter, kNoWindow));
    INFO(failureOf(device));
    REQUIRE(device.has_value());

    const auto budget = device->budget();

    REQUIRE_FALSE(budget.has_value());
    CHECK(budget.error().code == Result::ErrorUnsupported);
    CHECK_THAT(budget.error().message.text(), Catch::Matchers::ContainsSubstring("VK_EXT_memory_budget"));
}
