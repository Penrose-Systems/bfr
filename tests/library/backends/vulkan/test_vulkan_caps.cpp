// What the Vulkan backend reports about its adapters: what the driver says,
// and how each failed query comes back.

#include "bfr/core/error.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/rhi.h"
#include "rhi/rhi_test_support.h"
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
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using namespace bfr::rhi;
using namespace bfr::rhi::vulkan::test;

namespace {

// Each thing Vulkan promises about an adapter's info that info lacks.
std::vector<std::string> missingVulkanInfo(const AdapterInfo& info) {
    std::vector<std::string> missing;
    if (info.apiMajor != 1 || info.apiMinor < 3) {
        missing.emplace_back("Vulkan 1.3");
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

TEST_CASE("Vulkan lists only adapters at 1.3 or newer, each with a vendor, a driver name and a device UUID",
          "[vulkan][caps][gpu]") {
    std::vector<std::string> messages;
    const auto rhi = Rhi::create(recordingInto(messages), kValidation);
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());

    for (const Adapter& adapter : (*rhi)->adapters()) {
        INFO(adapter.name);
        CHECK_THAT(missingVulkanInfo(adapter.caps.info), Catch::Matchers::IsEmpty());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Vulkan reports what the driver says about an adapter", "[vulkan][caps][gpu]") {
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv vendor("BFR_ADAPTER_VENDOR_ID", "0x1234");
    const ScopedEnv noFloat16("BFR_ADAPTER_NO_FEATURE", "shaderFloat16");

    const auto rhi = Rhi::create(kVulkan, {});

    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());
    for (const Adapter& adapter : (*rhi)->adapters()) {
        INFO(adapter.name);
        CHECK(adapter.caps.info.vendorId == 0x1234);
        CHECK_FALSE(adapter.caps.features.shaderFloat16);
    }
}

TEST_CASE("Vulkan reports no sample counts for a format the API says it can't render to", "[vulkan][caps][gpu]") {
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv failing("BFR_FAIL", "vkGetPhysicalDeviceImageFormatProperties2");
    const ScopedEnv unsupported("BFR_FAIL_RESULT", "VK_ERROR_FORMAT_NOT_SUPPORTED");
    const auto rhi = Rhi::create(kVulkan, {});
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());

    const auto rgba8 = (*rhi)->formatCaps(0, Format::Rgba8Unorm);

    INFO(failureOf(rgba8));
    REQUIRE(rgba8.has_value());
    CHECK(rgba8->colorAttachment);
    CHECK(rgba8->sampleCounts == 0);
}

TEST_CASE("Vulkan reports a failed format query as the failure it is", "[vulkan][caps][gpu]") {
    const auto [name, code] = GENERATE(std::pair{"VK_ERROR_UNKNOWN", bfr::core::ErrorCode::RhiQueryFailed},
                                       std::pair{"VK_ERROR_OUT_OF_HOST_MEMORY", bfr::core::ErrorCode::RhiOutOfMemory});
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv failing("BFR_FAIL", "vkGetPhysicalDeviceImageFormatProperties2");
    const ScopedEnv result("BFR_FAIL_RESULT", name);
    const auto rhi = Rhi::create(kVulkan, {});
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());

    const auto rgba8 = (*rhi)->formatCaps(0, Format::Rgba8Unorm);

    INFO(name);
    REQUIRE_FALSE(rgba8.has_value());
    CHECK(rgba8.error().code == code);
    CHECK_THAT(rgba8.error().message,
               Catch::Matchers::Equals(std::string{"vkGetPhysicalDeviceImageFormatProperties2: "} + name));
}

TEST_CASE("Vulkan reports a failed surface query as the failure it is", "[vulkan][caps][gpu]") {
    const std::string function = GENERATE("vkGetPhysicalDeviceSurfaceCapabilitiesKHR",
                                          "vkGetPhysicalDeviceSurfaceFormatsKHR");
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv failing("BFR_FAIL", function);
    const auto rhi = Rhi::create(kVulkan, {});
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());
    const std::size_t adapter = cpuAdapter(**rhi);

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

TEST_CASE("Vulkan doesn't ask NVIDIA's driver about a headless window's swapchain on a loader that lets it crash",
          "[vulkan][caps][gpu]") {
    const LoaderVersion loader{"1.4.318"};
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv nvidia("BFR_ADAPTER_DRIVER_ID", std::to_string(VK_DRIVER_ID_NVIDIA_PROPRIETARY));
    const auto rhi = Rhi::create(kVulkan, {});
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());

    const auto caps = (*rhi)->surfaceCaps(cpuAdapter(**rhi), bfr::core::window::Headless{});

    REQUIRE_FALSE(caps.has_value());
    CHECK(caps.error().code == bfr::core::ErrorCode::RhiPresentUnsupported);
    CHECK_THAT(caps.error().message, Catch::Matchers::ContainsSubstring("can't present to this window"));
    CHECK_THAT(caps.error().message, !Catch::Matchers::ContainsSubstring("loader"));
}

TEST_CASE("Vulkan device without VK_EXT_memory_budget fails with RhiUnsupported", "[vulkan][caps][gpu]") {
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv hidden("BFR_ADAPTER_HIDE", "VK_EXT_memory_budget");
    const auto rhi = Rhi::create(kVulkan, {});
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
