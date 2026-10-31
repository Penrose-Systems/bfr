// What the Vulkan backend's Device does when a swapchain call fails, and how
// it picks the queue that presents.

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/window.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/instance.h"
#include "bfr/rhi/rhi_config.h"
#include "bfr/rhi/swapchain.h"
#include "rhi/rhi_test_support.h"
#include "scoped_env.h"
#include "vulkan_test_support.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_container_properties.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace bfr::rhi;
using namespace bfr::rhi::vulkan::test;

namespace {

constexpr std::uint32_t kWidth = 64;
constexpr std::uint32_t kHeight = 48;

// A headless window: the swapchain decides its size.
const bfr::core::window::NativeWindow kHeadless = bfr::core::window::Headless{};

// A description on the Vulkan backend that records errors in errors and
// writes every message from info up to log, one a line.
InstanceDesc errorsAndLogInto(std::vector<std::string>& errors, std::ostringstream& log) {
    return {.backend = Backend::Vulkan,
            .logger = {[&errors, &log](bfr::core::Severity severity, std::string_view message) {
                           if (severity == bfr::core::Severity::Error) {
                               errors.emplace_back(message);
                           }
                           log << message << '\n';
                       },
                       bfr::core::Severity::Info}};
}

// An Instance made with desc, whose recording outlives it, and a device that
// presents to a headless window.
struct HeadlessDevice {
    explicit HeadlessDevice(const InstanceDesc& desc, const RhiConfig& config = kValidation) {
        auto created = Instance::create(desc, config);
        INFO(failureOf(created));
        REQUIRE(created.has_value());
        instance = std::move(*created);
        device = presentingDevice(*instance, kHeadless);
    }

    // The device goes before its Instance.
    std::unique_ptr<Instance> instance;
    std::unique_ptr<Device> device;
};

} // namespace

TEST_CASE("Vulkan device fails with RhiWindowFailed when the API refuses a swapchain's surface", "[vulkan][gpu]") {
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv refused("BFR_FAIL", "vkCreateHeadlessSurfaceEXT");
    const auto rhi = Instance::create(kVulkan, {});
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());
    const auto device = (*rhi)->createDevice(0, kNoWindow, {});
    INFO(failureOf(device));
    REQUIRE(device.has_value());

    // The desc is never used: making the swapchain fails first.
    const auto swapchain = (*device)->createSwapchain(kHeadless, SwapchainDesc{});

    REQUIRE_FALSE(swapchain.has_value());
    CHECK(swapchain.error().code == bfr::core::ErrorCode::RhiWindowFailed);
    CHECK_THAT(swapchain.error().message,
               Catch::Matchers::ContainsSubstring("vkCreateHeadlessSurfaceEXT: VK_ERROR_UNKNOWN"));
}

TEST_CASE("Vulkan device fails with RhiPresentUnsupported when it can't present to the swapchain's window",
          "[vulkan][gpu]") {
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv unsupported("BFR_PRESENT_UNSUPPORTED", "1");
    const auto rhi = Instance::create(kVulkan, {});
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());
    const auto device = (*rhi)->createDevice(cpuAdapter(**rhi), kNoWindow, {});
    INFO(failureOf(device));
    REQUIRE(device.has_value());

    // The desc is never used: making the swapchain fails first.
    const auto swapchain = (*device)->createSwapchain(kHeadless, SwapchainDesc{});

    REQUIRE_FALSE(swapchain.has_value());
    CHECK(swapchain.error().code == bfr::core::ErrorCode::RhiPresentUnsupported);
}

// Validation sits above the failing call, so it checks what the RHI does with
// the failure and with the junk the call leaves.
TEST_CASE("Vulkan device reports a failed call when building a swapchain's images and leaves nothing behind",
          "[vulkan][gpu]") {
    // Building the images makes the rest of the calls.
    const std::string function = GENERATE("vkCreateSwapchainKHR", "vkGetSwapchainImagesKHR");
    INFO(function);
    const AdapterOverrideLayer adapterOverride{"VK_LAYER_KHRONOS_validation"};
    const ScopedEnv failing("BFR_FAIL", function);
    std::vector<std::string> errors;
    {
        const HeadlessDevice headless{recordingErrorsInto(errors)};
        const auto swapchain = headless.device->createSwapchain(kHeadless, swapchainDesc(*headless.instance));
        INFO(failureOf(swapchain));
        REQUIRE(swapchain.has_value());

        const auto images = headless.device->resizeSwapchain(*swapchain, kWidth, kHeight);

        REQUIRE_FALSE(images.has_value());
        CHECK(images.error().code == bfr::core::ErrorCode::RhiSwapchainFailed);
        CHECK_THAT(images.error().message, Catch::Matchers::ContainsSubstring(function + ": VK_ERROR_UNKNOWN"));
    }
    CHECK_THAT(errors, Catch::Matchers::IsEmpty());
}

TEST_CASE("Vulkan device reports a failed present query as the failure it is", "[vulkan][gpu]") {
    // The failing query leaves VK_TRUE behind, which must not be believed.
    const AdapterOverrideLayer adapterOverride{"VK_LAYER_KHRONOS_validation"};
    const ScopedEnv failing("BFR_FAIL", "vkGetPhysicalDeviceSurfaceSupportKHR");
    std::vector<std::string> errors;
    {
        const auto rhi = Instance::create(recordingErrorsInto(errors), kValidation);
        INFO(failureOf(rhi));
        REQUIRE(rhi.has_value());
        const auto device = (*rhi)->createDevice(cpuAdapter(**rhi), kNoWindow, {});
        INFO(failureOf(device));
        REQUIRE(device.has_value());

        // The desc is never used: making the swapchain fails first.
        const auto swapchain = (*device)->createSwapchain(kHeadless, SwapchainDesc{});

        REQUIRE_FALSE(swapchain.has_value());
        CHECK(swapchain.error().code == bfr::core::ErrorCode::RhiWindowFailed);
        CHECK_THAT(swapchain.error().message,
                   Catch::Matchers::ContainsSubstring("vkGetPhysicalDeviceSurfaceSupportKHR: VK_ERROR_UNKNOWN"));
    }
    CHECK_THAT(errors, Catch::Matchers::IsEmpty());
}

TEST_CASE("Vulkan device keeps running out of memory, a lost device and a lost surface apart from other failures",
          "[vulkan][gpu]") {
    using bfr::core::ErrorCode;
    const auto [result, code] = GENERATE(std::pair{"VK_ERROR_OUT_OF_HOST_MEMORY", ErrorCode::RhiOutOfMemory},
                                         std::pair{"VK_ERROR_DEVICE_LOST", ErrorCode::RhiDeviceLost},
                                         std::pair{"VK_ERROR_SURFACE_LOST_KHR", ErrorCode::RhiWindowLost});
    INFO(result);
    const AdapterOverrideLayer adapterOverride;
    const HeadlessDevice headless{kVulkan, {}};
    const auto swapchain = headless.device->createSwapchain(kHeadless, swapchainDesc(*headless.instance));
    INFO(failureOf(swapchain));
    REQUIRE(swapchain.has_value());
    const ScopedEnv failing("BFR_FAIL", "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
    const ScopedEnv failingWith("BFR_FAIL_RESULT", result);

    const auto images = headless.device->resizeSwapchain(*swapchain, kWidth, kHeight);

    REQUIRE_FALSE(images.has_value());
    CHECK(images.error().code == code);
    CHECK_THAT(images.error().message,
               Catch::Matchers::ContainsSubstring(std::string{"vkGetPhysicalDeviceSurfaceCapabilitiesKHR: "} + result));
    CHECK(headless.device->destroySwapchain(*swapchain).has_value());
}

TEST_CASE("Vulkan device builds a swapchain once a call the API refused succeeds again", "[vulkan][gpu]") {
    const std::string function = GENERATE("vkGetPhysicalDeviceSurfaceCapabilitiesKHR", "vkCreateSwapchainKHR",
                                          "vkGetSwapchainImagesKHR");
    INFO(function);
    const AdapterOverrideLayer adapterOverride{"VK_LAYER_KHRONOS_validation"};
    std::vector<std::string> errors;
    {
        const HeadlessDevice headless{recordingErrorsInto(errors)};
        const auto swapchain = headless.device->createSwapchain(kHeadless, swapchainDesc(*headless.instance));
        INFO(failureOf(swapchain));
        REQUIRE(swapchain.has_value());
        {
            const ScopedEnv failing("BFR_FAIL", function);
            REQUIRE_FALSE(headless.device->resizeSwapchain(*swapchain, kWidth, kHeight).has_value());
        }

        const auto images = headless.device->resizeSwapchain(*swapchain, kWidth, kHeight);

        INFO(failureOf(images));
        REQUIRE(images.has_value());
        CHECK(images->width == kWidth);
        CHECK(images->height == kHeight);
    }
    CHECK_THAT(errors, Catch::Matchers::IsEmpty());
}

TEST_CASE("Vulkan device presents from a queue family of its own when no graphics family presents", "[vulkan][gpu]") {
    // lavapipe also lists family 1, which does nothing but present, and is the
    // only family that presents.
    const AdapterOverrideLayer adapterOverride{"VK_LAYER_KHRONOS_validation"};
    const ScopedEnv separate("BFR_ADAPTER_SEPARATE_PRESENT", "1");
    std::vector<std::string> errors;
    std::ostringstream log;
    {
        const auto rhi = Instance::create(errorsAndLogInto(errors, log), kValidation);
        INFO(failureOf(rhi));
        REQUIRE(rhi.has_value());
        const auto device = (*rhi)->createDevice(cpuAdapter(**rhi), kHeadless, {});
        INFO(failureOf(device));
        REQUIRE(device.has_value());
        const auto swapchain = (*device)->createSwapchain(kHeadless, swapchainDesc(**rhi));
        INFO(failureOf(swapchain));
        REQUIRE(swapchain.has_value());

        const auto images = (*device)->resizeSwapchain(*swapchain, kWidth, kHeight);

        INFO(failureOf(images));
        CHECK(images.has_value());
    }
    CHECK_THAT(errors, Catch::Matchers::IsEmpty());
    if (bfr::core::kLoggingEnabled) {
        CHECK_THAT(log.str(), Catch::Matchers::ContainsSubstring(", queue families 0, 1, extensions"));
    }
}
