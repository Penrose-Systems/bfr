// What the Vulkan backend's Device does when a swapchain call fails, and how
// it picks the queue that presents.

#include "bfr.hpp"
#include "bfr/bfr.h"
#include "rhi/bfr_test_support.h"
#include "scoped_env.h"
#include "vulkan_test_support.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_container_properties.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace bfr;
using namespace bfr::test;
using namespace bfr::test::vulkan;

namespace {

constexpr std::uint32_t kWidth = 64;
constexpr std::uint32_t kHeight = 48;

} // namespace

TEST_CASE("Vulkan device fails with RhiWindowFailed when the API refuses a swapchain's surface", "[vulkan][gpu]") {
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv refused("BFR_FAIL", "vkCreateHeadlessSurfaceEXT");
    const auto rhi = Instance::create(vulkanDesc());
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());
    const auto device = rhi->createDevice(
        testDeviceWith(*rhi, kFirstAdapter, kNoWindow, [](DeviceDesc& desc) { desc.features.swapchain = true; }));
    INFO(failureOf(device));
    REQUIRE(device.has_value());

    // The desc is never used: making the swapchain fails first.
    const auto swapchain = device->createSwapchain(kTestSwapchain);

    REQUIRE_FALSE(swapchain.has_value());
    CHECK(swapchain.error().code == Result::ErrorWindowFailed);
    CHECK_THAT(swapchain.error().message.text(),
               Catch::Matchers::ContainsSubstring("vkCreateHeadlessSurfaceEXT: VK_ERROR_UNKNOWN"));
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
        auto swapchain = headless.device.createSwapchain(swapchainDesc(headless.instance));
        INFO(failureOf(swapchain));
        REQUIRE(swapchain.has_value());

        const auto images = swapchain->resize(kWidth, kHeight);

        REQUIRE_FALSE(images.has_value());
        CHECK(images.error().code == Result::ErrorSwapchainFailed);
        CHECK_THAT(images.error().message.text(), Catch::Matchers::ContainsSubstring(function + ": VK_ERROR_UNKNOWN"));
    }
    CHECK_THAT(errors, Catch::Matchers::IsEmpty());
}

TEST_CASE("Vulkan device keeps running out of memory, a lost device and a lost surface apart from other failures",
          "[vulkan][gpu]") {
    const auto [result, code] = GENERATE(std::pair{"VK_ERROR_OUT_OF_HOST_MEMORY", Result::ErrorOutOfMemory},
                                         std::pair{"VK_ERROR_DEVICE_LOST", Result::ErrorDeviceLost},
                                         std::pair{"VK_ERROR_SURFACE_LOST_KHR", Result::ErrorWindowLost});
    INFO(result);
    const AdapterOverrideLayer adapterOverride;
    const HeadlessDevice headless{vulkanDesc(), kValidationOff};
    auto swapchain = headless.device.createSwapchain(swapchainDesc(headless.instance));
    INFO(failureOf(swapchain));
    REQUIRE(swapchain.has_value());
    const ScopedEnv failing("BFR_FAIL", "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
    const ScopedEnv failingWith("BFR_FAIL_RESULT", result);

    const auto images = swapchain->resize(kWidth, kHeight);

    REQUIRE_FALSE(images.has_value());
    CHECK(images.error().code == code);
    CHECK_THAT(images.error().message.text(),
               Catch::Matchers::ContainsSubstring(std::string{"vkGetPhysicalDeviceSurfaceCapabilitiesKHR: "} + result));
    const bfr_swapchain swapchainHandle = swapchain->get();
    *swapchain = {};
    CHECK(resizeResult(headless.device, swapchainHandle, kWidth, kHeight) == Result::ErrorInvalidHandle);
}

TEST_CASE("Vulkan device builds a swapchain once a call the API refused succeeds again", "[vulkan][gpu]") {
    const std::string function = GENERATE("vkGetPhysicalDeviceSurfaceCapabilitiesKHR", "vkCreateSwapchainKHR",
                                          "vkGetSwapchainImagesKHR");
    INFO(function);
    const AdapterOverrideLayer adapterOverride{"VK_LAYER_KHRONOS_validation"};
    std::vector<std::string> errors;
    {
        const HeadlessDevice headless{recordingErrorsInto(errors)};
        auto swapchain = headless.device.createSwapchain(swapchainDesc(headless.instance));
        INFO(failureOf(swapchain));
        REQUIRE(swapchain.has_value());
        {
            const ScopedEnv failing("BFR_FAIL", function);
            REQUIRE_FALSE(swapchain->resize(kWidth, kHeight).has_value());
        }

        const auto images = swapchain->resize(kWidth, kHeight);

        INFO(failureOf(images));
        REQUIRE(images.has_value());
        CHECK(images->width == kWidth);
        CHECK(images->height == kHeight);
    }
    CHECK_THAT(errors, Catch::Matchers::IsEmpty());
}

TEST_CASE("Vulkan device passes acquire's timeout and gives VK_TIMEOUT as RhiTimeout", "[vulkan][gpu]") {
    constexpr std::uint64_t kNoWait = 0;
    const AdapterOverrideLayer adapterOverride{"VK_LAYER_KHRONOS_validation"};
    std::vector<std::string> errors;
    {
        const HeadlessDevice headless{recordingErrorsInto(errors)};
        const Device& device = headless.device;
        auto swapchain = device.createSwapchain(swapchainDesc(headless.instance));
        REQUIRE(swapchain.has_value());
        REQUIRE(swapchain->resize(kWidth, kHeight).has_value());
        {
            const ScopedEnv timeout("BFR_ACQUIRE_TIMEOUT", "1");
            const auto timedOut = swapchain->acquire(kNoWait);
            REQUIRE_FALSE(timedOut.has_value());
            CHECK(timedOut.error().code == Result::ErrorTimeout);
        }
        // Nothing was acquired, so the next acquire takes an image.
        CHECK(swapchain->acquire(kTestWait).has_value());
        const bfr_swapchain swapchainHandle = swapchain->get();
        *swapchain = {};
        CHECK(resizeResult(device, swapchainHandle, kWidth, kHeight) == Result::ErrorInvalidHandle);
    }
    CHECK_THAT(errors, Catch::Matchers::IsEmpty());
}

TEST_CASE("Vulkan device gives VK_SUBOPTIMAL_KHR from acquire and present as Suboptimal", "[vulkan][gpu]") {
    const AdapterOverrideLayer adapterOverride{"VK_LAYER_KHRONOS_validation"};
    std::vector<std::string> errors;
    {
        const HeadlessDevice headless{recordingErrorsInto(errors)};
        const Device& device = headless.device;
        auto swapchain = device.createSwapchain(swapchainDesc(headless.instance));
        REQUIRE(swapchain.has_value());
        REQUIRE(swapchain->resize(kWidth, kHeight).has_value());
        auto pool = device.createCommandPool({.queue = 0, .listCount = 2});
        REQUIRE(pool.has_value());
        // One frame: acquire, clear the image, submit, present.
        const auto frame = [&]() -> std::pair<SwapchainStatus, SwapchainStatus> {
            const auto acquired = swapchain->acquire(kTestWait);
            REQUIRE(acquired.has_value());
            const auto list = pool->beginCommands();
            REQUIRE(list.has_value());
            RenderingDesc clear{.colorAttachmentCount = 1,
                                .colorAttachments = {},
                                .depthStencil = {},
                                .occlusionQueries = {}};
            clear.colorAttachments[0] = {.texture = {}, .loadOp = LoadOp::Clear, .storeOp = StoreOp::Store};
            REQUIRE(device.beginRendering(*list, *swapchain, clear).has_value());
            REQUIRE(device.endRendering(*list).has_value());
            const auto sent = device.submit({.queue = 0, .commandLists = std::span{&*list, 1}, .waits = {}});
            REQUIRE(sent.has_value());
            const auto presented = swapchain->present();
            REQUIRE(presented.has_value());
            REQUIRE(device.wait(*sent, kTestWait).has_value());
            REQUIRE(pool->reset().has_value());
            return {*acquired, *presented};
        };

        CHECK(frame() == std::pair{SwapchainStatus::Matches, SwapchainStatus::Matches});
        const ScopedEnv suboptimal("BFR_SUBOPTIMAL", "1");
        CHECK(frame() == std::pair{SwapchainStatus::Suboptimal, SwapchainStatus::Suboptimal});
        const bfr_command_pool poolHandle = pool->get();
        *pool = {};
        CHECK(resetResult(device, poolHandle) == Result::ErrorInvalidHandle);
        const bfr_swapchain swapchainHandle = swapchain->get();
        *swapchain = {};
        CHECK(resizeResult(device, swapchainHandle, kWidth, kHeight) == Result::ErrorInvalidHandle);
    }
    CHECK_THAT(errors, Catch::Matchers::IsEmpty());
}
