#include "device.h"
#include "error.h"
#include "log.h"
#include "log_capture.h"
#include "platform/display_test_support.h"
#include "rhi.h"
#include "rhi_config.h"
#include "rhi_test_support.h"
#include "scoped_env.h"
#include "swapchain.h"
#include "window.h"

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
#include <utility>
#include <vector>

using namespace bfr::rhi;
using namespace bfr::rhi::test;

namespace {

constexpr std::uint32_t kWidth = 64;
constexpr std::uint32_t kHeight = 48;

// A headless window: the swapchain decides its size.
const bfr::core::window::NativeWindow kHeadless = bfr::core::window::Headless{};

// An Rhi made with desc, whose recording outlives it, and a device that
// presents to a headless window.
struct HeadlessDevice {
    explicit HeadlessDevice(const RhiDesc& desc, const RhiConfig& config = kValidation) {
        auto created = Rhi::create(desc, config);
        INFO(failureOf(created));
        REQUIRE(created.has_value());
        rhi = std::move(*created);
        device = presentingDevice(*rhi, kHeadless);
    }

    // The device goes before its Rhi.
    std::unique_ptr<Rhi> rhi;
    std::unique_ptr<Device> device;
};

} // namespace

TEST_CASE("Device builds a swapchain at the window's size", "[rhi][gpu]") {
    const auto display = bfr::platform::test::openDisplay();
    std::vector<std::string> messages;
    {
        const auto rhi = Rhi::create(recordingInto(messages), kValidation);
        INFO(failureOf(rhi));
        REQUIRE(rhi.has_value());
        const auto device = presentingDevice(**rhi, display->nativeWindow());
        const auto swapchain = device->createSwapchain(display->nativeWindow(), kSwapchainImages);
        INFO(failureOf(swapchain));
        REQUIRE(swapchain.has_value());

        const auto images = device->resizeSwapchain(*swapchain, display->width(), display->height());

        INFO(failureOf(images));
        REQUIRE(images.has_value());
        CHECK(images->width == display->width());
        CHECK(images->height == display->height());
        CHECK(images->count >= 2);
        CHECK(device->destroySwapchain(*swapchain).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device rebuilds a swapchain at a new size", "[rhi][gpu]") {
    std::vector<std::string> messages;
    {
        const HeadlessDevice headless{recordingInto(messages)};
        const auto swapchain = headless.device->createSwapchain(kHeadless, kSwapchainImages);
        INFO(failureOf(swapchain));
        REQUIRE(swapchain.has_value());
        const auto first = headless.device->resizeSwapchain(*swapchain, kWidth, kHeight);
        INFO(failureOf(first));
        REQUIRE(first.has_value());

        const auto images = headless.device->resizeSwapchain(*swapchain, kWidth / 2, kHeight / 2);

        INFO(failureOf(images));
        REQUIRE(images.has_value());
        CHECK(images->width == kWidth / 2);
        CHECK(images->height == kHeight / 2);
        CHECK(headless.device->destroySwapchain(*swapchain).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device destroys the swapchains left when it goes", "[rhi][gpu]") {
    std::vector<std::string> messages;
    {
        const HeadlessDevice headless{recordingInto(messages)};
        const auto swapchain = headless.device->createSwapchain(kHeadless, kSwapchainImages);
        INFO(failureOf(swapchain));
        REQUIRE(swapchain.has_value());
        const auto images = headless.device->resizeSwapchain(*swapchain, kWidth, kHeight);
        INFO(failureOf(images));
        REQUIRE(images.has_value());
    }

    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device empties a swapchain for a window with no area", "[rhi][gpu]") {
    std::vector<std::string> messages;
    {
        const HeadlessDevice headless{recordingInto(messages)};
        const auto swapchain = headless.device->createSwapchain(kHeadless, kSwapchainImages);
        INFO(failureOf(swapchain));
        REQUIRE(swapchain.has_value());
        const auto built = headless.device->resizeSwapchain(*swapchain, kWidth, kHeight);
        INFO(failureOf(built));
        REQUIRE(built.has_value());

        const auto images = headless.device->resizeSwapchain(*swapchain, 0, kHeight);

        INFO(failureOf(images));
        REQUIRE(images.has_value());
        CHECK(images->width == 0);
        CHECK(images->height == 0);
        CHECK(images->count == 0);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device fails with RhiInvalidHandle on a destroyed swapchain", "[rhi][gpu]") {
    std::vector<std::string> messages;
    const HeadlessDevice headless{recordingInto(messages)};
    const auto swapchain = headless.device->createSwapchain(kHeadless, kSwapchainImages);
    INFO(failureOf(swapchain));
    REQUIRE(swapchain.has_value());
    REQUIRE(headless.device->destroySwapchain(*swapchain).has_value());

    const auto resized = headless.device->resizeSwapchain(*swapchain, kWidth, kHeight);
    const auto again = headless.device->destroySwapchain(*swapchain);

    REQUIRE_FALSE(resized.has_value());
    CHECK(resized.error().code == bfr::core::ErrorCode::RhiInvalidHandle);
    REQUIRE_FALSE(again.has_value());
    CHECK(again.error().code == bfr::core::ErrorCode::RhiInvalidHandle);
}

TEST_CASE("Device destroying a null swapchain does nothing", "[rhi][gpu]") {
    std::vector<std::string> messages;
    const HeadlessDevice headless{recordingInto(messages)};

    CHECK(headless.device->destroySwapchain(SwapchainHandle{}).has_value());
}

TEST_CASE("Device fails with RhiSurfaceFailed for a swapchain without a window", "[rhi][gpu]") {
    std::vector<std::string> messages;
    const HeadlessDevice headless{recordingInto(messages)};

    const auto swapchain = headless.device->createSwapchain(kNoWindow, kSwapchainImages);

    REQUIRE_FALSE(swapchain.has_value());
    CHECK(swapchain.error().code == bfr::core::ErrorCode::RhiSurfaceFailed);
    CHECK_THAT(swapchain.error().message, Catch::Matchers::ContainsSubstring("no window"));
}

TEST_CASE("Device fails with RhiSurfaceFailed when the API refuses a swapchain's surface", "[rhi][gpu]") {
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv refused("BFR_FAIL", "vkCreateHeadlessSurfaceEXT");
    const auto rhi = Rhi::create({}, {});
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());
    const auto device = (*rhi)->createDevice(0, kNoWindow);
    INFO(failureOf(device));
    REQUIRE(device.has_value());

    const auto swapchain = (*device)->createSwapchain(kHeadless, kSwapchainImages);

    REQUIRE_FALSE(swapchain.has_value());
    CHECK(swapchain.error().code == bfr::core::ErrorCode::RhiSurfaceFailed);
    CHECK_THAT(swapchain.error().message,
               Catch::Matchers::ContainsSubstring("vkCreateHeadlessSurfaceEXT: VK_ERROR_OUT_OF_HOST_MEMORY"));
}

TEST_CASE("Device fails with RhiPresentUnsupported when it can't present to the swapchain's window", "[rhi][gpu]") {
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv unsupported("BFR_PRESENT_UNSUPPORTED", "1");
    const auto rhi = Rhi::create({}, {});
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());
    const auto device = (*rhi)->createDevice(headlessAdapter(**rhi), kNoWindow);
    INFO(failureOf(device));
    REQUIRE(device.has_value());
    const auto swapchain = (*device)->createSwapchain(kHeadless, kSwapchainImages);
    INFO(failureOf(swapchain));
    REQUIRE(swapchain.has_value());

    const auto images = (*device)->resizeSwapchain(*swapchain, kWidth, kHeight);

    REQUIRE_FALSE(images.has_value());
    CHECK(images.error().code == bfr::core::ErrorCode::RhiPresentUnsupported);
}

TEST_CASE("Device reports each swapchain call the API fails and leaves nothing behind", "[rhi][gpu]") {
    const std::string function = GENERATE("vkGetPhysicalDeviceSurfaceCapabilitiesKHR",
                                          "vkGetPhysicalDeviceSurfaceFormatsKHR", "vkCreateSwapchainKHR",
                                          "vkGetSwapchainImagesKHR");
    INFO(function);
    // Validation sits above the failing call, so it checks what the RHI does
    // with the failure and with the junk the call leaves.
    const AdapterOverrideLayer adapterOverride{"VK_LAYER_KHRONOS_validation"};
    const ScopedEnv failing("BFR_FAIL", function);
    std::vector<std::string> errors;
    {
        const HeadlessDevice headless{recordingErrorsInto(errors)};
        const auto swapchain = headless.device->createSwapchain(kHeadless, kSwapchainImages);
        INFO(failureOf(swapchain));
        REQUIRE(swapchain.has_value());

        const auto images = headless.device->resizeSwapchain(*swapchain, kWidth, kHeight);

        REQUIRE_FALSE(images.has_value());
        CHECK(images.error().code == bfr::core::ErrorCode::RhiSwapchainFailed);
        CHECK_THAT(images.error().message,
                   Catch::Matchers::ContainsSubstring(function + ": VK_ERROR_OUT_OF_HOST_MEMORY"));
    }
    CHECK_THAT(errors, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device fails with RhiSwapchainFailed for a surface that lists no formats", "[rhi][gpu]") {
    const AdapterOverrideLayer adapterOverride{"VK_LAYER_KHRONOS_validation"};
    const ScopedEnv noFormats("BFR_SURFACE_NO_FORMATS", "1");
    std::vector<std::string> errors;
    {
        const HeadlessDevice headless{recordingErrorsInto(errors)};
        const auto swapchain = headless.device->createSwapchain(kHeadless, kSwapchainImages);
        INFO(failureOf(swapchain));
        REQUIRE(swapchain.has_value());

        const auto images = headless.device->resizeSwapchain(*swapchain, kWidth, kHeight);

        REQUIRE_FALSE(images.has_value());
        CHECK(images.error().code == bfr::core::ErrorCode::RhiSwapchainFailed);
        CHECK_THAT(images.error().message, Catch::Matchers::ContainsSubstring("no formats"));
    }
    CHECK_THAT(errors, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device treats a failed present query as no and empties the swapchain", "[rhi][gpu]") {
    // The failing query leaves VK_TRUE behind, which must not be believed.
    const AdapterOverrideLayer adapterOverride{"VK_LAYER_KHRONOS_validation"};
    const ScopedEnv failing("BFR_FAIL", "vkGetPhysicalDeviceSurfaceSupportKHR");
    std::vector<std::string> errors;
    {
        const auto rhi = Rhi::create(recordingErrorsInto(errors), kValidation);
        INFO(failureOf(rhi));
        REQUIRE(rhi.has_value());
        const auto device = (*rhi)->createDevice(headlessAdapter(**rhi), kNoWindow);
        INFO(failureOf(device));
        REQUIRE(device.has_value());
        const auto swapchain = (*device)->createSwapchain(kHeadless, kSwapchainImages);
        INFO(failureOf(swapchain));
        REQUIRE(swapchain.has_value());

        const auto images = (*device)->resizeSwapchain(*swapchain, kWidth, kHeight);

        REQUIRE_FALSE(images.has_value());
        CHECK(images.error().code == bfr::core::ErrorCode::RhiPresentUnsupported);
    }
    CHECK_THAT(errors, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device fails with RhiInvalidHandle for a forged swapchain handle", "[rhi][gpu]") {
    // Past the pool's capacity; an unused slot with a live generation; the used
    // slot with a generation it never had; the used slot with a free one.
    const SwapchainHandle forged = GENERATE(SwapchainHandle{.index = 100000, .generation = 1},
                                            SwapchainHandle{.index = 3, .generation = 1},
                                            SwapchainHandle{.index = 0, .generation = 7},
                                            SwapchainHandle{.index = 0, .generation = 2});
    INFO("index " << forged.index << ", generation " << forged.generation);
    std::vector<std::string> messages;
    const HeadlessDevice headless{recordingInto(messages)};
    const auto swapchain = headless.device->createSwapchain(kHeadless, kSwapchainImages);
    INFO(failureOf(swapchain));
    REQUIRE(swapchain.has_value());
    REQUIRE(swapchain->index == 0);
    REQUIRE(swapchain->generation == 1);

    const auto resized = headless.device->resizeSwapchain(forged, kWidth, kHeight);
    const auto destroyed = headless.device->destroySwapchain(forged);

    REQUIRE_FALSE(resized.has_value());
    CHECK(resized.error().code == bfr::core::ErrorCode::RhiInvalidHandle);
    REQUIRE_FALSE(destroyed.has_value());
    CHECK(destroyed.error().code == bfr::core::ErrorCode::RhiInvalidHandle);
}

TEST_CASE("Device builds the image count its swapchain was created with", "[rhi][gpu]") {
    // More than the default and than lavapipe's minimum.
    constexpr std::uint32_t kImages = 6;
    std::vector<std::string> messages;
    {
        const HeadlessDevice headless{recordingInto(messages), kValidation};
        const auto swapchain = headless.device->createSwapchain(kHeadless, kImages);
        INFO(failureOf(swapchain));
        REQUIRE(swapchain.has_value());

        const auto images = headless.device->resizeSwapchain(*swapchain, kWidth, kHeight);

        INFO(failureOf(images));
        REQUIRE(images.has_value());
        CHECK(images->count >= kImages);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device builds a swapchain once a call the API refused succeeds again", "[rhi][gpu]") {
    const std::string function = GENERATE("vkGetPhysicalDeviceSurfaceCapabilitiesKHR",
                                          "vkGetPhysicalDeviceSurfaceFormatsKHR", "vkCreateSwapchainKHR",
                                          "vkGetSwapchainImagesKHR");
    INFO(function);
    const AdapterOverrideLayer adapterOverride{"VK_LAYER_KHRONOS_validation"};
    std::vector<std::string> errors;
    {
        const HeadlessDevice headless{recordingErrorsInto(errors)};
        const auto swapchain = headless.device->createSwapchain(kHeadless, kSwapchainImages);
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

TEST_CASE("Device presents from a queue family of its own when no graphics family presents", "[rhi][gpu]") {
    // lavapipe also lists family 1, which does nothing but present, and is the
    // only family that presents.
    const AdapterOverrideLayer adapterOverride{"VK_LAYER_KHRONOS_validation"};
    const ScopedEnv separate("BFR_ADAPTER_SEPARATE_PRESENT", "1");
    std::vector<std::string> errors;
    std::ostringstream log;
    captureLogInto(log);
    {
        const auto rhi = Rhi::create(recordingErrorsInto(errors), kValidation);
        INFO(failureOf(rhi));
        REQUIRE(rhi.has_value());
        const auto device = (*rhi)->createDevice(headlessAdapter(**rhi), kHeadless);
        INFO(failureOf(device));
        REQUIRE(device.has_value());
        const auto swapchain = (*device)->createSwapchain(kHeadless, kSwapchainImages);
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
