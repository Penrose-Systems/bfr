// What every backend's Device does with a swapchain, through the interface
// alone. Each case runs on every backend the build has.

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/window.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/rhi.h"
#include "bfr/rhi/rhi_config.h"
#include "bfr/rhi/swapchain.h"
#include "rhi_test_support.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_container_properties.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
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
    explicit HeadlessDevice(const TestRhiDesc& desc, const RhiConfig& config = kValidation) {
        auto created = createRhi(desc, config);
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

TEST_CASE("Device builds a headless swapchain at the size asked for", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const HeadlessDevice headless{recordingInto(backend, messages)};
        const auto swapchain = headless.device->createSwapchain(kHeadless, swapchainRequest(*headless.rhi));
        INFO(failureOf(swapchain));
        REQUIRE(swapchain.has_value());

        const auto images = headless.device->resizeSwapchain(*swapchain, kWidth, kHeight);

        INFO(failureOf(images));
        REQUIRE(images.has_value());
        CHECK(images->width == kWidth);
        CHECK(images->height == kHeight);
        CHECK(images->count >= 2);
        CHECK(headless.device->destroySwapchain(*swapchain).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device rebuilds a swapchain at a new size", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const HeadlessDevice headless{recordingInto(backend, messages)};
        const auto swapchain = headless.device->createSwapchain(kHeadless, swapchainRequest(*headless.rhi));
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

TEST_CASE("Device destroys the swapchains left when it goes, and says so", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const HeadlessDevice headless{recordingInto(backend, messages)};
        const auto swapchain = headless.device->createSwapchain(kHeadless, swapchainRequest(*headless.rhi));
        INFO(failureOf(swapchain));
        REQUIRE(swapchain.has_value());
        const auto images = headless.device->resizeSwapchain(*swapchain, kWidth, kHeight);
        INFO(failureOf(images));
        REQUIRE(images.has_value());
    }

    // Validation reports nothing; the device warns once, when logging is
    // compiled in.
    const std::size_t warnings = bfr::core::kLoggingEnabled ? 1 : 0;
    REQUIRE(messages.size() == warnings);
    for (const std::string& message : messages) {
        CHECK_THAT(message, Catch::Matchers::ContainsSubstring("1 ") &&
                                Catch::Matchers::ContainsSubstring("swapchains were left when the device went"));
    }
}

TEST_CASE("Device empties a swapchain for a window with no area", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const HeadlessDevice headless{recordingInto(backend, messages)};
        const auto swapchain = headless.device->createSwapchain(kHeadless, swapchainRequest(*headless.rhi));
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
        CHECK(headless.device->destroySwapchain(*swapchain).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device fails with RhiInvalidHandle on a destroyed swapchain", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    const HeadlessDevice headless{recordingInto(backend, messages)};
    const auto swapchain = headless.device->createSwapchain(kHeadless, swapchainRequest(*headless.rhi));
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

TEST_CASE("Device destroying a null swapchain does nothing", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    const HeadlessDevice headless{recordingInto(backend, messages)};

    CHECK(headless.device->destroySwapchain(SwapchainHandle{}).has_value());
}

TEST_CASE("Device fails with RhiSurfaceFailed for a swapchain without a window", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    const HeadlessDevice headless{recordingInto(backend, messages)};

    const auto swapchain = headless.device->createSwapchain(kNoWindow, swapchainRequest(*headless.rhi));

    REQUIRE_FALSE(swapchain.has_value());
    CHECK(swapchain.error().code == bfr::core::ErrorCode::RhiSurfaceFailed);
    CHECK_THAT(swapchain.error().message, Catch::Matchers::ContainsSubstring("no window"));
}

TEST_CASE("Device fails with RhiInvalidHandle for a forged swapchain handle", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    // Past the pool's capacity; an unused slot with a live generation; the used
    // slot with a generation it never had; the used slot with a free one.
    const SwapchainHandle forged = GENERATE(SwapchainHandle{100000, 1}, SwapchainHandle{3, 1}, SwapchainHandle{0, 7},
                                            SwapchainHandle{0, 2});
    INFO("index " << forged.index() << ", generation " << forged.generation());
    std::vector<std::string> messages;
    const HeadlessDevice headless{recordingInto(backend, messages)};
    const auto swapchain = headless.device->createSwapchain(kHeadless, swapchainRequest(*headless.rhi));
    INFO(failureOf(swapchain));
    REQUIRE(swapchain.has_value());
    REQUIRE(swapchain->index() == 0);
    REQUIRE(swapchain->generation() == 1);

    const auto resized = headless.device->resizeSwapchain(forged, kWidth, kHeight);
    const auto destroyed = headless.device->destroySwapchain(forged);

    REQUIRE_FALSE(resized.has_value());
    CHECK(resized.error().code == bfr::core::ErrorCode::RhiInvalidHandle);
    REQUIRE_FALSE(destroyed.has_value());
    CHECK(destroyed.error().code == bfr::core::ErrorCode::RhiInvalidHandle);
}

TEST_CASE("Device builds the image count its swapchain was created with", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const HeadlessDevice headless{recordingInto(backend, messages), kValidation};
        const auto surface = headless.rhi->surfaceCaps(headlessAdapter(*headless.rhi), kHeadless);
        INFO(failureOf(surface));
        REQUIRE(surface.has_value());
        // One more than the window's least, more than the default, when the
        // window allows that many.
        const bool roomForMore = surface->maxImageCount == 0 || surface->maxImageCount > surface->minImageCount;
        const std::uint32_t wanted = surface->minImageCount + (roomForMore ? 1 : 0);
        const auto swapchain = headless.device->createSwapchain(kHeadless, {.imageCount = wanted});
        INFO(failureOf(swapchain));
        REQUIRE(swapchain.has_value());

        const auto images = headless.device->resizeSwapchain(*swapchain, kWidth, kHeight);

        INFO(failureOf(images));
        REQUIRE(images.has_value());
        CHECK(images->count >= wanted);
        CHECK(headless.device->destroySwapchain(*swapchain).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}
