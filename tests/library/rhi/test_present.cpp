// What every backend does from acquiring a swapchain image to presenting it,
// through the interface alone. Each case runs on every backend the build has.

#include "bfr/core/error.h"
#include "bfr/core/window.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/rhi.h"
#include "bfr/rhi/swapchain.h"
#include "rhi_test_support.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_container_properties.hpp>

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
constexpr int kFrames = 3;

const bfr::core::window::NativeWindow kHeadless = bfr::core::window::Headless{};

constexpr RenderingDesc kClear{.loadOp = LoadOp::Clear, .storeOp = StoreOp::Store, .clearColor = {0.2F, 0.4F, 0.6F, 1}};

// An Rhi made with desc, a device that presents to a headless window, and a
// swapchain on it built at kWidth by kHeight.
struct HeadlessSwapchain {
    explicit HeadlessSwapchain(const TestRhiDesc& desc) {
        auto created = createRhi(desc, kValidation);
        INFO(failureOf(created));
        REQUIRE(created.has_value());
        rhi = std::move(*created);
        device = presentingDevice(*rhi, kHeadless);
        auto made = device->createSwapchain(kHeadless, kSwapchainRequest);
        INFO(failureOf(made));
        REQUIRE(made.has_value());
        swapchain = *made;
        const auto images = device->resizeSwapchain(swapchain, kWidth, kHeight);
        INFO(failureOf(images));
        REQUIRE(images.has_value());
    }
    ~HeadlessSwapchain() { CHECK(device->destroySwapchain(swapchain).has_value()); }
    HeadlessSwapchain(const HeadlessSwapchain&) = delete;
    HeadlessSwapchain& operator=(const HeadlessSwapchain&) = delete;
    HeadlessSwapchain(HeadlessSwapchain&&) = delete;
    HeadlessSwapchain& operator=(HeadlessSwapchain&&) = delete;

    std::unique_ptr<Rhi> rhi;
    std::unique_ptr<Device> device;
    SwapchainHandle swapchain;
};

} // namespace

TEST_CASE("Device clears and presents a headless swapchain, frame after frame", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const HeadlessSwapchain headless{recordingInto(backend, messages)};
        Device& device = *headless.device;
        for (int frame = 0; frame < kFrames; ++frame) {
            INFO("frame " << frame);
            const auto acquired = device.acquire(headless.swapchain);
            INFO(failureOf(acquired));
            REQUIRE(acquired.has_value());
            const auto commands = device.beginCommands();
            INFO(failureOf(commands));
            REQUIRE(commands.has_value());
            CHECK(device.beginRendering(*commands, headless.swapchain, kClear).has_value());
            CHECK(device.endRendering(*commands).has_value());
            const auto submitted = device.submit(*commands);
            INFO(failureOf(submitted));
            REQUIRE(submitted.has_value());
            const auto presented = device.present(headless.swapchain);
            INFO(failureOf(presented));
            CHECK(presented.has_value());
        }
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device refuses to acquire from a swapchain with no images, and says to resize it", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const HeadlessSwapchain headless{recordingInto(backend, messages)};
        Device& device = *headless.device;
        REQUIRE(device.resizeSwapchain(headless.swapchain, 0, 0).has_value());

        const auto acquired = device.acquire(headless.swapchain);

        REQUIRE_FALSE(acquired.has_value());
        CHECK(acquired.error().code == bfr::core::ErrorCode::RhiOutOfDate);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device refuses calls out of order with RhiInvalidCall", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const HeadlessSwapchain headless{recordingInto(backend, messages)};
        Device& device = *headless.device;
        const auto commands = device.beginCommands();
        REQUIRE(commands.has_value());

        // Rendering before an image is acquired, ending rendering that never
        // began, presenting with no image.
        CHECK(device.beginRendering(*commands, headless.swapchain, kClear).error().code ==
              bfr::core::ErrorCode::RhiInvalidCall);
        CHECK(device.endRendering(*commands).error().code == bfr::core::ErrorCode::RhiInvalidCall);
        CHECK(device.present(headless.swapchain).error().code == bfr::core::ErrorCode::RhiInvalidCall);

        // Submitting while rendering is open.
        REQUIRE(device.acquire(headless.swapchain).has_value());
        REQUIRE(device.beginRendering(*commands, headless.swapchain, kClear).has_value());
        CHECK(device.submit(*commands).error().code == bfr::core::ErrorCode::RhiInvalidCall);
        REQUIRE(device.endRendering(*commands).has_value());
        REQUIRE(device.submit(*commands).has_value());
        CHECK(device.present(headless.swapchain).has_value());

        // A submitted list is gone.
        CHECK(device.submit(*commands).error().code == bfr::core::ErrorCode::RhiInvalidHandle);
        CHECK(device.destroyCommandList(*commands).error().code == bfr::core::ErrorCode::RhiInvalidHandle);
        CHECK(device.destroyCommandList({}).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}
