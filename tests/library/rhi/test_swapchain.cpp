// What every backend's Device does with a swapchain, through the public API
// alone. Each case runs on every backend the build has.

#include "bfr.hpp"
#include "bfr/bfr.h"
#include "bfr_test_support.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_container_properties.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

using namespace bfr;
using namespace bfr::test;

namespace {

constexpr std::uint32_t kWidth = 64;
constexpr std::uint32_t kHeight = 48;

// The bits of a handle: its slot's index in the low kHandleIndexBits, the
// slot's generation above them. bfr.h does not state them; a forged handle
// needs them.
constexpr unsigned kHandleIndexBits = 24;
constexpr std::uint64_t kHandleIndexMask = (std::uint64_t{1} << kHandleIndexBits) - 1;

// A swapchain handle the device never gave out, by its slot and generation.
struct ForgedSwapchain {
    std::uint32_t index{};
    std::uint64_t generation{};
};

bfr_swapchain bitsOf(const ForgedSwapchain& forged) {
    return {.bits = (forged.generation << kHandleIndexBits) | (forged.index & kHandleIndexMask)};
}

} // namespace

TEST_CASE("Device builds a headless swapchain at the size asked for", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const HeadlessDevice headless{recordingInto(backend, messages)};
        auto swapchain = headless.device.createSwapchain(swapchainDesc(headless.instance));
        INFO(failureOf(swapchain));
        REQUIRE(swapchain.has_value());

        const auto images = swapchain->resize(kWidth, kHeight);

        INFO(failureOf(images));
        REQUIRE(images.has_value());
        CHECK(images->width == kWidth);
        CHECK(images->height == kHeight);
        CHECK(images->count >= 2);
        const SwapchainRef destroyedSwapchain = *swapchain;
        *swapchain = {};
        CHECK(resizeResult(headless.device, destroyedSwapchain.get(), kWidth, kHeight) == Result::ErrorInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device refuses a swapchain of a format BFR does not have", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const HeadlessDevice headless{recordingInto(backend, messages)};
        SwapchainDesc desc = swapchainDesc(headless.instance);
        desc.format = Format::Undefined;

        CHECK(headless.device.createSwapchain(desc).error().code == Result::ErrorInvalidArgument);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device rebuilds a swapchain at a new size", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const HeadlessDevice headless{recordingInto(backend, messages)};
        auto swapchain = headless.device.createSwapchain(swapchainDesc(headless.instance));
        INFO(failureOf(swapchain));
        REQUIRE(swapchain.has_value());
        const auto first = swapchain->resize(kWidth, kHeight);
        INFO(failureOf(first));
        REQUIRE(first.has_value());

        const auto images = swapchain->resize(kWidth / 2, kHeight / 2);

        INFO(failureOf(images));
        REQUIRE(images.has_value());
        CHECK(images->width == kWidth / 2);
        CHECK(images->height == kHeight / 2);
        const SwapchainRef destroyedSwapchain = *swapchain;
        *swapchain = {};
        CHECK(resizeResult(headless.device, destroyedSwapchain.get(), kWidth, kHeight) == Result::ErrorInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device destroys the swapchains left when it goes, and says so", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        // Made through bfr.h: a Swapchain keeps its Device alive, so only a
        // swapchain no Swapchain owns can be left when the device goes.
        const HeadlessDevice headless{recordingInto(backend, messages)};
        const bfr_swapchain_desc desc = detail::toC(swapchainDesc(headless.instance), nullptr);
        bfr_swapchain swapchain{};
        const Result created = resultOf(bfr_create_swapchain(headless.device.get(), &desc, &swapchain));
        INFO(failureOf(created));
        REQUIRE(created == Result::Success);
        const Result resized = resizeResult(headless.device, swapchain, kWidth, kHeight);
        INFO(failureOf(resized));
        REQUIRE(resized == Result::Success);
    }

    // Validation reports nothing; the device warns once, when logging is
    // compiled in.
    const std::size_t warnings = BFR_LOGGING_ENABLED ? 1 : 0;
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
        auto swapchain = headless.device.createSwapchain(swapchainDesc(headless.instance));
        INFO(failureOf(swapchain));
        REQUIRE(swapchain.has_value());
        const auto built = swapchain->resize(kWidth, kHeight);
        INFO(failureOf(built));
        REQUIRE(built.has_value());

        const auto images = swapchain->resize(0, kHeight);

        INFO(failureOf(images));
        REQUIRE(images.has_value());
        CHECK(images->width == 0);
        CHECK(images->height == 0);
        CHECK(images->count == 0);
        const SwapchainRef destroyedSwapchain = *swapchain;
        *swapchain = {};
        CHECK(resizeResult(headless.device, destroyedSwapchain.get(), kWidth, kHeight) == Result::ErrorInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device fails with RhiInvalidHandle on a destroyed swapchain", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    const HeadlessDevice headless{recordingInto(backend, messages)};
    auto swapchain = headless.device.createSwapchain(swapchainDesc(headless.instance));
    INFO(failureOf(swapchain));
    REQUIRE(swapchain.has_value());
    const SwapchainRef destroyedSwapchain = *swapchain;
    *swapchain = {};

    const Result resized = resizeResult(headless.device, destroyedSwapchain.get(), kWidth, kHeight);
    // d1: stale destroy reported as an Error message (track B)
    bfr_destroy_swapchain(headless.device.get(), destroyedSwapchain.get());

    REQUIRE_FALSE(resized == Result::Success);
    CHECK(resized == Result::ErrorInvalidHandle);
}

TEST_CASE("Device destroying a null swapchain does nothing", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    const HeadlessDevice headless{recordingInto(backend, messages)};

    // d1: null destroy: no message (track B)
    bfr_destroy_swapchain(headless.device.get(), SwapchainRef{}.get());
}

TEST_CASE("Device fails with ErrorInvalidArgument for a swapchain without a window", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    const HeadlessDevice headless{recordingInto(backend, messages)};
    SwapchainDesc desc = swapchainDesc(headless.instance);
    desc.window = kNoWindow;

    const auto swapchain = headless.device.createSwapchain(desc);

    REQUIRE_FALSE(swapchain.has_value());
    CHECK(swapchain.error().code == Result::ErrorInvalidArgument);
    CHECK_THAT(swapchain.error().message.text(), Catch::Matchers::ContainsSubstring("needs a window"));
}

TEST_CASE("Device fails with RhiInvalidHandle for a forged swapchain handle", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    // Past the pool's capacity; an unused slot with a live generation; the used
    // slot with a generation it never had; the used slot with a free one.
    const ForgedSwapchain forged = GENERATE(ForgedSwapchain{100000, 1}, ForgedSwapchain{3, 1}, ForgedSwapchain{0, 7},
                                            ForgedSwapchain{0, 2});
    INFO("index " << forged.index << ", generation " << forged.generation);
    std::vector<std::string> messages;
    const HeadlessDevice headless{recordingInto(backend, messages)};
    const auto swapchain = headless.device.createSwapchain(swapchainDesc(headless.instance));
    INFO(failureOf(swapchain));
    REQUIRE(swapchain.has_value());
    REQUIRE((swapchain->get().bits & kHandleIndexMask) == 0);
    REQUIRE((swapchain->get().bits >> kHandleIndexBits) == 1);

    const Result resized = resizeResult(headless.device, bitsOf(forged), kWidth, kHeight);
    // d1: stale destroy reported as an Error message (track B)
    bfr_destroy_swapchain(headless.device.get(), bitsOf(forged));

    REQUIRE_FALSE(resized == Result::Success);
    CHECK(resized == Result::ErrorInvalidHandle);
    // The forged destroy left the live swapchain alone.
    CHECK(resizeResult(headless.device, swapchain->get(), kWidth, kHeight) == Result::Success);
}

TEST_CASE("Device builds the image count its swapchain was created with", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const HeadlessDevice headless{recordingInto(backend, messages), kValidation};
        const auto surface = headless.instance.swapchainCaps(headlessAdapter(headless.instance), {.window = kHeadless});
        INFO(failureOf(surface));
        REQUIRE(surface.has_value());
        // One more than the window's least, more than the default, when the
        // window allows that many.
        const bool roomForMore = surface->maxImageCount == 0 || surface->maxImageCount > surface->minImageCount;
        const std::uint32_t wanted = surface->minImageCount + (roomForMore ? 1 : 0);
        SwapchainDesc desc = kTestSwapchain;
        desc.imageCount = wanted;
        auto swapchain = headless.device.createSwapchain(desc);
        INFO(failureOf(swapchain));
        REQUIRE(swapchain.has_value());

        const auto images = swapchain->resize(kWidth, kHeight);

        INFO(failureOf(images));
        REQUIRE(images.has_value());
        CHECK(images->count >= wanted);
        const SwapchainRef destroyedSwapchain = *swapchain;
        *swapchain = {};
        CHECK(resizeResult(headless.device, destroyedSwapchain.get(), kWidth, kHeight) == Result::ErrorInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}
