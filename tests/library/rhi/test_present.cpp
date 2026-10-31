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

#include <array>
#include <cstdint>
#include <memory>
#include <span>
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

bfr::core::Result<Token> submitOne(Device& device, CommandListHandle commandList, std::uint32_t queue = 0,
                                   std::span<const Token> waits = {}) {
    return device.submit({.queue = queue, .commandLists = std::span{&commandList, 1}, .waits = waits});
}

struct HeadlessSwapchain {
    explicit HeadlessSwapchain(const TestRhiDesc& desc) {
        auto created = createRhi(desc, kValidation);
        INFO(failureOf(created));
        REQUIRE(created.has_value());
        rhi = std::move(*created);
        device = presentingDevice(*rhi, kHeadless);
        auto made = device->createSwapchain(kHeadless, swapchainRequest(*rhi));
        INFO(failureOf(made));
        REQUIRE(made.has_value());
        swapchain = *made;
        const auto images = device->resizeSwapchain(swapchain, kWidth, kHeight);
        INFO(failureOf(images));
        REQUIRE(images.has_value());
        const auto madePool = device->createCommandPool(0);
        INFO(failureOf(madePool));
        REQUIRE(madePool.has_value());
        pool = *madePool;
    }
    ~HeadlessSwapchain() {
        CHECK(device->destroyCommandPool(pool).has_value());
        CHECK(device->destroySwapchain(swapchain).has_value());
    }
    HeadlessSwapchain(const HeadlessSwapchain&) = delete;
    HeadlessSwapchain& operator=(const HeadlessSwapchain&) = delete;
    HeadlessSwapchain(HeadlessSwapchain&&) = delete;
    HeadlessSwapchain& operator=(HeadlessSwapchain&&) = delete;

    std::unique_ptr<Rhi> rhi;
    std::unique_ptr<Device> device;
    SwapchainHandle swapchain;
    CommandPoolHandle pool;
};

} // namespace

TEST_CASE("Device clears and presents a headless swapchain, frame after frame", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const HeadlessSwapchain headless{recordingInto(backend, messages)};
        Device& device = *headless.device;
        Token last;
        for (int frame = 0; frame < kFrames; ++frame) {
            INFO("frame " << frame);
            REQUIRE(device.wait(last).has_value());
            const auto reset = device.resetCommandPool(headless.pool);
            INFO(failureOf(reset));
            REQUIRE(reset.has_value());
            const auto acquired = device.acquire(headless.swapchain);
            INFO(failureOf(acquired));
            REQUIRE(acquired.has_value());
            const auto commands = device.beginCommands(headless.pool);
            INFO(failureOf(commands));
            REQUIRE(commands.has_value());
            CHECK(device.beginRendering(*commands, headless.swapchain, kClear).has_value());
            CHECK(device.endRendering(*commands).has_value());
            const auto submitted = submitOne(device, *commands);
            INFO(failureOf(submitted));
            REQUIRE(submitted.has_value());
            CHECK(submitted->value > last.value);
            last = *submitted;
            const auto presented = device.present(headless.swapchain);
            INFO(failureOf(presented));
            CHECK(presented.has_value());
        }
        CHECK(device.wait(last).has_value());
        CHECK(device.poll(last).value_or(false));
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
        const auto commands = device.beginCommands(headless.pool);
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
        CHECK(submitOne(device, *commands).error().code == bfr::core::ErrorCode::RhiInvalidCall);
        REQUIRE(device.endRendering(*commands).has_value());
        REQUIRE(submitOne(device, *commands).has_value());
        CHECK(device.present(headless.swapchain).has_value());

        // A submitted list is gone.
        CHECK(submitOne(device, *commands).error().code == bfr::core::ErrorCode::RhiInvalidHandle);
        CHECK(device.destroyCommandList(*commands).error().code == bfr::core::ErrorCode::RhiInvalidHandle);
        CHECK(device.destroyCommandList({}).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device completes a default token at once, and refuses a token no submit returned", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const HeadlessSwapchain headless{recordingInto(backend, messages)};
        Device& device = *headless.device;

        CHECK(device.poll(Token{}).value_or(false));
        CHECK(device.wait(Token{}).has_value());
        CHECK(device.wait(Token{.queue = 1, .value = 0}).error().code == bfr::core::ErrorCode::RhiInvalidCall);
        CHECK(device.poll(Token{.queue = 0, .value = 1}).error().code == bfr::core::ErrorCode::RhiInvalidCall);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device resets a command pool only once its lists are submitted and run", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const HeadlessSwapchain headless{recordingInto(backend, messages)};
        Device& device = *headless.device;

        const auto commands = device.beginCommands(headless.pool);
        REQUIRE(commands.has_value());
        CHECK(device.resetCommandPool(headless.pool).error().code == bfr::core::ErrorCode::RhiInvalidCall);
        REQUIRE(device.destroyCommandList(*commands).has_value());
        CHECK(device.resetCommandPool(headless.pool).has_value());

        REQUIRE(device.acquire(headless.swapchain).has_value());
        const auto frame = device.beginCommands(headless.pool);
        REQUIRE(frame.has_value());
        REQUIRE(device.beginRendering(*frame, headless.swapchain, kClear).has_value());
        REQUIRE(device.endRendering(*frame).has_value());
        const auto token = submitOne(device, *frame);
        REQUIRE(token.has_value());
        REQUIRE(device.present(headless.swapchain).has_value());
        REQUIRE(device.wait(*token).has_value());
        CHECK(device.poll(*token).value_or(false));
        CHECK(device.resetCommandPool(headless.pool).has_value());

        const auto other = device.createCommandPool(0);
        REQUIRE(other.has_value());
        const auto orphan = device.beginCommands(*other);
        REQUIRE(orphan.has_value());
        REQUIRE(device.destroyCommandPool(*other).has_value());
        CHECK(device.beginCommands(*other).error().code == bfr::core::ErrorCode::RhiInvalidHandle);
        CHECK(device.resetCommandPool(*other).error().code == bfr::core::ErrorCode::RhiInvalidHandle);
        CHECK(device.destroyCommandPool(*other).error().code == bfr::core::ErrorCode::RhiInvalidHandle);
        CHECK(device.destroyCommandPool({}).has_value());
        CHECK(device.destroyCommandList(*orphan).error().code == bfr::core::ErrorCode::RhiInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device refuses a pool, a submit and a token on a queue it does not have", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const HeadlessSwapchain headless{recordingInto(backend, messages)};
        Device& device = *headless.device;
        constexpr std::uint32_t kMissing = 7;

        CHECK(device.createCommandPool(kMissing).error().code == bfr::core::ErrorCode::RhiInvalidCall);
        CHECK(device.submit({.queue = kMissing, .commandLists = {}, .waits = {}}).error().code ==
              bfr::core::ErrorCode::RhiInvalidCall);
        CHECK(device.wait(Token{.queue = kMissing, .value = 0}).error().code == bfr::core::ErrorCode::RhiInvalidCall);
        CHECK(device.poll(Token{.queue = kMissing, .value = 0}).error().code == bfr::core::ErrorCode::RhiInvalidCall);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device submits several lists at once and each waits only on what it is told to", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const HeadlessSwapchain headless{recordingInto(backend, messages)};
        Device& device = *headless.device;

        REQUIRE(device.acquire(headless.swapchain).has_value());
        const auto first = device.beginCommands(headless.pool);
        const auto second = device.beginCommands(headless.pool);
        REQUIRE(first.has_value());
        REQUIRE(second.has_value());
        REQUIRE(device.beginRendering(*second, headless.swapchain, kClear).has_value());
        REQUIRE(device.endRendering(*second).has_value());
        const std::array lists{*first, *second};
        const auto token = device.submit({.queue = 0, .commandLists = lists, .waits = {}});
        INFO(failureOf(token));
        REQUIRE(token.has_value());
        REQUIRE(device.present(headless.swapchain).has_value());

        // A later submit may wait on an earlier token, never on one not yet returned.
        const auto third = device.beginCommands(headless.pool);
        REQUIRE(third.has_value());
        const Token future{.queue = 0, .value = token->value + 1};
        CHECK(submitOne(device, *third, 0, std::span{&future, 1}).error().code == bfr::core::ErrorCode::RhiInvalidCall);
        const auto later = submitOne(device, *third, 0, std::span{&*token, 1});
        INFO(failureOf(later));
        REQUIRE(later.has_value());
        CHECK(later->value > token->value);
        REQUIRE(device.wait(*later).has_value());
        CHECK(device.poll(*token).value_or(false));
        CHECK(device.destroyCommandList(*first).error().code == bfr::core::ErrorCode::RhiInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device runs a compute queue beside graphics when the adapter has one", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        auto created = createRhi(recordingInto(backend, messages), kValidation);
        REQUIRE(created.has_value());
        Rhi& rhi = **created;
        const std::size_t adapter = headlessAdapter(rhi);
        const std::uint32_t computeQueues = std::min<std::uint32_t>(rhi.adapters()[adapter].caps.queues.compute, 1);
        auto opened = rhi.createDevice(adapter, kHeadless, {.compute = computeQueues});
        INFO(failureOf(opened));
        REQUIRE(opened.has_value());
        Device& device = **opened;
        if (computeQueues == 0) {
            CHECK(device.createCommandPool(1).error().code == bfr::core::ErrorCode::RhiInvalidCall);
            return;
        }

        const auto graphicsPool = device.createCommandPool(0);
        const auto computePool = device.createCommandPool(1);
        REQUIRE(graphicsPool.has_value());
        REQUIRE(computePool.has_value());
        const auto graphics = device.beginCommands(*graphicsPool);
        const auto compute = device.beginCommands(*computePool);
        REQUIRE(graphics.has_value());
        REQUIRE(compute.has_value());

        // A compute list neither renders nor goes to the graphics queue.
        auto swapchain = device.createSwapchain(kHeadless, swapchainRequest(rhi));
        REQUIRE(swapchain.has_value());
        REQUIRE(device.resizeSwapchain(*swapchain, kWidth, kHeight).has_value());
        REQUIRE(device.acquire(*swapchain).has_value());
        CHECK(device.beginRendering(*compute, *swapchain, kClear).error().code == bfr::core::ErrorCode::RhiInvalidCall);
        CHECK(submitOne(device, *compute, 0).error().code == bfr::core::ErrorCode::RhiInvalidCall);

        // Compute waits on graphics across queues, and both tokens complete.
        REQUIRE(device.beginRendering(*graphics, *swapchain, kClear).has_value());
        REQUIRE(device.endRendering(*graphics).has_value());
        const auto rendered = submitOne(device, *graphics, 0);
        INFO(failureOf(rendered));
        REQUIRE(rendered.has_value());
        REQUIRE(device.present(*swapchain).has_value());
        const auto computed = submitOne(device, *compute, 1, std::span{&*rendered, 1});
        INFO(failureOf(computed));
        REQUIRE(computed.has_value());
        CHECK(computed->queue == 1);
        REQUIRE(device.wait(*computed).has_value());
        CHECK(device.poll(*rendered).value_or(false));
        CHECK(device.resetCommandPool(*computePool).has_value());
        CHECK(device.resetCommandPool(*graphicsPool).has_value());
        CHECK(device.destroyCommandPool(*computePool).has_value());
        CHECK(device.destroyCommandPool(*graphicsPool).has_value());
        CHECK(device.destroySwapchain(*swapchain).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}
