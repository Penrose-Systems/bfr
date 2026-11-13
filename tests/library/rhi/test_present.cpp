// What every backend does from acquiring a swapchain image to presenting it,
// through the interface alone. Each case runs on every backend the build has.

#include "bfr/core/error.h"
#include "bfr/core/handle.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/instance.h"
#include "bfr/rhi/swapchain.h"
#include "bfr/rhi/texture.h"
#include "rhi_test_support.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_container_properties.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace bfr::rhi;
using namespace bfr::rhi::test;

namespace {

constexpr std::uint32_t kWidth = 64;
constexpr std::uint32_t kHeight = 48;
constexpr int kFrames = 3;

const bfr::core::window::NativeWindow kHeadless = bfr::core::window::Headless{};

// The swapchain's image, cleared.
constexpr RenderingDesc kClear{.colorAttachmentCount = 1,
                               .colorAttachments = {{{.texture = {},
                                                      .loadOp = LoadOp::Clear,
                                                      .storeOp = StoreOp::Store,
                                                      .clearColor = {0.2F, 0.4F, 0.6F, 1}}}},
                               .depthStencil = {},
                               .occlusionQueries = {}};

bfr::core::Result<Token> submitOne(Device& device, CommandListHandle commandList, std::uint32_t queue = 0,
                                   std::span<const Token> waits = {}) {
    return device.submit({.queue = queue, .commandLists = std::span{&commandList, 1}, .waits = waits});
}

struct HeadlessSwapchain {
    explicit HeadlessSwapchain(const InstanceDesc& desc) {
        auto created = createInstance(desc, kValidation);
        INFO(failureOf(created));
        REQUIRE(created.has_value());
        instance = std::move(*created);
        device = presentingDevice(*instance, kHeadless);
        auto made = device->createSwapchain(kHeadless, swapchainDesc(*instance));
        INFO(failureOf(made));
        REQUIRE(made.has_value());
        swapchain = *made;
        const auto images = device->resizeSwapchain(swapchain, kWidth, kHeight);
        INFO(failureOf(images));
        REQUIRE(images.has_value());
        const auto madePool = device->createCommandPool({.queue = 0, .listCount = kPoolLists});
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

    std::unique_ptr<Instance> instance;
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
            REQUIRE(device.wait(last, kTestWait).has_value());
            const auto reset = device.resetCommandPool(headless.pool);
            INFO(failureOf(reset));
            REQUIRE(reset.has_value());
            const auto acquired = device.acquire(headless.swapchain, kTestWait);
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
        CHECK(device.wait(last, kTestWait).has_value());
        CHECK(device.poll(last).value_or(false));
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device resolves a multisampled texture into a swapchain's image and presents it", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const HeadlessSwapchain headless{recordingInto(backend, messages)};
        Device& device = *headless.device;
        const Format format = swapchainDesc(*headless.instance).format;
        constexpr std::uint32_t kSamples = 4;
        const auto caps = headless.instance->formatCaps(headlessAdapter(*headless.instance), format);
        REQUIRE(caps.has_value());
        if ((caps->sampleCounts & kSamples) == 0 || !caps->resolve) {
            SKIP("the adapter does not resolve the swapchain's format from 4 samples");
        }
        const auto texture = device.createTexture({.dimension = TextureDimension::D2,
                                                   .format = format,
                                                   .width = kWidth,
                                                   .height = kHeight,
                                                   .depth = 1,
                                                   .layers = 1,
                                                   .mipLevels = 1,
                                                   .samples = kSamples,
                                                   .usage = {.colorAttachment = true}});
        INFO(failureOf(texture));
        REQUIRE(texture.has_value());
        const auto acquired = device.acquire(headless.swapchain, kTestWait);
        INFO(failureOf(acquired));
        REQUIRE(acquired.has_value());
        const auto commands = device.beginCommands(headless.pool);
        REQUIRE(commands.has_value());
        CHECK(device.textureBarrier(*commands, *texture, TextureLayout::Undefined, TextureLayout::ColorTarget, {})
                  .has_value());
        RenderingDesc resolving = kClear;
        resolving.colorAttachments[0].texture = *texture;
        resolving.colorAttachments[0].storeOp = StoreOp::DontCare;
        const auto begun = device.beginRendering(*commands, headless.swapchain, resolving);
        INFO(failureOf(begun));
        CHECK(begun.has_value());
        CHECK(device.endRendering(*commands).has_value());
        const auto submitted = submitOne(device, *commands);
        INFO(failureOf(submitted));
        REQUIRE(submitted.has_value());
        const auto presented = device.present(headless.swapchain);
        INFO(failureOf(presented));
        CHECK(presented.has_value());
        CHECK(device.wait(*submitted, kTestWait).has_value());
        CHECK(device.destroyTexture(*texture).has_value());
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

        const auto acquired = device.acquire(headless.swapchain, kTestWait);

        REQUIRE_FALSE(acquired.has_value());
        CHECK(acquired.error().code == bfr::core::ErrorCode::RhiOutOfDate);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device refuses a swapchain call with no image acquired with RhiInvalidCall", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const HeadlessSwapchain headless{recordingInto(backend, messages)};
        Device& device = *headless.device;
        const auto commands = device.beginCommands(headless.pool);
        REQUIRE(commands.has_value());

        // Rendering to a swapchain, or presenting it, needs an acquired image.
        CHECK(device.beginRendering(*commands, headless.swapchain, kClear).error().code ==
              bfr::core::ErrorCode::RhiInvalidCall);
        CHECK(device.present(headless.swapchain).error().code == bfr::core::ErrorCode::RhiInvalidCall);

        REQUIRE(device.acquire(headless.swapchain, kTestWait).has_value());
        REQUIRE(device.beginRendering(*commands, headless.swapchain, kClear).has_value());
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

TEST_CASE("Device completes a default token at once, and has not passed one no submit returned", "[rhi][agnostic]") {
    constexpr std::uint64_t kNoWait = 0;
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const HeadlessSwapchain headless{recordingInto(backend, messages)};
        Device& device = *headless.device;

        CHECK(device.poll(Token{}).value_or(false));
        CHECK(device.wait(Token{}, kTestWait).has_value());
        CHECK(device.wait(Token{.queue = 1, .value = 0}, kTestWait).error().code ==
              bfr::core::ErrorCode::RhiInvalidCall);
        const auto ahead = device.poll(Token{.queue = 0, .value = 1});
        INFO(failureOf(ahead));
        CHECK_FALSE(ahead.value_or(true));
        const auto waited = device.wait(Token{.queue = 0, .value = 1}, kNoWait);
        REQUIRE_FALSE(waited.has_value());
        CHECK(waited.error().code == bfr::core::ErrorCode::RhiTimeout);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device resets a command pool once the GPU has run its lists", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const HeadlessSwapchain headless{recordingInto(backend, messages)};
        Device& device = *headless.device;

        const auto commands = device.beginCommands(headless.pool);
        REQUIRE(commands.has_value());
        // Vulkan's reset ends a list still recording; D3D12 and Metal refuse it.
        if (backend.backend == Backend::Vulkan) {
            CHECK(device.resetCommandPool(headless.pool).has_value());
            CHECK(device.destroyCommandList(*commands).error().code == bfr::core::ErrorCode::RhiInvalidHandle);
        }
        else {
            CHECK(device.resetCommandPool(headless.pool).error().code == bfr::core::ErrorCode::RhiInvalidCall);
            REQUIRE(device.destroyCommandList(*commands).has_value());
            CHECK(device.resetCommandPool(headless.pool).has_value());
        }

        REQUIRE(device.acquire(headless.swapchain, kTestWait).has_value());
        const auto frame = device.beginCommands(headless.pool);
        REQUIRE(frame.has_value());
        REQUIRE(device.beginRendering(*frame, headless.swapchain, kClear).has_value());
        REQUIRE(device.endRendering(*frame).has_value());
        const auto token = submitOne(device, *frame);
        REQUIRE(token.has_value());
        REQUIRE(device.present(headless.swapchain).has_value());
        REQUIRE(device.wait(*token, kTestWait).has_value());
        CHECK(device.poll(*token).value_or(false));
        CHECK(device.resetCommandPool(headless.pool).has_value());

        const auto other = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
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

        CHECK(device.createCommandPool({.queue = kMissing, .listCount = kPoolLists}).error().code ==
              bfr::core::ErrorCode::RhiInvalidCall);
        CHECK(device.submit({.queue = kMissing, .commandLists = {}, .waits = {}}).error().code ==
              bfr::core::ErrorCode::RhiInvalidCall);
        CHECK(device.wait(Token{.queue = kMissing, .value = 0}, kTestWait).error().code ==
              bfr::core::ErrorCode::RhiInvalidCall);
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

        REQUIRE(device.acquire(headless.swapchain, kTestWait).has_value());
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

        // A later submit waits on an earlier token.
        const auto third = device.beginCommands(headless.pool);
        REQUIRE(third.has_value());
        const auto later = submitOne(device, *third, 0, std::span{&*token, 1});
        INFO(failureOf(later));
        REQUIRE(later.has_value());
        CHECK(later->value > token->value);
        REQUIRE(device.wait(*later, kTestWait).has_value());
        CHECK(device.poll(*token).value_or(false));
        CHECK(device.destroyCommandList(*first).error().code == bfr::core::ErrorCode::RhiInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device runs a compute queue beside graphics when the adapter has one", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        auto created = createInstance(recordingInto(backend, messages), kValidation);
        REQUIRE(created.has_value());
        Instance& rhi = **created;
        const std::size_t adapter = headlessAdapter(rhi);
        const std::uint32_t computeQueues = std::min<std::uint32_t>(rhi.adapters()[adapter].caps.queues.compute, 1);
        auto opened = rhi.createDevice(adapter, kHeadless, testDeviceWith([&](DeviceDesc& desc) {
                                           desc.compute = std::span{kTestQueues}.first(computeQueues);
                                       }));
        INFO(failureOf(opened));
        REQUIRE(opened.has_value());
        Device& device = **opened;
        if (computeQueues == 0) {
            CHECK(device.createCommandPool({.queue = 1, .listCount = kPoolLists}).error().code ==
                  bfr::core::ErrorCode::RhiInvalidCall);
            return;
        }

        const auto graphicsPool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        const auto computePool = device.createCommandPool({.queue = 1, .listCount = kPoolLists});
        REQUIRE(graphicsPool.has_value());
        REQUIRE(computePool.has_value());
        const auto graphics = device.beginCommands(*graphicsPool);
        const auto compute = device.beginCommands(*computePool);
        REQUIRE(graphics.has_value());
        REQUIRE(compute.has_value());

        // A compute list neither renders nor goes to the graphics queue.
        auto swapchain = device.createSwapchain(kHeadless, swapchainDesc(rhi));
        REQUIRE(swapchain.has_value());
        REQUIRE(device.resizeSwapchain(*swapchain, kWidth, kHeight).has_value());
        REQUIRE(device.acquire(*swapchain, kTestWait).has_value());
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
        REQUIRE(device.wait(*computed, kTestWait).has_value());
        CHECK(device.poll(*rendered).value_or(false));
        CHECK(device.resetCommandPool(*computePool).has_value());
        CHECK(device.resetCommandPool(*graphicsPool).has_value());
        CHECK(device.destroyCommandPool(*computePool).has_value());
        CHECK(device.destroyCommandPool(*graphicsPool).has_value());
        CHECK(device.destroySwapchain(*swapchain).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device frees a pool destroyed while the GPU runs its lists once the GPU is done", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        auto created = createInstance(recordingInto(backend, messages), kValidation);
        REQUIRE(created.has_value());
        Instance& rhi = **created;
        auto opened = rhi.createDevice(headlessAdapter(rhi), kHeadless, testDeviceWith([](DeviceDesc& desc) {
                                           desc.maxCommandLists = kPoolLists;
                                           desc.maxCommandPools = 1;
                                       }));
        INFO(failureOf(opened));
        REQUIRE(opened.has_value());
        Device& device = **opened;
        const auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        const auto list = device.beginCommands(*pool);
        REQUIRE(list.has_value());
        const auto token = device.submit({.queue = 0, .commandLists = std::span{&*list, 1}, .waits = {}});
        INFO(failureOf(token));
        REQUIRE(token.has_value());

        // Destroyed before the GPU is known to be done, the pool is gone from
        // the caller at once, and its room is back once the GPU has passed it.
        REQUIRE(device.destroyCommandPool(*pool).has_value());
        CHECK(device.resetCommandPool(*pool).error().code == bfr::core::ErrorCode::RhiInvalidHandle);
        REQUIRE(device.wait(*token, kTestWait).has_value());
        const auto next = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        INFO(failureOf(next));
        REQUIRE(next.has_value());
        CHECK(device.destroyCommandPool(*next).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device serves pools made, submitted, waited on and destroyed from several threads at once",
          "[rhi][agnostic]") {
    // Each thread holds one pool at a time, live or awaiting the GPU, so a
    // device of one pool per thread never runs out. Catch2 is not called off
    // the test thread: each thread records what failed, checked after joining.
    constexpr std::uint32_t kThreads = 4;
    constexpr int kRounds = 8;
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        auto created = createInstance(recordingInto(backend, messages), kValidation);
        REQUIRE(created.has_value());
        Instance& rhi = **created;
        auto opened = rhi.createDevice(headlessAdapter(rhi), kHeadless, testDeviceWith([](DeviceDesc& desc) {
                                           desc.maxCommandLists = kThreads * kPoolLists;
                                           desc.maxCommandPools = kThreads;
                                       }));
        INFO(failureOf(opened));
        REQUIRE(opened.has_value());
        Device& device = **opened;

        std::array<std::string, kThreads> failures;
        {
            std::vector<std::jthread> threads;
            threads.reserve(kThreads);
            for (std::string& failure : failures) {
                threads.emplace_back([&device, &failure] {
                    for (int round = 0; round < kRounds && failure.empty(); ++round) {
                        const auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
                        if (!pool) {
                            failure = "createCommandPool: " + pool.error().message.text();
                            return;
                        }
                        const auto list = device.beginCommands(*pool);
                        if (!list) {
                            failure = "beginCommands: " + list.error().message.text();
                            return;
                        }
                        const auto token = device.submit(
                            {.queue = 0, .commandLists = std::span{&*list, 1}, .waits = {}});
                        if (!token) {
                            failure = "submit: " + token.error().message.text();
                            return;
                        }
                        if (const auto waited = device.wait(*token, kTestWait); !waited) {
                            failure = "wait: " + waited.error().message.text();
                            return;
                        }
                        if (const auto destroyed = device.destroyCommandPool(*pool); !destroyed) {
                            failure = "destroyCommandPool: " + destroyed.error().message.text();
                            return;
                        }
                    }
                });
            }
        }
        for (const std::string& failure : failures) {
            CHECK(failure.empty());
        }

        // Every pool's room is back once the threads are done.
        std::vector<CommandPoolHandle> pools;
        for (std::uint32_t index = 0; index < kThreads; ++index) {
            const auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
            INFO(failureOf(pool));
            REQUIRE(pool.has_value());
            pools.push_back(*pool);
        }
        for (const CommandPoolHandle pool : pools) {
            CHECK(device.destroyCommandPool(pool).has_value());
        }
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device opens the graphics queues it is asked for, numbered from 0", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        auto created = createInstance(recordingInto(backend, messages), kValidation);
        REQUIRE(created.has_value());
        Instance& rhi = **created;
        constexpr std::uint32_t kGraphics = 2;
        const auto adapters = rhi.adapters();
        const auto found = std::ranges::find_if(adapters, [](const Adapter& adapter) {
            return adapter.caps.queues.graphics >= kGraphics;
        });
        if (found == adapters.end()) {
            SKIP("no adapter has " << kGraphics << " graphics queues");
        }
        const auto adapter = static_cast<std::size_t>(found - adapters.begin());
        auto opened = rhi.createDevice(adapter, kNoWindow, testDeviceWith([](DeviceDesc& desc) {
                                           desc.graphics = std::span{kTestQueues}.first(kGraphics);
                                       }));
        INFO(failureOf(opened));
        REQUIRE(opened.has_value());
        Device& device = **opened;
        // Queue 1 is the second graphics queue: its lists submit to it.
        const auto pool = device.createCommandPool({.queue = 1, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        const auto list = device.beginCommands(*pool);
        REQUIRE(list.has_value());
        const auto token = device.submit({.queue = 1, .commandLists = std::span{&*list, 1}, .waits = {}});
        INFO(failureOf(token));
        REQUIRE(token.has_value());
        CHECK(device.wait(*token, kTestWait).has_value());
        CHECK(device.destroyCommandPool(*pool).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device ranks its queues where the backend does, and fails with RhiUnsupported where it doesn't",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        auto created = createInstance(recordingInto(backend, messages), kValidation);
        REQUIRE(created.has_value());
        Instance& rhi = **created;
        const auto adapters = rhi.adapters();
        const auto found = std::ranges::find_if(adapters, [](const Adapter& adapter) {
            return adapter.caps.queues.graphics != 0 && adapter.caps.queues.compute != 0;
        });
        if (found == adapters.end()) {
            SKIP("no adapter has a graphics and a compute queue to rank");
        }
        const auto adapter = static_cast<std::size_t>(found - adapters.begin());
        const Queues& queues = found->caps.queues;
        constexpr std::array kLow{QueueDesc{.rank = 0, .level = QueueLevel::Normal}};
        constexpr std::array kHigh{QueueDesc{.rank = 1, .level = QueueLevel::Normal}};

        auto opened = rhi.createDevice(adapter, kNoWindow, testDeviceWith([&](DeviceDesc& desc) {
                                           desc.graphics = kLow;
                                           desc.compute = kHigh;
                                       }));

        INFO(failureOf(opened));
        if (queues.ranks == 0) {
            REQUIRE_FALSE(opened.has_value());
            CHECK(opened.error().code == bfr::core::ErrorCode::RhiUnsupported);
        }
        else {
            CHECK(opened.has_value());
        }
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device opens queues at normal, and fails with RhiUnsupported for a level its backend has no setting for",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        auto created = createInstance(recordingInto(backend, messages), kValidation);
        REQUIRE(created.has_value());
        Instance& rhi = **created;
        const std::size_t adapter = headlessAdapter(rhi);
        const QueueLevels& levels = rhi.adapters()[adapter].caps.queues.graphicsLevels;
        constexpr std::array kNormal{QueueDesc{.rank = 0, .level = QueueLevel::Normal}};
        constexpr std::array kHigh{QueueDesc{.rank = 0, .level = QueueLevel::High}};

        const auto normal = rhi.createDevice(adapter, kNoWindow,
                                             testDeviceWith([&](DeviceDesc& desc) { desc.graphics = kNormal; }));
        INFO(failureOf(normal));
        CHECK(normal.has_value());
        if (!levels.normal && !levels.low && !levels.high && !levels.realtime) {
            const auto high = rhi.createDevice(adapter, kNoWindow,
                                               testDeviceWith([&](DeviceDesc& desc) { desc.graphics = kHigh; }));
            REQUIRE_FALSE(high.has_value());
            CHECK(high.error().code == bfr::core::ErrorCode::RhiUnsupported);
        }
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device fails with RhiInvalidCall for a swapchain on a queue it lacks", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        auto created = createInstance(recordingInto(backend, messages), kValidation);
        REQUIRE(created.has_value());
        Instance& rhi = **created;
        auto opened = rhi.createDevice(headlessAdapter(rhi), kHeadless, kTestDevice);
        INFO(failureOf(opened));
        REQUIRE(opened.has_value());
        SwapchainDesc desc = swapchainDesc(rhi);
        desc.queue = 1;

        const auto swapchain = (*opened)->createSwapchain(kHeadless, desc);

        REQUIRE_FALSE(swapchain.has_value());
        CHECK(swapchain.error().code == bfr::core::ErrorCode::RhiInvalidCall);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device holds the lists, pools and swapchains it was opened for, and no more", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        auto created = createInstance(recordingInto(backend, messages), kValidation);
        REQUIRE(created.has_value());
        Instance& rhi = **created;
        const std::size_t adapter = headlessAdapter(rhi);
        constexpr std::uint32_t kLists = 2;
        // A maximum a handle cannot name is refused; 0 is none.
        CHECK(rhi.createDevice(adapter, kHeadless, testDeviceWith([](DeviceDesc& desc) {
                                   desc.maxCommandLists = bfr::core::Handle<void>::kMaxIndices + 1;
                               }))
                  .error()
                  .code == bfr::core::ErrorCode::RhiInvalidCall);
        auto opened = rhi.createDevice(adapter, kHeadless, testDeviceWith([](DeviceDesc& desc) {
                                           desc.maxCommandLists = kLists;
                                           desc.maxCommandPools = 1;
                                           desc.maxSwapchains = 1;
                                       }));
        INFO(failureOf(opened));
        REQUIRE(opened.has_value());
        Device& device = **opened;

        const auto swapchain = device.createSwapchain(kHeadless, swapchainDesc(rhi));
        REQUIRE(swapchain.has_value());
        CHECK(device.createSwapchain(kHeadless, swapchainDesc(rhi)).error().code ==
              bfr::core::ErrorCode::RhiLimitReached);

        CHECK(device.createCommandPool({.queue = 0, .listCount = kLists + 1}).error().code ==
              bfr::core::ErrorCode::RhiLimitReached);
        const auto pool = device.createCommandPool({.queue = 0, .listCount = kLists});
        INFO(failureOf(pool));
        REQUIRE(pool.has_value());
        CHECK(device.createCommandPool({.queue = 0, .listCount = 0}).error().code ==
              bfr::core::ErrorCode::RhiLimitReached);

        // The pool hands out each of its lists once until it is reset.
        const auto first = device.beginCommands(*pool);
        const auto second = device.beginCommands(*pool);
        REQUIRE(first.has_value());
        REQUIRE(second.has_value());
        CHECK(device.beginCommands(*pool).error().code == bfr::core::ErrorCode::RhiLimitReached);
        const std::array lists{*first, *second};
        const auto token = device.submit({.queue = 0, .commandLists = lists, .waits = {}});
        INFO(failureOf(token));
        REQUIRE(token.has_value());
        REQUIRE(device.wait(*token, kTestWait).has_value());
        REQUIRE(device.resetCommandPool(*pool).has_value());
        const auto again = device.beginCommands(*pool);
        REQUIRE(again.has_value());
        CHECK(*again != *first);
        CHECK(*again != *second);
        REQUIRE(device.destroyCommandList(*again).has_value());

        // A destroyed pool's lists and room come back for the next pool.
        REQUIRE(device.destroyCommandPool(*pool).has_value());
        const auto next = device.createCommandPool({.queue = 0, .listCount = kLists});
        INFO(failureOf(next));
        REQUIRE(next.has_value());
        CHECK(device.destroyCommandPool(*next).has_value());
        CHECK(device.destroySwapchain(*swapchain).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}
