// What every backend does from acquiring a swapchain image to presenting it,
// through the public API alone. Each case runs on every backend the build has.

#include "bfr.hpp"
#include "bfr/bfr.h"
#include "bfr_test_support.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_container_properties.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace bfr;
using namespace bfr::test;

namespace {

constexpr std::uint32_t kWidth = 64;
constexpr std::uint32_t kHeight = 48;
constexpr int kFrames = 3;

// The slots a handle can name: its index takes the low 24 of its bits. bfr.h
// does not state it; a maximum past it needs it.
constexpr unsigned kHandleIndexBits = 24;
constexpr std::uint32_t kHandleIndices = std::uint32_t{1} << kHandleIndexBits;

// The swapchain's image, cleared.
constexpr RenderingDesc kClear{.colorAttachmentCount = 1,
                               .colorAttachments = {{{.texture = {},
                                                      .loadOp = LoadOp::Clear,
                                                      .storeOp = StoreOp::Store,
                                                      .clearColor = {0.2F, 0.4F, 0.6F, 1}}}},
                               .depthStencil = {},
                               .occlusionQueries = {}};

Expected<Token> submitOne(const Device& device, CommandList commandList, std::uint32_t queue = 0,
                          std::span<const Token> waits = {}) {
    return device.submit({.queue = queue, .commandLists = std::span{&commandList, 1}, .waits = waits});
}

// Members go in reverse: the pool, the swapchain, the device, the Instance.
struct HeadlessSwapchain {
    explicit HeadlessSwapchain(const InstanceDesc& desc) {
        auto created = createInstance(desc, kValidation);
        INFO(failureOf(created));
        REQUIRE(created.has_value());
        instance = std::move(*created);
        device = presentingDevice(instance, kHeadless);
        auto made = device.createSwapchain(swapchainDesc(instance));
        INFO(failureOf(made));
        REQUIRE(made.has_value());
        swapchain = std::move(*made);
        const auto images = swapchain.resize(kWidth, kHeight);
        INFO(failureOf(images));
        REQUIRE(images.has_value());
        auto madePool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        INFO(failureOf(madePool));
        REQUIRE(madePool.has_value());
        pool = std::move(*madePool);
    }
    ~HeadlessSwapchain() {
        const CommandPoolRef destroyedPool = pool;
        pool = {};
        CHECK(resetResult(device, destroyedPool.get()) == Result::ErrorInvalidHandle);
        const SwapchainRef destroyedSwapchain = swapchain;
        swapchain = {};
        CHECK(resizeResult(device, destroyedSwapchain.get(), kWidth, kHeight) == Result::ErrorInvalidHandle);
    }
    HeadlessSwapchain(const HeadlessSwapchain&) = delete;
    HeadlessSwapchain& operator=(const HeadlessSwapchain&) = delete;
    HeadlessSwapchain(HeadlessSwapchain&&) = delete;
    HeadlessSwapchain& operator=(HeadlessSwapchain&&) = delete;

    Instance instance;
    Device device;
    Swapchain swapchain;
    CommandPool pool;
};

// The first adapter whose test family has room for count queues, and that
// family; none where no adapter has.
std::optional<std::pair<std::uint32_t, std::uint32_t>> roomFor(const Instance& rhi, std::uint32_t count) {
    const auto adapters = static_cast<std::uint32_t>(adaptersOf(rhi).size());
    for (std::uint32_t adapter = 0; adapter < adapters; ++adapter) {
        const std::uint32_t family = testFamily(rhi, adapter, kNoWindow);
        const std::uint32_t has = queueFamiliesOf(rhi, adapter).at(family).count;
        if (has == 0 || has >= count) {
            return std::pair{adapter, family};
        }
    }
    return std::nullopt;
}

} // namespace

TEST_CASE("Device clears and presents a headless swapchain, frame after frame", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        HeadlessSwapchain headless{recordingInto(backend, messages)};
        const Device& device = headless.device;
        Token last;
        for (int frame = 0; frame < kFrames; ++frame) {
            INFO("frame " << frame);
            REQUIRE(device.wait(last, kTestWait).has_value());
            const auto reset = headless.pool.reset();
            INFO(failureOf(reset));
            REQUIRE(reset.has_value());
            const auto acquired = headless.swapchain.acquire(kTestWait);
            INFO(failureOf(acquired));
            REQUIRE(acquired.has_value());
            const auto commands = headless.pool.beginCommands();
            INFO(failureOf(commands));
            REQUIRE(commands.has_value());
            CHECK(device.beginRendering(*commands, headless.swapchain, kClear).has_value());
            CHECK(device.endRendering(*commands).has_value());
            const auto submitted = submitOne(device, *commands);
            INFO(failureOf(submitted));
            REQUIRE(submitted.has_value());
            CHECK(submitted->value > last.value);
            last = *submitted;
            const auto presented = headless.swapchain.present();
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
        HeadlessSwapchain headless{recordingInto(backend, messages)};
        const Device& device = headless.device;
        const Format format = swapchainDesc(headless.instance).format;
        constexpr std::uint32_t kSamples = 4;
        const auto caps = formatCapsOf(headless.instance, headlessAdapter(headless.instance), format);
        INFO(failureOf(caps));
        REQUIRE(caps.has_value());
        if ((caps->sampleCounts & kSamples) == 0 || !caps->resolve) {
            SKIP("the adapter does not resolve the swapchain's format from 4 samples");
        }
        auto texture = device.createTexture({.dimension = TextureDimension::D2,
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
        const auto acquired = headless.swapchain.acquire(kTestWait);
        INFO(failureOf(acquired));
        REQUIRE(acquired.has_value());
        const auto commands = headless.pool.beginCommands();
        REQUIRE(commands.has_value());
        CHECK(textureBarrier(device, *commands, *texture, TextureLayout::Undefined, TextureLayout::ColorTarget, {})
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
        const auto presented = headless.swapchain.present();
        INFO(failureOf(presented));
        CHECK(presented.has_value());
        CHECK(device.wait(*submitted, kTestWait).has_value());
        const TextureRef destroyedTexture = *texture;
        *texture = {};
        CHECK(slotsResult(device, destroyedTexture.get()) == Result::ErrorInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device refuses to acquire from a swapchain with no images, and says to resize it", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        HeadlessSwapchain headless{recordingInto(backend, messages)};
        REQUIRE(headless.swapchain.resize(0, 0).has_value());

        const auto acquired = headless.swapchain.acquire(kTestWait);

        REQUIRE_FALSE(acquired.has_value());
        CHECK(acquired.error().code == Result::ErrorOutOfDate);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device refuses a swapchain call with no image acquired with RhiInvalidCall", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        HeadlessSwapchain headless{recordingInto(backend, messages)};
        const Device& device = headless.device;
        const auto commands = headless.pool.beginCommands();
        REQUIRE(commands.has_value());

        // Rendering to a swapchain, or presenting it, needs an acquired image.
        CHECK(device.beginRendering(*commands, headless.swapchain, kClear).error().code ==
              Result::ErrorInvalidArgument);
        CHECK(headless.swapchain.present().error().code == Result::ErrorInvalidArgument);

        REQUIRE(headless.swapchain.acquire(kTestWait).has_value());
        REQUIRE(device.beginRendering(*commands, headless.swapchain, kClear).has_value());
        REQUIRE(device.endRendering(*commands).has_value());
        REQUIRE(submitOne(device, *commands).has_value());
        CHECK(headless.swapchain.present().has_value());

        // A submitted list is gone.
        CHECK(submitOne(device, *commands).error().code == Result::ErrorInvalidHandle);
        // d1: stale destroy reported as an Error message (track B)
        device.destroyCommandList(*commands);
        // d1: null destroy: no message (track B)
        device.destroyCommandList({});
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device completes a default token at once, and has not passed one no submit returned", "[rhi][agnostic]") {
    constexpr std::uint64_t kNoWait = 0;
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const HeadlessSwapchain headless{recordingInto(backend, messages)};
        const Device& device = headless.device;

        CHECK(device.poll(Token{}).value_or(false));
        CHECK(device.wait(Token{}, kTestWait).has_value());
        CHECK(device.wait(Token{.queue = 1, .value = 0}, kTestWait).error().code == Result::ErrorInvalidArgument);
        const auto ahead = device.poll(Token{.queue = 0, .value = 1});
        INFO(failureOf(ahead));
        CHECK_FALSE(ahead.value_or(true));
        const auto waited = device.wait(Token{.queue = 0, .value = 1}, kNoWait);
        REQUIRE_FALSE(waited.has_value());
        CHECK(waited.error().code == Result::ErrorTimeout);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device resets a command pool once the GPU has run its lists", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        HeadlessSwapchain headless{recordingInto(backend, messages)};
        const Device& device = headless.device;

        const auto commands = headless.pool.beginCommands();
        REQUIRE(commands.has_value());
        // Vulkan's reset ends a list still recording; D3D12 and Metal refuse it.
        if (backend.backend == Backend::Vulkan) {
            CHECK(headless.pool.reset().has_value());
            CHECK(submitOne(device, *commands).error().code == Result::ErrorInvalidHandle);
            // d1: stale destroy reported as an Error message (track B)
            device.destroyCommandList(*commands);
        }
        else {
            CHECK(headless.pool.reset().error().code == Result::ErrorInvalidArgument);
            device.destroyCommandList(*commands);
            REQUIRE(submitOne(device, *commands).error().code == Result::ErrorInvalidHandle);
            CHECK(headless.pool.reset().has_value());
        }

        REQUIRE(headless.swapchain.acquire(kTestWait).has_value());
        const auto frame = headless.pool.beginCommands();
        REQUIRE(frame.has_value());
        REQUIRE(device.beginRendering(*frame, headless.swapchain, kClear).has_value());
        REQUIRE(device.endRendering(*frame).has_value());
        const auto token = submitOne(device, *frame);
        REQUIRE(token.has_value());
        REQUIRE(headless.swapchain.present().has_value());
        REQUIRE(device.wait(*token, kTestWait).has_value());
        CHECK(device.poll(*token).value_or(false));
        CHECK(headless.pool.reset().has_value());

        auto other = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(other.has_value());
        const auto orphan = other->beginCommands();
        REQUIRE(orphan.has_value());
        const CommandPoolRef destroyedOther = *other;
        *other = {};
        bfr_command_list list{};
        CHECK(resultOf(bfr_begin_commands(device.get(), destroyedOther.get(), &list)) == Result::ErrorInvalidHandle);
        CHECK(resetResult(device, destroyedOther.get()) == Result::ErrorInvalidHandle);
        // d1: stale destroy reported as an Error message (track B)
        bfr_destroy_command_pool(device.get(), destroyedOther.get());
        // d1: null destroy: no message (track B)
        bfr_destroy_command_pool(device.get(), CommandPoolRef{}.get());
        // The destroyed pool's lists went with it.
        CHECK(submitOne(device, *orphan).error().code == Result::ErrorInvalidHandle);
        // d1: stale destroy reported as an Error message (track B)
        device.destroyCommandList(*orphan);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device refuses a pool, a submit and a token on a queue it does not have", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const HeadlessSwapchain headless{recordingInto(backend, messages)};
        const Device& device = headless.device;
        constexpr std::uint32_t kMissing = 7;

        CHECK(device.createCommandPool({.queue = kMissing, .listCount = kPoolLists}).error().code ==
              Result::ErrorInvalidArgument);
        CHECK(device.submit({.queue = kMissing, .commandLists = {}, .waits = {}}).error().code ==
              Result::ErrorInvalidArgument);
        CHECK(device.wait(Token{.queue = kMissing, .value = 0}, kTestWait).error().code ==
              Result::ErrorInvalidArgument);
        CHECK(device.poll(Token{.queue = kMissing, .value = 0}).error().code == Result::ErrorInvalidArgument);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device submits several lists at once and each waits only on what it is told to", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        HeadlessSwapchain headless{recordingInto(backend, messages)};
        const Device& device = headless.device;

        REQUIRE(headless.swapchain.acquire(kTestWait).has_value());
        const auto first = headless.pool.beginCommands();
        const auto second = headless.pool.beginCommands();
        REQUIRE(first.has_value());
        REQUIRE(second.has_value());
        REQUIRE(device.beginRendering(*second, headless.swapchain, kClear).has_value());
        REQUIRE(device.endRendering(*second).has_value());
        const std::array lists{*first, *second};
        const auto token = device.submit({.queue = 0, .commandLists = lists, .waits = {}});
        INFO(failureOf(token));
        REQUIRE(token.has_value());
        REQUIRE(headless.swapchain.present().has_value());

        // A later submit waits on an earlier token.
        const auto third = headless.pool.beginCommands();
        REQUIRE(third.has_value());
        const auto later = submitOne(device, *third, 0, std::span{&*token, 1});
        INFO(failureOf(later));
        REQUIRE(later.has_value());
        CHECK(later->value > token->value);
        REQUIRE(device.wait(*later, kTestWait).has_value());
        CHECK(device.poll(*token).value_or(false));
        // A list submitted beside another is gone with it.
        CHECK(submitOne(device, *first).error().code == Result::ErrorInvalidHandle);
        // d1: stale destroy reported as an Error message (track B)
        device.destroyCommandList(*first);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device runs a queue of a family without graphics beside graphics when the adapter has one",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        auto created = createInstance(recordingInto(backend, messages), kValidation);
        REQUIRE(created.has_value());
        const Instance& rhi = *created;
        const std::uint32_t adapter = headlessAdapter(rhi);
        const auto computeFamily = familyWhere(rhi, adapter, [](const QueueFamily& family) {
            return family.compute && !family.graphics;
        });
        std::vector<QueueDesc> queues{testQueues(testFamily(rhi, adapter, kHeadless)).front()};
        if (computeFamily) {
            queues.push_back(testQueues(*computeFamily).front());
        }
        auto opened = rhi.createDevice(
            testDeviceWith(rhi, adapter, kHeadless, [&queues](DeviceDesc& desc) { desc.queues = queues; }));
        INFO(failureOf(opened));
        REQUIRE(opened.has_value());
        const Device& device = *opened;
        if (!computeFamily) {
            CHECK(device.createCommandPool({.queue = 1, .listCount = kPoolLists}).error().code ==
                  Result::ErrorInvalidArgument);
            return;
        }

        auto graphicsPool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        auto computePool = device.createCommandPool({.queue = 1, .listCount = kPoolLists});
        REQUIRE(graphicsPool.has_value());
        REQUIRE(computePool.has_value());
        const auto graphics = graphicsPool->beginCommands();
        const auto compute = computePool->beginCommands();
        REQUIRE(graphics.has_value());
        REQUIRE(compute.has_value());

        // A compute list neither renders nor goes to the graphics queue.
        auto swapchain = device.createSwapchain(swapchainDesc(rhi));
        REQUIRE(swapchain.has_value());
        REQUIRE(swapchain->resize(kWidth, kHeight).has_value());
        REQUIRE(swapchain->acquire(kTestWait).has_value());
        CHECK(device.beginRendering(*compute, *swapchain, kClear).error().code == Result::ErrorInvalidArgument);
        CHECK(submitOne(device, *compute, 0).error().code == Result::ErrorInvalidArgument);

        // Compute waits on graphics across queues, and both tokens complete.
        REQUIRE(device.beginRendering(*graphics, *swapchain, kClear).has_value());
        REQUIRE(device.endRendering(*graphics).has_value());
        const auto rendered = submitOne(device, *graphics, 0);
        INFO(failureOf(rendered));
        REQUIRE(rendered.has_value());
        REQUIRE(swapchain->present().has_value());
        const auto computed = submitOne(device, *compute, 1, std::span{&*rendered, 1});
        INFO(failureOf(computed));
        REQUIRE(computed.has_value());
        CHECK(computed->queue == 1);
        REQUIRE(device.wait(*computed, kTestWait).has_value());
        CHECK(device.poll(*rendered).value_or(false));
        CHECK(computePool->reset().has_value());
        CHECK(graphicsPool->reset().has_value());
        const CommandPoolRef destroyedCompute = *computePool;
        *computePool = {};
        CHECK(resetResult(device, destroyedCompute.get()) == Result::ErrorInvalidHandle);
        const CommandPoolRef destroyedGraphics = *graphicsPool;
        *graphicsPool = {};
        CHECK(resetResult(device, destroyedGraphics.get()) == Result::ErrorInvalidHandle);
        const SwapchainRef destroyedSwapchain = *swapchain;
        *swapchain = {};
        CHECK(resizeResult(device, destroyedSwapchain.get(), kWidth, kHeight) == Result::ErrorInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device frees a pool destroyed while the GPU runs its lists at the garbage collection after the GPU is done",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        auto created = createInstance(recordingInto(backend, messages), kValidation);
        REQUIRE(created.has_value());
        const Instance& rhi = *created;
        auto opened = rhi.createDevice(testDeviceWith(rhi, headlessAdapter(rhi), kHeadless, [](DeviceDesc& desc) {
            desc.maxCommandLists = kPoolLists;
            desc.maxCommandPools = 1;
        }));
        INFO(failureOf(opened));
        REQUIRE(opened.has_value());
        const Device& device = *opened;
        auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        const auto list = pool->beginCommands();
        REQUIRE(list.has_value());
        const auto token = device.submit({.queue = 0, .commandLists = std::span{&*list, 1}, .waits = {}});
        INFO(failureOf(token));
        REQUIRE(token.has_value());

        // Destroyed before the GPU is known to be done, the pool is gone from
        // the caller at once, and its room is back at the first garbage
        // collection after the GPU has passed it.
        const CommandPoolRef destroyedPool = *pool;
        *pool = {};
        CHECK(resetResult(device, destroyedPool.get()) == Result::ErrorInvalidHandle);
        REQUIRE(device.wait(*token, kTestWait).has_value());
        CHECK(device.createCommandPool({.queue = 0, .listCount = kPoolLists}).error().code ==
              Result::ErrorLimitReached);
        REQUIRE(device.runGarbageCollection().has_value());
        auto next = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        INFO(failureOf(next));
        REQUIRE(next.has_value());
        const CommandPoolRef destroyedNext = *next;
        *next = {};
        CHECK(resetResult(device, destroyedNext.get()) == Result::ErrorInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device serves pools made, submitted, waited on and destroyed from several threads at once",
          "[rhi][agnostic]") {
    // A destroyed pool holds its room until garbage is collected, so the
    // device holds a pool for each round of each thread. Catch2 is not called
    // off the test thread: each thread records what failed, checked after
    // joining.
    constexpr std::uint32_t kThreads = 4;
    constexpr std::uint32_t kRounds = 8;
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        auto created = createInstance(recordingInto(backend, messages), kValidation);
        REQUIRE(created.has_value());
        const Instance& rhi = *created;
        auto opened = rhi.createDevice(testDeviceWith(rhi, headlessAdapter(rhi), kHeadless, [](DeviceDesc& desc) {
            desc.maxCommandLists = kThreads * kRounds * kPoolLists;
            desc.maxCommandPools = kThreads * kRounds;
        }));
        INFO(failureOf(opened));
        REQUIRE(opened.has_value());
        const Device& device = *opened;

        std::array<std::string, kThreads> failures;
        {
            std::vector<std::jthread> threads;
            threads.reserve(kThreads);
            for (std::string& failure : failures) {
                threads.emplace_back([&device, &failure] {
                    for (std::uint32_t round = 0; round < kRounds && failure.empty(); ++round) {
                        auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
                        if (!pool) {
                            failure = "createCommandPool: " + pool.error().message.text();
                            return;
                        }
                        const auto list = pool->beginCommands();
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
                        const CommandPoolRef destroyedPool = *pool;
                        *pool = {};
                        if (const Result reset = resetResult(device, destroyedPool.get());
                            reset != Result::ErrorInvalidHandle) {
                            failure = "a destroyed pool reset with result " + std::to_string(std::to_underlying(reset));
                            return;
                        }
                    }
                });
            }
        }
        for (const std::string& failure : failures) {
            CHECK(failure.empty());
        }

        // Every pool's room is back once garbage is collected.
        REQUIRE(device.runGarbageCollection().has_value());
        std::vector<CommandPool> pools;
        for (std::uint32_t index = 0; index < kThreads * kRounds; ++index) {
            auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
            INFO(failureOf(pool));
            REQUIRE(pool.has_value());
            pools.push_back(std::move(*pool));
        }
        for (CommandPool& pool : pools) {
            const CommandPoolRef destroyedPool = pool;
            pool = {};
            CHECK(resetResult(device, destroyedPool.get()) == Result::ErrorInvalidHandle);
        }
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device opens the queues of a family it is asked for, numbered from 0", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        auto created = createInstance(recordingInto(backend, messages), kValidation);
        REQUIRE(created.has_value());
        const Instance& rhi = *created;
        constexpr std::uint32_t kQueues = 2;
        const auto found = roomFor(rhi, kQueues);
        if (!found) {
            SKIP("no adapter's family that runs graphics has " << kQueues << " queues");
        }
        const std::span<const QueueDesc> queues = testQueues(found->second, kQueues);
        auto opened = rhi.createDevice(
            testDeviceWith(rhi, found->first, kNoWindow, [queues](DeviceDesc& desc) { desc.queues = queues; }));
        INFO(failureOf(opened));
        REQUIRE(opened.has_value());
        const Device& device = *opened;
        // Queue 1 is the family's second queue: its lists submit to it.
        auto pool = device.createCommandPool({.queue = 1, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        const auto list = pool->beginCommands();
        REQUIRE(list.has_value());
        const auto token = device.submit({.queue = 1, .commandLists = std::span{&*list, 1}, .waits = {}});
        INFO(failureOf(token));
        REQUIRE(token.has_value());
        CHECK(device.wait(*token, kTestWait).has_value());
        const CommandPoolRef destroyedPool = *pool;
        *pool = {};
        CHECK(resetResult(device, destroyedPool.get()) == Result::ErrorInvalidHandle);
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
        const Instance& rhi = *created;
        constexpr std::uint32_t kQueues = 2;
        const auto found = roomFor(rhi, kQueues);
        if (!found) {
            SKIP("no adapter's family that runs graphics has " << kQueues << " queues to rank");
        }
        const auto [adapter, family] = *found;
        const std::array ranked{QueueDesc{.family = family, .rank = 0, .level = QueueLevel::Normal},
                                QueueDesc{.family = family, .rank = 1, .level = QueueLevel::Normal}};

        auto opened = rhi.createDevice(
            testDeviceWith(rhi, adapter, kNoWindow, [&ranked](DeviceDesc& desc) { desc.queues = ranked; }));

        INFO(failureOf(opened));
        if (capsOf(rhi, adapter).queueRanks == 0) {
            REQUIRE_FALSE(opened.has_value());
            CHECK(opened.error().code == Result::ErrorUnsupported);
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
        const Instance& rhi = *created;
        const std::uint32_t adapter = headlessAdapter(rhi);
        const std::uint32_t family = testFamily(rhi, adapter, kNoWindow);
        const QueueLevels levels = queueFamiliesOf(rhi, adapter).at(family).levels;
        const std::array normalQueue{QueueDesc{.family = family, .rank = 0, .level = QueueLevel::Normal}};
        const std::array highQueue{QueueDesc{.family = family, .rank = 0, .level = QueueLevel::High}};

        const auto normal = rhi.createDevice(
            testDeviceWith(rhi, adapter, kNoWindow, [&](DeviceDesc& desc) { desc.queues = normalQueue; }));
        INFO(failureOf(normal));
        CHECK(normal.has_value());
        if (!levels.normal && !levels.low && !levels.high && !levels.realtime) {
            const auto high = rhi.createDevice(
                testDeviceWith(rhi, adapter, kNoWindow, [&](DeviceDesc& desc) { desc.queues = highQueue; }));
            REQUIRE_FALSE(high.has_value());
            CHECK(high.error().code == Result::ErrorUnsupported);
        }
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Instance lists each adapter's queue families and finds one by what it runs and where it presents",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        auto created = createInstance(recordingInto(backend, messages), kValidation);
        REQUIRE(created.has_value());
        const Instance& rhi = *created;
        const std::uint32_t adapter = headlessAdapter(rhi);
        const std::vector<QueueFamily> families = queueFamiliesOf(rhi, adapter);
        REQUIRE_FALSE(families.empty());

        const auto any = rhi.findQueueFamily(adapter, {});
        const auto presenting = rhi.findQueueFamily(adapter, {.graphics = true, .presents = &kHeadless});

        INFO(failureOf(any) << ' ' << failureOf(presenting));
        REQUIRE(any.has_value());
        CHECK(*any == 0);
        REQUIRE(presenting.has_value());
        CHECK(families.at(*presenting).graphics);
        const auto presents = rhi.queueFamilyPresents(adapter, *presenting, {.window = kHeadless});
        INFO(failureOf(presents));
        REQUIRE(presents.has_value());
        CHECK(*presents);
        // No family before it both runs graphics and presents.
        for (std::uint32_t family = 0; family < *presenting; ++family) {
            const auto earlier = rhi.queueFamilyPresents(adapter, family, {.window = kHeadless});
            REQUIRE(earlier.has_value());
            CHECK_FALSE((families.at(family).graphics && *earlier));
        }
        const auto past = static_cast<std::uint32_t>(families.size());
        CHECK(rhi.queueFamilyPresents(adapter, past, {.window = kHeadless}).error().code ==
              Result::ErrorInvalidArgument);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device fails with RhiInvalidCall for a queue of a family the adapter lacks, or more than a family has",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        auto created = createInstance(recordingInto(backend, messages), kValidation);
        REQUIRE(created.has_value());
        const Instance& rhi = *created;
        const std::vector<QueueFamily> families = queueFamiliesOf(rhi, kFirstAdapter);
        const auto past = static_cast<std::uint32_t>(families.size());

        const auto lacked = rhi.createDevice(testDeviceWith(rhi, kFirstAdapter, kNoWindow, [past](DeviceDesc& desc) {
            desc.queues = testQueues(past);
        }));

        REQUIRE_FALSE(lacked.has_value());
        CHECK(lacked.error().code == Result::ErrorInvalidArgument);
        const auto counted = std::ranges::find_if(families, [](const QueueFamily& family) {
            return family.count != 0 && family.count < kTestFamilyQueues;
        });
        if (counted == families.end()) {
            SKIP("no family counts fewer queues than the tests open");
        }
        const auto family = static_cast<std::uint32_t>(counted - families.begin());
        const auto tooMany = rhi.createDevice(testDeviceWith(rhi, kFirstAdapter, kNoWindow, [&](DeviceDesc& desc) {
            desc.queues = testQueues(family, std::size_t{counted->count} + 1);
        }));
        REQUIRE_FALSE(tooMany.has_value());
        CHECK(tooMany.error().code == Result::ErrorInvalidArgument);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device fails with RhiPresentUnsupported for a swapchain on a queue whose family doesn't present",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        auto created = createInstance(recordingInto(backend, messages), kValidation);
        REQUIRE(created.has_value());
        const Instance& rhi = *created;
        // A family of an adapter that presents headless whose own answer is
        // no, if the backend has one.
        const std::uint32_t adapter = headlessAdapter(rhi);
        const auto families = static_cast<std::uint32_t>(queueFamiliesOf(rhi, adapter).size());
        std::optional<std::uint32_t> silent;
        for (std::uint32_t family = 0; family < families && !silent; ++family) {
            const auto presents = rhi.queueFamilyPresents(adapter, family, {.window = kHeadless});
            INFO(failureOf(presents));
            REQUIRE(presents.has_value());
            if (!*presents) {
                silent = family;
            }
        }
        if (!silent) {
            SKIP("every family of the adapter presents to a headless window");
        }
        auto opened = rhi.createDevice(testDeviceWith(rhi, adapter, kNoWindow, [&silent](DeviceDesc& desc) {
            desc.queues = testQueues(*silent);
        }));
        INFO(failureOf(opened));
        REQUIRE(opened.has_value());

        const auto swapchain = opened->createSwapchain(swapchainDesc(rhi));

        REQUIRE_FALSE(swapchain.has_value());
        CHECK(swapchain.error().code == Result::ErrorPresentUnsupported);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device fails with RhiInvalidCall for a swapchain on a queue it lacks", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        auto created = createInstance(recordingInto(backend, messages), kValidation);
        REQUIRE(created.has_value());
        const Instance& rhi = *created;
        auto opened = rhi.createDevice(testDeviceOn(rhi, headlessAdapter(rhi), kHeadless));
        INFO(failureOf(opened));
        REQUIRE(opened.has_value());
        SwapchainDesc desc = swapchainDesc(rhi);
        desc.queue = 1;

        const auto swapchain = opened->createSwapchain(desc);

        REQUIRE_FALSE(swapchain.has_value());
        CHECK(swapchain.error().code == Result::ErrorInvalidArgument);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device holds the lists, pools and swapchains it was opened for, and no more", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        auto created = createInstance(recordingInto(backend, messages), kValidation);
        REQUIRE(created.has_value());
        const Instance& rhi = *created;
        const std::uint32_t adapter = headlessAdapter(rhi);
        constexpr std::uint32_t kLists = 2;
        // A maximum a handle cannot name is refused; 0 is none.
        CHECK(rhi.createDevice(testDeviceWith(rhi, adapter, kHeadless,
                                              [](DeviceDesc& desc) { desc.maxCommandLists = kHandleIndices + 1; }))
                  .error()
                  .code == Result::ErrorInvalidArgument);
        auto opened = rhi.createDevice(testDeviceWith(rhi, adapter, kHeadless, [](DeviceDesc& desc) {
            desc.maxCommandLists = kLists;
            desc.maxCommandPools = 1;
            desc.maxSwapchains = 1;
        }));
        INFO(failureOf(opened));
        REQUIRE(opened.has_value());
        const Device& device = *opened;

        auto swapchain = device.createSwapchain(swapchainDesc(rhi));
        REQUIRE(swapchain.has_value());
        CHECK(device.createSwapchain(swapchainDesc(rhi)).error().code == Result::ErrorLimitReached);

        CHECK(device.createCommandPool({.queue = 0, .listCount = kLists + 1}).error().code ==
              Result::ErrorLimitReached);
        auto pool = device.createCommandPool({.queue = 0, .listCount = kLists});
        INFO(failureOf(pool));
        REQUIRE(pool.has_value());
        CHECK(device.createCommandPool({.queue = 0, .listCount = 0}).error().code == Result::ErrorLimitReached);

        // The pool hands out each of its lists once until it is reset.
        const auto first = pool->beginCommands();
        const auto second = pool->beginCommands();
        REQUIRE(first.has_value());
        REQUIRE(second.has_value());
        CHECK(pool->beginCommands().error().code == Result::ErrorLimitReached);
        const std::array lists{*first, *second};
        const auto token = device.submit({.queue = 0, .commandLists = lists, .waits = {}});
        INFO(failureOf(token));
        REQUIRE(token.has_value());
        REQUIRE(device.wait(*token, kTestWait).has_value());
        REQUIRE(pool->reset().has_value());
        const auto again = pool->beginCommands();
        REQUIRE(again.has_value());
        // CommandList's defaulted == is deleted: bfr_command_list has none.
        CHECK(again->get().bits != first->get().bits);
        CHECK(again->get().bits != second->get().bits);
        device.destroyCommandList(*again);
        REQUIRE(submitOne(device, *again).error().code == Result::ErrorInvalidHandle);

        // A destroyed pool's lists and room come back for the next pool once
        // garbage is collected.
        const CommandPoolRef destroyedPool = *pool;
        *pool = {};
        REQUIRE(resetResult(device, destroyedPool.get()) == Result::ErrorInvalidHandle);
        CHECK(device.createCommandPool({.queue = 0, .listCount = kLists}).error().code == Result::ErrorLimitReached);
        REQUIRE(device.runGarbageCollection().has_value());
        auto next = device.createCommandPool({.queue = 0, .listCount = kLists});
        INFO(failureOf(next));
        REQUIRE(next.has_value());
        const CommandPoolRef destroyedNext = *next;
        *next = {};
        CHECK(resetResult(device, destroyedNext.get()) == Result::ErrorInvalidHandle);
        const SwapchainRef destroyedSwapchain = *swapchain;
        *swapchain = {};
        CHECK(resizeResult(device, destroyedSwapchain.get(), kWidth, kHeight) == Result::ErrorInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}
