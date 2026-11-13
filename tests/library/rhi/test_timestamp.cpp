// What every backend does with GPU timestamps and call timings, through the
// interface alone. Each case runs on every backend the build has, and writes
// only what the adapter's caps list.

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/call_timing.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/instance.h"
#include "bfr/rhi/texture.h"
#include "bfr/rhi/timestamp.h"
#include "rhi_test_support.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_container_properties.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace bfr::rhi;
using namespace bfr::rhi::test;
using bfr::core::ErrorCode;

namespace {

constexpr std::uint32_t kSide = 16;
constexpr std::uint64_t kReadbackBytes = 4096;

// An Instance and a device on its first adapter, presenting nowhere.
struct TimestampDevice {
    explicit TimestampDevice(const InstanceDesc& desc) {
        auto created = createInstance(desc, kValidation);
        INFO(failureOf(created));
        REQUIRE(created.has_value());
        instance = std::move(*created);
        auto made = instance->createDevice(0, kNoWindow, kTestDevice);
        INFO(failureOf(made));
        REQUIRE(made.has_value());
        device = std::move(*made);
    }

    [[nodiscard]] const Timestamps& caps() const { return instance->adapters()[0].caps.timestamps; }

    std::unique_ptr<Instance> instance;
    std::unique_ptr<Device> device;
};

// Each stage that stages lists.
std::vector<TimestampStage> listed(const TimestampStages& stages) {
    std::vector<TimestampStage> out;
    for (const auto& [has, stage] :
         {std::pair{stages.end, TimestampStage::End}, std::pair{stages.start, TimestampStage::Start},
          std::pair{stages.vertex, TimestampStage::Vertex}, std::pair{stages.fragment, TimestampStage::Fragment},
          std::pair{stages.compute, TimestampStage::Compute}, std::pair{stages.copy, TimestampStage::Copy},
          std::pair{stages.task, TimestampStage::Task}, std::pair{stages.mesh, TimestampStage::Mesh}}) {
        if (has) {
            out.push_back(stage);
        }
    }
    return out;
}

// The ticks a buffer holds, entryBytes apart.
std::vector<std::uint64_t> ticksIn(std::span<const std::byte> bytes, std::uint32_t entryBytes, std::size_t count) {
    std::vector<std::uint64_t> ticks(count);
    for (std::size_t index = 0; index < count; ++index) {
        std::memcpy(&ticks[index], bytes.subspan(index * entryBytes).data(), sizeof(std::uint64_t));
    }
    return ticks;
}

} // namespace

TEST_CASE("Device writes a timestamp at each stage the caps list, in rendering and out, and resolves them as ticks",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const TimestampDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        const Timestamps& caps = made.caps();
        if (!caps.graphics) {
            SKIP("the adapter's graphics queues write no timestamps");
        }
        REQUIRE(caps.entryBytes >= sizeof(std::uint64_t));
        const std::vector<TimestampStage> outside = listed(caps.outsideRendering);
        const std::vector<TimestampStage> inside = listed(caps.insideRendering);
        // The outside stages, then the inside ones, then one more outside
        // after rendering, which no earlier timestamp comes after.
        const auto count = static_cast<std::uint32_t>(outside.size() + inside.size() + 1);
        const auto pool = device.createTimestampPool({.queue = 0, .count = count});
        INFO(failureOf(pool));
        REQUIRE(pool.has_value());
        const auto readback = device.createBuffer(
            {.size = kReadbackBytes, .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
        REQUIRE(readback.has_value());
        const auto target = device.createTexture({.dimension = TextureDimension::D2,
                                                  .format = Format::Rgba8Unorm,
                                                  .width = kSide,
                                                  .height = kSide,
                                                  .depth = 1,
                                                  .layers = 1,
                                                  .mipLevels = 1,
                                                  .samples = 1,
                                                  .usage = {.colorAttachment = true}});
        REQUIRE(target.has_value());
        const auto commands = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(commands.has_value());
        auto list = device.beginCommands(*commands);
        REQUIRE(list.has_value());

        CHECK(device.resetTimestamps(*list, *pool, 0, count).has_value());
        std::uint32_t index = 0;
        for (const TimestampStage stage : outside) {
            INFO("outside rendering, stage " << static_cast<int>(stage));
            CHECK(device.writeTimestamp(*list, *pool, index++, stage, TimestampGranularity::Relaxed).has_value());
        }
        CHECK(device.textureBarrier(*list, *target, TextureLayout::Undefined, TextureLayout::ColorTarget, {})
                  .has_value());
        RenderingDesc rendering{.colorAttachmentCount = 1,
                                .colorAttachments = {},
                                .depthStencil = {},
                                .occlusionQueries = {}};
        rendering.colorAttachments[0] = {.texture = *target, .loadOp = LoadOp::Clear, .storeOp = StoreOp::Store};
        REQUIRE(device.beginRendering(*list, {}, rendering).has_value());
        for (const TimestampStage stage : inside) {
            INFO("inside rendering, stage " << static_cast<int>(stage));
            CHECK(device.writeTimestamp(*list, *pool, index++, stage, TimestampGranularity::Relaxed).has_value());
        }
        REQUIRE(device.endRendering(*list).has_value());
        const std::uint32_t last = index;
        CHECK(
            device.writeTimestamp(*list, *pool, last, TimestampStage::End, TimestampGranularity::Relaxed).has_value());
        CHECK(device.resolveTimestamps(*list, *pool, 0, count, *readback, 0).has_value());
        const auto sent = device.submit({.queue = 0, .commandLists = std::span{&*list, 1}, .waits = {}});
        INFO(failureOf(sent));
        REQUIRE(sent.has_value());
        REQUIRE(device.wait(*sent, kTestWait).has_value());

        const auto bytes = device.mapBuffer(*readback);
        REQUIRE(bytes.has_value());
        const std::vector<std::uint64_t> ticks = ticksIn(*bytes, caps.entryBytes, count);
        // The last waits for all the work before it, the first for none.
        CHECK(ticks[last] >= ticks[0]);
        CHECK(ticks[last] != 0);

        CHECK(device.destroyCommandPool(*commands).has_value());
        CHECK(device.destroyTexture(*target).has_value());
        CHECK(device.destroyBuffer(*readback).has_value());
        CHECK(device.destroyTimestampPool(*pool).has_value());
        CHECK(device.destroyTimestampPool(*pool).error().code == ErrorCode::RhiInvalidHandle);
        CHECK(device.destroyTimestampPool({}).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device gives a queue's timestamp rate in the backend's one unit, and refuses a queue it lacks",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const TimestampDevice made{recordingInto(backend, messages)};
        if (!made.caps().graphics) {
            SKIP("the adapter's graphics queues write no timestamps");
        }
        const auto rate = made.device->timestampRate(0);
        INFO(failureOf(rate));
        REQUIRE(rate.has_value());
        CHECK((rate->periodNs > 0) != (rate->frequency != 0));
        CHECK(made.device->timestampRate(1).error().code == ErrorCode::RhiInvalidCall);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device reads each host clock the caps list beside a GPU timestamp, and refuses the others",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const TimestampDevice made{recordingInto(backend, messages)};
        const HostClocks& clocks = made.caps().hostClocks;
        for (const auto& [has, clock] : {std::pair{clocks.monotonic, HostClock::Monotonic},
                                         std::pair{clocks.monotonicRaw, HostClock::MonotonicRaw},
                                         std::pair{clocks.performanceCounter, HostClock::PerformanceCounter}}) {
            INFO("host clock " << static_cast<int>(clock));
            const auto taken = made.device->calibrateTimestamps(0, clock);
            if (!has) {
                CHECK(taken.error().code == ErrorCode::RhiUnsupported);
                continue;
            }
            INFO(failureOf(taken));
            REQUIRE(taken.has_value());
            CHECK(taken->gpu != 0);
            CHECK(taken->host != 0);
        }
        CHECK(made.device->calibrateTimestamps(0, static_cast<HostClock>(99)).error().code ==
              ErrorCode::RhiInvalidCall);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device refuses a timestamp stage or granularity BFR doesn't have", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const TimestampDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        const auto pool = device.createTimestampPool({.queue = 0, .count = 1});
        REQUIRE(pool.has_value());
        const auto commands = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(commands.has_value());
        auto list = device.beginCommands(*commands);
        REQUIRE(list.has_value());

        constexpr auto kPast = 99;
        CHECK(device.writeTimestamp(*list, *pool, 0, static_cast<TimestampStage>(kPast), TimestampGranularity::Relaxed)
                  .error()
                  .code == ErrorCode::RhiInvalidCall);
        CHECK(device.writeTimestamp(*list, *pool, 0, TimestampStage::End, static_cast<TimestampGranularity>(kPast))
                  .error()
                  .code == ErrorCode::RhiInvalidCall);
        CHECK(device.createTimestampPool({.queue = 1, .count = 1}).error().code == ErrorCode::RhiInvalidCall);
        CHECK(device.destroyCommandList(*list).has_value());
        CHECK(device.destroyCommandPool(*commands).has_value());
        CHECK(device.destroyTimestampPool(*pool).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device counts the CPU time of each call when the InstanceDesc asks for it, and only then",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    constexpr std::uint64_t kBuffers = 3;
    std::vector<std::string> messages;
    {
        InstanceDesc desc = recordingInto(backend, messages);
        desc.callTimings = true;
        const TimestampDevice made{desc};
        Device& device = *made.device;
        for (std::uint64_t round = 0; round < kBuffers; ++round) {
            const auto buffer = device.createBuffer(
                {.size = kReadbackBytes, .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
            REQUIRE(buffer.has_value());
            CHECK(device.destroyBuffer(*buffer).has_value());
        }

        std::array<CallTiming, kCallCount> timings{};
        REQUIRE(device.callTimings(timings).has_value());
        const CallTiming& creates = timings.at(static_cast<std::size_t>(Call::CreateBuffer));
        CHECK(creates.count == kBuffers);
        CHECK(creates.totalNs >= creates.maxNs);
        CHECK(creates.maxNs != 0);
        CHECK(timings.at(static_cast<std::size_t>(Call::DestroyBuffer)).count == kBuffers);
        CHECK(timings.at(static_cast<std::size_t>(Call::Draw)).count == 0);

        std::array<CallTiming, kCallCount> own{};
        REQUIRE(made.instance->callTimings(own).has_value());
        CHECK(own.at(static_cast<std::size_t>(Call::InstanceCreateDevice)).count == 1);
    }
    {
        const TimestampDevice untimed{recordingInto(backend, messages)};
        std::array<CallTiming, kCallCount> timings{};
        CHECK(untimed.device->callTimings(timings).error().code == ErrorCode::RhiInvalidCall);
        CHECK(untimed.instance->callTimings(timings).error().code == ErrorCode::RhiInvalidCall);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device writes a timestamp at each stage a compute or copy queue's caps list", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        auto instance = createInstance(recordingInto(backend, messages), kValidation);
        INFO(failureOf(instance));
        REQUIRE(instance.has_value());
        const Caps& caps = (*instance)->adapters()[0].caps;
        if (caps.queues.compute == 0 || caps.queues.copy == 0) {
            SKIP("the adapter has no compute or no copy queue");
        }
        // Queue 0 is graphics, 1 compute and 2 copy, as DeviceDesc numbers them.
        constexpr std::uint32_t kGraphics = 0;
        constexpr std::uint32_t kCompute = 1;
        constexpr std::uint32_t kCopy = 2;
        auto opened = (*instance)->createDevice(0, kNoWindow, testDeviceWith([](DeviceDesc& desc) {
                                                    desc.compute = std::span{kTestQueues}.first(1);
                                                    desc.copy = std::span{kTestQueues}.first(1);
                                                }));
        INFO(failureOf(opened));
        REQUIRE(opened.has_value());
        Device& device = **opened;
        const auto [queue, writes,
                    stages] = GENERATE_COPY(std::tuple{kCompute, caps.timestamps.compute,
                                                       caps.timestamps.computeStages},
                                            std::tuple{kCopy, caps.timestamps.copy, caps.timestamps.copyStages});
        INFO("queue " << queue);
        if (!writes) {
            SKIP("this kind of queue writes no timestamps");
        }
        const std::vector<TimestampStage> listedStages = listed(stages);
        constexpr std::array kEveryStage{TimestampStage::End,      TimestampStage::Start,   TimestampStage::Vertex,
                                         TimestampStage::Fragment, TimestampStage::Compute, TimestampStage::Copy,
                                         TimestampStage::Task,     TimestampStage::Mesh};
        constexpr auto kStages = static_cast<std::uint32_t>(kEveryStage.size());
        const auto pool = device.createTimestampPool({.queue = queue, .count = kStages});
        REQUIRE(pool.has_value());
        // Where this kind of queue resets no timestamps, the graphics queue
        // resets them first.
        const bool resets = queue == kCompute ? caps.timestamps.computeResets : caps.timestamps.copyResets;
        const auto resetting = device.createCommandPool({.queue = kGraphics, .listCount = kPoolLists});
        REQUIRE(resetting.has_value());
        const auto reset = device.beginCommands(*resetting);
        REQUIRE(reset.has_value());
        CHECK(device.resetTimestamps(*reset, *pool, 0, kStages).has_value());
        const auto sent = device.submit({.queue = kGraphics, .commandLists = std::span{&*reset, 1}, .waits = {}});
        INFO(failureOf(sent));
        REQUIRE(sent.has_value());
        REQUIRE(device.wait(*sent, kTestWait).has_value());
        const auto commands = device.createCommandPool({.queue = queue, .listCount = kPoolLists});
        REQUIRE(commands.has_value());
        const auto list = device.beginCommands(*commands);
        REQUIRE(list.has_value());
        if (resets) {
            CHECK(device.resetTimestamps(*list, *pool, 0, 0).has_value());
        }
        for (std::uint32_t index = 0; index < kStages; ++index) {
            const TimestampStage stage = kEveryStage.at(index);
            INFO("stage " << static_cast<int>(stage));
            if (std::ranges::contains(listedStages, stage)) {
                CHECK(device.writeTimestamp(*list, *pool, index, stage, TimestampGranularity::Relaxed).has_value());
            }
        }
        CHECK(device.destroyCommandList(*list).has_value());
        CHECK(device.destroyCommandPool(*commands).has_value());
        CHECK(device.destroyCommandPool(*resetting).has_value());
        CHECK(device.destroyTimestampPool(*pool).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}
