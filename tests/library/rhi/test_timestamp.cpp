// What every backend does with GPU timestamps and call timings, through the
// interface alone. Each case runs on every backend the build has, and writes
// only what the adapter's caps list.

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
#include <cstring>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace bfr;
using namespace bfr::test;

namespace {

constexpr std::uint32_t kSide = 16;
constexpr std::uint64_t kReadbackBytes = 4096;

// What made's adapter reports of timestamps; the test fails when it can't say.
CapsTimestamps timestampCaps(const FirstAdapterDevice& made) {
    return capsOf(made.instance, kFirstAdapter).timestamps;
}

// The family of made's device's queue 0.
QueueFamily queueFamily(const FirstAdapterDevice& made) {
    return testQueueFamily(made.instance, kFirstAdapter);
}

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

// Asks desc for the features the stages caps list take: task and mesh
// shaders, ray tracing pipelines and acceleration structure builds, each where
// the adapter has it.
void askStageFeatures(DeviceDesc& desc, const Caps& caps) {
    desc.features.taskShader = caps.features.taskShader;
    desc.features.meshShader = caps.features.meshShader;
    desc.features.rayTracingPipeline = caps.features.rayTracingPipeline;
    desc.features.accelerationStructure = caps.features.accelerationStructure;
    desc.features.bufferAddress = caps.features.accelerationStructure;
}

} // namespace

TEST_CASE("Device writes a timestamp at each stage the caps list, in rendering and out, and resolves them as ticks",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages), askStageFeatures};
        const Device& device = made.device;
        const CapsTimestamps caps = timestampCaps(made);
        const QueueFamily family = queueFamily(made);
        if (!family.timestamps) {
            SKIP("the device's queue writes no timestamps");
        }
        REQUIRE(caps.entryBytes >= sizeof(std::uint64_t));
        const std::vector<TimestampStage> outside = listed(family.timestampStages);
        const std::vector<TimestampStage> inside = listed(caps.insideRendering);
        // The outside stages, then the inside ones, then one more outside
        // after rendering, which no earlier timestamp comes after.
        const auto count = static_cast<std::uint32_t>(outside.size() + inside.size() + 1);
        auto pool = device.createTimestampPool({.queue = 0, .count = count});
        INFO(failureOf(pool));
        REQUIRE(pool.has_value());
        auto readback = device.createBuffer(
            {.size = kReadbackBytes, .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
        REQUIRE(readback.has_value());
        auto target = device.createTexture({.dimension = TextureDimension::D2,
                                            .format = Format::Rgba8Unorm,
                                            .width = kSide,
                                            .height = kSide,
                                            .depth = 1,
                                            .layers = 1,
                                            .mipLevels = 1,
                                            .samples = 1,
                                            .usage = {.colorAttachment = true}});
        REQUIRE(target.has_value());
        auto commands = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(commands.has_value());
        auto list = commands->beginCommands();
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

        const auto mapped = readback->map();
        REQUIRE(mapped.has_value());
        const std::span bytes{static_cast<const std::byte*>(*mapped), kReadbackBytes};
        const std::vector<std::uint64_t> ticks = ticksIn(bytes, caps.entryBytes, count);
        // The last waits for all the work before it, the first for none.
        CHECK(ticks[last] >= ticks[0]);
        CHECK(ticks[last] != 0);

        const bfr_command_pool staleCommands = commands->get();
        *commands = {};
        CHECK(resetResult(device, staleCommands) == Result::ErrorInvalidHandle);
        const bfr_texture staleTarget = target->get();
        *target = {};
        CHECK(slotsResult(device, staleTarget) == Result::ErrorInvalidHandle);
        const bfr_buffer staleReadback = readback->get();
        *readback = {};
        CHECK(slotsResult(device, staleReadback) == Result::ErrorInvalidHandle);
        const bfr_timestamp_pool stale = destroyed(*pool);
        CHECK(resetTimestampsResult(device, 0, stale) == Result::ErrorInvalidHandle);
        // d1: stale destroy reported as an Error message (track B)
        bfr_destroy_timestamp_pool(device.get(), stale);
        // d1: null destroy: no message (track B)
        bfr_destroy_timestamp_pool(device.get(), {});
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device gives a queue's timestamp rate in the backend's one unit, and refuses a queue it lacks",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        if (!queueFamily(made).timestamps) {
            SKIP("the device's queue writes no timestamps");
        }
        const auto rate = made.device.timestampRate(0);
        INFO(failureOf(rate));
        REQUIRE(rate.has_value());
        CHECK((rate->periodNs > 0) != (rate->frequency != 0));
        CHECK(made.device.timestampRate(1).error().code == Result::ErrorInvalidArgument);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device reads each host clock the caps list beside a GPU timestamp, and refuses the others",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages), [](DeviceDesc& desc, const Caps& caps) {
                                          const HostClocks& clocks = caps.timestamps.hostClocks;
                                          desc.features.timestampCalibration = clocks.monotonic ||
                                                                               clocks.monotonicRaw ||
                                                                               clocks.performanceCounter;
                                      }};
        const HostClocks clocks = timestampCaps(made).hostClocks;
        for (const auto& [has, clock] : {std::pair{clocks.monotonic, HostClock::Monotonic},
                                         std::pair{clocks.monotonicRaw, HostClock::MonotonicRaw},
                                         std::pair{clocks.performanceCounter, HostClock::PerformanceCounter}}) {
            INFO("host clock " << static_cast<int>(clock));
            const auto taken = made.device.calibrateTimestamps(0, clock);
            if (!has) {
                CHECK(taken.error().code == Result::ErrorUnsupported);
                continue;
            }
            INFO(failureOf(taken));
            REQUIRE(taken.has_value());
            CHECK(taken->gpu != 0);
            CHECK(taken->host != 0);
        }
        CHECK(made.device.calibrateTimestamps(0, static_cast<HostClock>(99)).error().code ==
              Result::ErrorInvalidArgument);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device refuses a timestamp stage or granularity BFR doesn't have", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        auto pool = device.createTimestampPool({.queue = 0, .count = 1});
        REQUIRE(pool.has_value());
        auto commands = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(commands.has_value());
        auto list = commands->beginCommands();
        REQUIRE(list.has_value());

        constexpr auto kPast = 99;
        CHECK(device.writeTimestamp(*list, *pool, 0, static_cast<TimestampStage>(kPast), TimestampGranularity::Relaxed)
                  .error()
                  .code == Result::ErrorInvalidArgument);
        CHECK(device.writeTimestamp(*list, *pool, 0, TimestampStage::End, static_cast<TimestampGranularity>(kPast))
                  .error()
                  .code == Result::ErrorInvalidArgument);
        CHECK(device.createTimestampPool({.queue = 1, .count = 1}).error().code == Result::ErrorInvalidArgument);
        device.destroyCommandList(*list);
        CHECK(device.barrier(*list).error().code == Result::ErrorInvalidHandle);
        const bfr_command_pool staleCommands = commands->get();
        *commands = {};
        CHECK(resetResult(device, staleCommands) == Result::ErrorInvalidHandle);
        const bfr_timestamp_pool gone = destroyed(*pool);
        CHECK(resetTimestampsResult(device, 0, gone) == Result::ErrorInvalidHandle);
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
        const FirstAdapterDevice made{desc};
        const Device& device = made.device;
        for (std::uint64_t round = 0; round < kBuffers; ++round) {
            auto buffer = device.createBuffer(
                {.size = kReadbackBytes, .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
            REQUIRE(buffer.has_value());
            const bfr_buffer stale = buffer->get();
            *buffer = {};
            CHECK(slotsResult(device, stale) == Result::ErrorInvalidHandle);
        }

        const auto timings = device.callTimings();
        REQUIRE(timings.has_value());
        const CallTiming& creates = timings->at(static_cast<std::size_t>(Call::CreateBuffer));
        CHECK(creates.count == kBuffers);
        CHECK(creates.totalNs >= creates.maxNs);
        CHECK(creates.maxNs != 0);
        CHECK(timings->at(static_cast<std::size_t>(Call::DestroyBuffer)).count == kBuffers);
        CHECK(timings->at(static_cast<std::size_t>(Call::Draw)).count == 0);

        const auto own = made.instance.instanceCallTimings();
        REQUIRE(own.has_value());
        CHECK(own->at(static_cast<std::size_t>(Call::InstanceCreateDevice)).count == 1);
    }
    {
        const FirstAdapterDevice untimed{recordingInto(backend, messages)};
        CHECK(untimed.device.callTimings().error().code == Result::ErrorInvalidArgument);
        CHECK(untimed.instance.instanceCallTimings().error().code == Result::ErrorInvalidArgument);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device writes a timestamp at each stage the family of a queue without graphics lists", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        auto instance = createInstance(recordingInto(backend, messages), kValidation);
        INFO(failureOf(instance));
        REQUIRE(instance.has_value());
        const std::vector<QueueFamily> families = queueFamiliesOf(*instance, kFirstAdapter);
        const auto computeFamily = familyWhere(*instance, kFirstAdapter, [](const QueueFamily& family) {
            return family.compute && !family.graphics;
        });
        const auto copyFamily = familyWhere(*instance, kFirstAdapter, [](const QueueFamily& family) {
            return family.copy && !family.graphics && !family.compute;
        });
        if (!computeFamily || !copyFamily) {
            SKIP("the adapter has no family that runs compute but not graphics, or none that runs copies alone");
        }
        // Queue 0 runs graphics, 1 compute and 2 copies, as DeviceDesc numbers them.
        constexpr std::uint32_t kGraphics = 0;
        constexpr std::uint32_t kCompute = 1;
        constexpr std::uint32_t kCopy = 2;
        const std::array queues{testQueues(testFamily(*instance, kFirstAdapter, kNoWindow)).front(),
                                testQueues(*computeFamily).front(), testQueues(*copyFamily).front()};
        const Caps caps = capsOf(*instance, kFirstAdapter);
        auto opened = instance->createDevice(
            testDeviceWith(*instance, kFirstAdapter, kNoWindow, [&queues, &caps](DeviceDesc& desc) {
                desc.queues = queues;
                askStageFeatures(desc, caps);
            }));
        INFO(failureOf(opened));
        REQUIRE(opened.has_value());
        const Device& device = *opened;
        const auto [queue, family] = GENERATE_COPY(std::pair{kCompute, families.at(*computeFamily)},
                                                   std::pair{kCopy, families.at(*copyFamily)});
        INFO("queue " << queue);
        if (!family.timestamps) {
            SKIP("this queue's family writes no timestamps");
        }
        const TimestampStages& stages = family.timestampStages;
        const std::vector<TimestampStage> listedStages = listed(stages);
        constexpr std::array kEveryStage{TimestampStage::End,      TimestampStage::Start,   TimestampStage::Vertex,
                                         TimestampStage::Fragment, TimestampStage::Compute, TimestampStage::Copy,
                                         TimestampStage::Task,     TimestampStage::Mesh};
        constexpr auto kStages = static_cast<std::uint32_t>(kEveryStage.size());
        auto pool = device.createTimestampPool({.queue = queue, .count = kStages});
        REQUIRE(pool.has_value());
        // Where this queue's family resets no timestamps, the graphics queue
        // resets them first.
        const bool resets = family.timestampResets;
        auto resetting = device.createCommandPool({.queue = kGraphics, .listCount = kPoolLists});
        REQUIRE(resetting.has_value());
        const auto reset = resetting->beginCommands();
        REQUIRE(reset.has_value());
        CHECK(device.resetTimestamps(*reset, *pool, 0, kStages).has_value());
        const auto sent = device.submit({.queue = kGraphics, .commandLists = std::span{&*reset, 1}, .waits = {}});
        INFO(failureOf(sent));
        REQUIRE(sent.has_value());
        REQUIRE(device.wait(*sent, kTestWait).has_value());
        auto commands = device.createCommandPool({.queue = queue, .listCount = kPoolLists});
        REQUIRE(commands.has_value());
        const auto list = commands->beginCommands();
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
        device.destroyCommandList(*list);
        CHECK(device.barrier(*list).error().code == Result::ErrorInvalidHandle);
        const bfr_command_pool staleCommands = commands->get();
        *commands = {};
        CHECK(resetResult(device, staleCommands) == Result::ErrorInvalidHandle);
        const bfr_command_pool staleResetting = resetting->get();
        *resetting = {};
        CHECK(resetResult(device, staleResetting) == Result::ErrorInvalidHandle);
        const bfr_timestamp_pool gone = destroyed(*pool);
        CHECK(resetTimestampsResult(device, kGraphics, gone) == Result::ErrorInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}
