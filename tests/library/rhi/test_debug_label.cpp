// What every backend does with debug labels, through the interface alone.
// Each case runs on every backend the build has, and labels only what the
// adapter's caps list.

#include "bfr.hpp"
#include "bfr/bfr.h"
#include "bfr_test_support.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_container_properties.hpp>

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace bfr;
using namespace bfr::test;

namespace {

constexpr std::uint32_t kLongest = 8;
constexpr std::string_view kFull = "12345678";
constexpr std::string_view kTooLong = "123456789";
constexpr std::uint32_t kNoQueue = 99;
constexpr std::uint32_t kSide = 4;
constexpr std::uint32_t kVolumeSide = 8;
constexpr std::uint32_t kVolumeLevels = 4;

// An Instance and a device on its first adapter that takes labels of up to
// longest bytes.
struct LabelDevice {
    LabelDevice(const InstanceDesc& desc, std::uint32_t longest) {
        auto created = createInstance(desc, kValidation);
        INFO(failureOf(created));
        REQUIRE(created.has_value());
        instance = std::move(*created);
        auto made = instance.createDevice(
            testDeviceWith(instance, kFirstAdapter, kNoWindow,
                           [longest](DeviceDesc& device) { device.maxDebugLabelLength = longest; }));
        INFO(failureOf(made));
        REQUIRE(made.has_value());
        device = std::move(*made);
    }

    // The first adapter's features; the test fails when the caps can't be read.
    [[nodiscard]] CapsFeatures features() const { return capsOf(instance, kFirstAdapter).features; }

    Instance instance;
    Device device;
};

} // namespace

TEST_CASE("Device labels a list's work and a queue's where the caps say so, and the backend reports nothing wrong",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const LabelDevice made{recordingInto(backend, messages), kLongest};
        const Device& device = made.device;
        auto commands = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(commands.has_value());
        auto list = commands->beginCommands();
        REQUIRE(list.has_value());

        if (made.features().debugLabels) {
            CHECK(device.beginDebugLabel(*list, {.name = "frame", .color = {1.0F, 0.0F, 0.0F, 1.0F}}).has_value());
            CHECK(device.beginDebugLabel(*list, {.name = kFull}).has_value());
            CHECK(device.insertDebugLabel(*list, {.name = ""}).has_value());
            CHECK(device.endDebugLabel(*list).has_value());
            CHECK(device.endDebugLabel(*list).has_value());
        }
        else {
            CHECK(device.beginDebugLabel(*list, {.name = "frame"}).error().code == Result::ErrorUnsupported);
            CHECK(device.endDebugLabel(*list).error().code == Result::ErrorUnsupported);
            CHECK(device.insertDebugLabel(*list, {.name = "frame"}).error().code == Result::ErrorUnsupported);
        }
        if (made.features().queueDebugLabels) {
            CHECK(device.beginQueueDebugLabel(0, {.name = "submit", .color = {0.0F, 1.0F, 0.0F, 1.0F}}).has_value());
        }
        else {
            CHECK(device.beginQueueDebugLabel(0, {.name = "submit"}).error().code == Result::ErrorUnsupported);
            CHECK(device.endQueueDebugLabel(0).error().code == Result::ErrorUnsupported);
            CHECK(device.insertQueueDebugLabel(0, {.name = "submit"}).error().code == Result::ErrorUnsupported);
        }
        const auto sent = device.submit({.queue = 0, .commandLists = std::span{&*list, 1}, .waits = {}});
        INFO(failureOf(sent));
        REQUIRE(sent.has_value());
        if (made.features().queueDebugLabels) {
            CHECK(device.insertQueueDebugLabel(0, {.name = kFull}).has_value());
            CHECK(device.endQueueDebugLabel(0).has_value());
        }
        REQUIRE(device.wait(*sent, kTestWait).has_value());
        const bfr_command_pool stale = commands->get();
        *commands = {};
        CHECK(resetResult(device, stale) == Result::ErrorInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device refuses a label longer than its room or holding a '\\0', a stale list and a queue it lacks",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    const LabelDevice made{describe(backend), kLongest};
    if (!made.features().debugLabels || !made.features().queueDebugLabels) {
        SKIP("the adapter takes no debug labels");
    }
    const Device& device = made.device;
    auto commands = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
    REQUIRE(commands.has_value());
    auto list = commands->beginCommands();
    REQUIRE(list.has_value());
    constexpr std::string_view kHoldsEnd{"a\0b", 3};

    CHECK(device.beginDebugLabel(*list, {.name = kTooLong}).error().code == Result::ErrorLimitReached);
    CHECK(device.insertDebugLabel(*list, {.name = kTooLong}).error().code == Result::ErrorLimitReached);
    CHECK(device.beginDebugLabel(*list, {.name = kHoldsEnd}).error().code == Result::ErrorInvalidArgument);
    CHECK(device.insertDebugLabel(*list, {.name = kHoldsEnd}).error().code == Result::ErrorInvalidArgument);
    CHECK(device.beginQueueDebugLabel(0, {.name = kTooLong}).error().code == Result::ErrorLimitReached);
    CHECK(device.insertQueueDebugLabel(0, {.name = kHoldsEnd}).error().code == Result::ErrorInvalidArgument);
    CHECK(device.beginQueueDebugLabel(kNoQueue, {.name = "submit"}).error().code == Result::ErrorInvalidArgument);
    CHECK(device.endQueueDebugLabel(kNoQueue).error().code == Result::ErrorInvalidArgument);
    CHECK(device.insertQueueDebugLabel(kNoQueue, {.name = "submit"}).error().code == Result::ErrorInvalidArgument);
    CHECK(device.beginDebugLabel({}, {.name = "frame"}).error().code == Result::ErrorInvalidHandle);
    CHECK(device.endDebugLabel({}).error().code == Result::ErrorInvalidHandle);
    CHECK(device.insertDebugLabel({}, {.name = "frame"}).error().code == Result::ErrorInvalidHandle);
    const bfr_command_pool stale = commands->get();
    *commands = {};
    CHECK(resetResult(device, stale) == Result::ErrorInvalidHandle);
}

TEST_CASE("Device with no room for labels takes only empty names", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    const LabelDevice made{describe(backend), 0};
    if (!made.features().debugLabels) {
        SKIP("the adapter takes no debug labels");
    }
    const Device& device = made.device;
    auto commands = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
    REQUIRE(commands.has_value());
    auto list = commands->beginCommands();
    REQUIRE(list.has_value());

    CHECK(device.insertDebugLabel(*list, {.name = ""}).has_value());
    CHECK(device.insertDebugLabel(*list, {.name = "a"}).error().code == Result::ErrorLimitReached);
    const bfr_command_pool stale = commands->get();
    *commands = {};
    CHECK(resetResult(device, stale) == Result::ErrorInvalidHandle);
}

TEST_CASE("Device names the objects it makes where the caps say so, and refuses a name it can't take",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const LabelDevice made{recordingInto(backend, messages), 0};
        const Device& device = made.device;
        constexpr std::uint64_t kBytes = 256;
        constexpr std::string_view kHoldsEnd{"a\0b", 3};
        const BufferDesc named{.size = kBytes, .usage = {.copyDst = true}, .name = "vertices"};
        const TextureDesc texture{.dimension = TextureDimension::D2,
                                  .format = Format::Rgba8Unorm,
                                  .width = kSide,
                                  .height = kSide,
                                  .depth = 1,
                                  .layers = 1,
                                  .mipLevels = 1,
                                  .samples = 1,
                                  .usage = {.sampled = true, .colorAttachment = true},
                                  .name = "albedo"};
        if (!made.features().objectNames) {
            CHECK(device.createBuffer(named).error().code == Result::ErrorUnsupported);
            CHECK(device.createTexture(texture).error().code == Result::ErrorUnsupported);
            CHECK(device.createTimestampPool({.queue = 0, .count = 1, .name = "frame"}).error().code ==
                  Result::ErrorUnsupported);
            auto unnamed = device.createBuffer({.size = kBytes, .usage = {.copyDst = true}});
            REQUIRE(unnamed.has_value());
            const bfr_buffer stale = unnamed->get();
            *unnamed = {};
            CHECK(slotsResult(device, stale) == Result::ErrorInvalidHandle);
        }
        else {
            auto buffer = device.createBuffer(named);
            INFO(failureOf(buffer));
            REQUIRE(buffer.has_value());
            auto image = device.createTexture(texture);
            INFO(failureOf(image));
            REQUIRE(image.has_value());
            auto timestamps = device.createTimestampPool({.queue = 0, .count = 1, .name = "frame"});
            INFO(failureOf(timestamps));
            REQUIRE(timestamps.has_value());
            CHECK(device.createBuffer({.size = kBytes, .usage = {.copyDst = true}, .name = kHoldsEnd}).error().code ==
                  Result::ErrorInvalidArgument);
            // A 3D color target with levels, whose deeper levels have fewer
            // slices to draw to.
            if (made.features().volumeColorTargets) {
                auto volume = device.createTexture({.dimension = TextureDimension::D3,
                                                    .format = Format::Rgba8Unorm,
                                                    .width = kVolumeSide,
                                                    .height = kVolumeSide,
                                                    .depth = kVolumeSide,
                                                    .layers = 1,
                                                    .mipLevels = kVolumeLevels,
                                                    .samples = 1,
                                                    .usage = {.colorAttachment = true},
                                                    .name = "voxels"});
                INFO(failureOf(volume));
                REQUIRE(volume.has_value());
                const bfr_texture staleVolume = volume->get();
                *volume = {};
                CHECK(slotsResult(device, staleVolume) == Result::ErrorInvalidHandle);
            }
            const bfr_timestamp_pool goneTimestamps = destroyed(*timestamps);
            CHECK(resetTimestampsResult(device, 0, goneTimestamps) == Result::ErrorInvalidHandle);
            const bfr_texture staleImage = image->get();
            *image = {};
            CHECK(slotsResult(device, staleImage) == Result::ErrorInvalidHandle);
            const bfr_buffer staleBuffer = buffer->get();
            *buffer = {};
            CHECK(slotsResult(device, staleBuffer) == Result::ErrorInvalidHandle);
        }
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}
