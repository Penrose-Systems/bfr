// What every backend does with debug labels, through the interface alone.
// Each case runs on every backend the build has, and labels only what the
// adapter's caps list.

#include "bfr/core/error.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/debug_label.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/instance.h"
#include "bfr/rhi/texture.h"
#include "bfr/rhi/timestamp.h"
#include "rhi_test_support.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_container_properties.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace bfr::rhi;
using namespace bfr::rhi::test;
using bfr::core::ErrorCode;

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
        auto made = instance->createDevice(0, kNoWindow, testDeviceWith([longest](DeviceDesc& device) {
                                               device.maxDebugLabelLength = longest;
                                           }));
        INFO(failureOf(made));
        REQUIRE(made.has_value());
        device = std::move(*made);
    }

    [[nodiscard]] const Features& features() const { return instance->adapters()[0].caps.features; }

    std::unique_ptr<Instance> instance;
    std::unique_ptr<Device> device;
};

} // namespace

TEST_CASE("Device labels a list's work and a queue's where the caps say so, and the backend reports nothing wrong",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const LabelDevice made{recordingInto(backend, messages), kLongest};
        Device& device = *made.device;
        const auto commands = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(commands.has_value());
        auto list = device.beginCommands(*commands);
        REQUIRE(list.has_value());

        if (made.features().debugLabels) {
            CHECK(device.beginDebugLabel(*list, {.name = "frame", .color = {1.0F, 0.0F, 0.0F, 1.0F}}).has_value());
            CHECK(device.beginDebugLabel(*list, {.name = kFull}).has_value());
            CHECK(device.insertDebugLabel(*list, {.name = ""}).has_value());
            CHECK(device.endDebugLabel(*list).has_value());
            CHECK(device.endDebugLabel(*list).has_value());
        }
        else {
            CHECK(device.beginDebugLabel(*list, {.name = "frame"}).error().code == ErrorCode::RhiUnsupported);
            CHECK(device.endDebugLabel(*list).error().code == ErrorCode::RhiUnsupported);
            CHECK(device.insertDebugLabel(*list, {.name = "frame"}).error().code == ErrorCode::RhiUnsupported);
        }
        if (made.features().queueDebugLabels) {
            CHECK(device.beginQueueDebugLabel(0, {.name = "submit", .color = {0.0F, 1.0F, 0.0F, 1.0F}}).has_value());
        }
        else {
            CHECK(device.beginQueueDebugLabel(0, {.name = "submit"}).error().code == ErrorCode::RhiUnsupported);
            CHECK(device.endQueueDebugLabel(0).error().code == ErrorCode::RhiUnsupported);
            CHECK(device.insertQueueDebugLabel(0, {.name = "submit"}).error().code == ErrorCode::RhiUnsupported);
        }
        const auto sent = device.submit({.queue = 0, .commandLists = std::span{&*list, 1}, .waits = {}});
        INFO(failureOf(sent));
        REQUIRE(sent.has_value());
        if (made.features().queueDebugLabels) {
            CHECK(device.insertQueueDebugLabel(0, {.name = kFull}).has_value());
            CHECK(device.endQueueDebugLabel(0).has_value());
        }
        REQUIRE(device.wait(*sent, kTestWait).has_value());
        CHECK(device.destroyCommandPool(*commands).has_value());
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
    Device& device = *made.device;
    const auto commands = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
    REQUIRE(commands.has_value());
    auto list = device.beginCommands(*commands);
    REQUIRE(list.has_value());
    constexpr std::string_view kHoldsEnd{"a\0b", 3};

    CHECK(device.beginDebugLabel(*list, {.name = kTooLong}).error().code == ErrorCode::RhiLimitReached);
    CHECK(device.insertDebugLabel(*list, {.name = kTooLong}).error().code == ErrorCode::RhiLimitReached);
    CHECK(device.beginDebugLabel(*list, {.name = kHoldsEnd}).error().code == ErrorCode::RhiInvalidCall);
    CHECK(device.insertDebugLabel(*list, {.name = kHoldsEnd}).error().code == ErrorCode::RhiInvalidCall);
    CHECK(device.beginQueueDebugLabel(0, {.name = kTooLong}).error().code == ErrorCode::RhiLimitReached);
    CHECK(device.insertQueueDebugLabel(0, {.name = kHoldsEnd}).error().code == ErrorCode::RhiInvalidCall);
    CHECK(device.beginQueueDebugLabel(kNoQueue, {.name = "submit"}).error().code == ErrorCode::RhiInvalidCall);
    CHECK(device.endQueueDebugLabel(kNoQueue).error().code == ErrorCode::RhiInvalidCall);
    CHECK(device.insertQueueDebugLabel(kNoQueue, {.name = "submit"}).error().code == ErrorCode::RhiInvalidCall);
    CHECK(device.beginDebugLabel({}, {.name = "frame"}).error().code == ErrorCode::RhiInvalidHandle);
    CHECK(device.endDebugLabel({}).error().code == ErrorCode::RhiInvalidHandle);
    CHECK(device.insertDebugLabel({}, {.name = "frame"}).error().code == ErrorCode::RhiInvalidHandle);
    CHECK(device.destroyCommandPool(*commands).has_value());
}

TEST_CASE("Device with no room for labels takes only empty names", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    const LabelDevice made{describe(backend), 0};
    if (!made.features().debugLabels) {
        SKIP("the adapter takes no debug labels");
    }
    Device& device = *made.device;
    const auto commands = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
    REQUIRE(commands.has_value());
    auto list = device.beginCommands(*commands);
    REQUIRE(list.has_value());

    CHECK(device.insertDebugLabel(*list, {.name = ""}).has_value());
    CHECK(device.insertDebugLabel(*list, {.name = "a"}).error().code == ErrorCode::RhiLimitReached);
    CHECK(device.destroyCommandPool(*commands).has_value());
}

TEST_CASE("Device names the objects it makes where the caps say so, and refuses a name it can't take",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const LabelDevice made{recordingInto(backend, messages), 0};
        Device& device = *made.device;
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
            CHECK(device.createBuffer(named).error().code == ErrorCode::RhiUnsupported);
            CHECK(device.createTexture(texture).error().code == ErrorCode::RhiUnsupported);
            CHECK(device.createTimestampPool({.queue = 0, .count = 1, .name = "frame"}).error().code ==
                  ErrorCode::RhiUnsupported);
            const auto unnamed = device.createBuffer({.size = kBytes, .usage = {.copyDst = true}});
            REQUIRE(unnamed.has_value());
            CHECK(device.destroyBuffer(*unnamed).has_value());
        }
        else {
            const auto buffer = device.createBuffer(named);
            INFO(failureOf(buffer));
            REQUIRE(buffer.has_value());
            const auto image = device.createTexture(texture);
            INFO(failureOf(image));
            REQUIRE(image.has_value());
            const auto timestamps = device.createTimestampPool({.queue = 0, .count = 1, .name = "frame"});
            INFO(failureOf(timestamps));
            REQUIRE(timestamps.has_value());
            CHECK(device.createBuffer({.size = kBytes, .usage = {.copyDst = true}, .name = kHoldsEnd}).error().code ==
                  ErrorCode::RhiInvalidCall);
            // A 3D color target with levels, whose deeper levels have fewer
            // slices to draw to.
            if (made.features().volumeColorTargets) {
                const auto volume = device.createTexture({.dimension = TextureDimension::D3,
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
                CHECK(device.destroyTexture(*volume).has_value());
            }
            CHECK(device.destroyTimestampPool(*timestamps).has_value());
            CHECK(device.destroyTexture(*image).has_value());
            CHECK(device.destroyBuffer(*buffer).has_value());
        }
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}
