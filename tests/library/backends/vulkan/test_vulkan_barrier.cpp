// What the Vulkan backend refuses of a barrier: an access none of its stages
// pairs with, a stage of a feature the device lacks, or one the list's queue
// family doesn't run, each a rule of vkCmdPipelineBarrier2's; and that it
// records more barriers than its stack chunk holds, checking all of them
// first.

#include "bfr.hpp"
#include "bfr/bfr.h"
#include "rhi/barrier_check.h"
#include "rhi/bfr_test_support.h"
#include "vulkan_test_support.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_container_properties.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace bfr;
using namespace bfr::test;
using namespace bfr::test::vulkan;

namespace {

// More layers, one barrier each, than two of the backend's chunks hold, and
// more global barriers than one does.
constexpr std::uint32_t kChunkedLayers = (2 * rhi::kBarrierChunk) + 1;
constexpr std::size_t kChunkedGlobals = rhi::kBarrierChunk + 1;
constexpr std::uint32_t kSide = 4;

// What device says of one global barrier from before to after, recorded on
// queue.
Result globalResult(const Device& device, std::uint32_t queue, const BarrierScope& before, const BarrierScope& after) {
    return inList(device, queue,
                  [&](const Device& on, CommandList list) { return globalBarrier(on, list, before, after); });
}

} // namespace

TEST_CASE("Vulkan device refuses a barrier whose access none of its stages pairs with, or whose stage or access "
          "needs a feature the device lacks",
          "[vulkan][gpu]") {
    std::vector<std::string> messages;
    {
        auto created = createInstance(recordingInto(messages), kValidation);
        INFO(failureOf(created));
        REQUIRE(created.has_value());
        const Instance instance = std::move(*created);
        auto made = instance.createDevice(testDeviceOn(instance, kFirstAdapter, kNoWindow));
        INFO(failureOf(made));
        REQUIRE(made.has_value());
        const Device& device = *made;
        const auto caps = device.deviceCaps();
        REQUIRE(caps.has_value());

        // VUID-VkMemoryBarrier2-srcAccessMask-03905 and -dstAccessMask-03914.
        CHECK(globalResult(device, 0, {.stages = {.copy = true}, .accesses = {.sampledRead = true}}, kCopyReads) ==
              Result::ErrorInvalidArgument);
        CHECK(
            globalResult(device, 0, kCopyWrites, {.stages = {.computeShader = true}, .accesses = {.copyRead = true}}) ==
            Result::ErrorInvalidArgument);
        // An access with no stage at all, which no rule pairs.
        CHECK(globalResult(device, 0, {.accesses = {.copyWrite = true}}, kCopyReads) == Result::ErrorInvalidArgument);
        // The test device asks for no mesh shaders or acceleration structures
        // (VUID-VkMemoryBarrier2-srcStageMask-03934, -srcAccessMask-parameter).
        REQUIRE_FALSE(caps->features.meshShader);
        REQUIRE_FALSE(caps->features.accelerationStructure);
        CHECK(globalResult(device, 0, {.stages = {.meshShader = true}}, kCopyReads) == Result::ErrorUnsupported);
        CHECK(globalResult(device, 0, {.stages = {.all = true}, .accesses = {.accelerationStructureRead = true}},
                           kCopyReads) == Result::ErrorUnsupported);
        // Each pairing the rules allow is taken.
        CHECK(globalResult(device, 0, kComputeWrites, kCopyReads) == Result::Success);
        CHECK(globalResult(device, 0, {.stages = {.all = true}, .accesses = {.storageWrite = true}},
                           {.stages = {.drawIndirect = true}, .accesses = {.indirectRead = true}}) == Result::Success);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Vulkan device refuses a barrier stage the list's queue family doesn't run", "[vulkan][gpu]") {
    std::vector<std::string> messages;
    {
        auto created = createInstance(recordingInto(messages), kValidation);
        INFO(failureOf(created));
        REQUIRE(created.has_value());
        const Instance instance = std::move(*created);
        const auto computeOnly = familyWhere(instance, kFirstAdapter, [](const QueueFamily& family) {
            return family.compute && !family.graphics;
        });
        if (!computeOnly.has_value()) {
            SKIP("The adapter has no family that runs compute but not graphics");
        }
        auto made = instance.createDevice(testDeviceWith(instance, kFirstAdapter, kNoWindow, [&](DeviceDesc& desc) {
            desc.queues = testQueues(*computeOnly);
        }));
        INFO(failureOf(made));
        REQUIRE(made.has_value());
        const Device& device = *made;

        // VUID-vkCmdPipelineBarrier2-srcStageMask-09673 and -dstStageMask-09674.
        CHECK(globalResult(device, 0, {.stages = {.fragmentShader = true}}, kCopyReads) == Result::ErrorUnsupported);
        CHECK(globalResult(device, 0, kComputeWrites, {.stages = {.colorOutput = true}}) == Result::ErrorUnsupported);
        CHECK(globalResult(device, 0, kComputeWrites, kCopyReads) == Result::Success);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Vulkan device records more barriers than a chunk holds, and records none when the last is refused",
          "[vulkan][gpu]") {
    std::vector<std::string> messages;
    {
        auto created = createInstance(recordingInto(messages), kValidation);
        INFO(failureOf(created));
        REQUIRE(created.has_value());
        const Instance instance = std::move(*created);
        auto made = instance.createDevice(testDeviceOn(instance, kFirstAdapter, kNoWindow));
        INFO(failureOf(made));
        REQUIRE(made.has_value());
        const Device& device = *made;
        const Texture texture = bfr::test::created(device, {.dimension = TextureDimension::D2,
                                                            .format = Format::Rgba8Unorm,
                                                            .width = kSide,
                                                            .height = kSide,
                                                            .depth = 1,
                                                            .layers = kChunkedLayers,
                                                            .mipLevels = 1,
                                                            .samples = 1,
                                                            .usage = {.copySrc = true, .copyDst = true}});
        const std::vector<GlobalBarrier> globals(kChunkedGlobals, {.before = kCopyWrites, .after = kCopyReads});
        std::vector<TextureBarrier> layers;
        for (std::uint32_t layer = 0; layer < kChunkedLayers; ++layer) {
            layers.push_back(layoutBarrier(texture, TextureLayout::Undefined, TextureLayout::CopyDst,
                                           {.layer = layer, .layerCount = 1}));
        }
        auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        const auto list = pool->beginCommands();
        REQUIRE(list.has_value());

        const auto recorded = device.barrier(*list, {.globals = globals, .textures = layers});
        INFO(failureOf(recorded));
        CHECK(recorded.has_value());
        // Past the texture: the whole desc is refused before a chunk is
        // recorded, so the list still holds only the barriers above.
        layers.back().range = {.layer = kChunkedLayers, .layerCount = 1};
        CHECK(resultOf(device.barrier(*list, {.globals = globals, .textures = layers})) ==
              Result::ErrorInvalidArgument);
        const auto sent = device.submit({.queue = 0, .commandLists = std::span{&*list, 1}, .waits = {}});
        INFO(failureOf(sent));
        REQUIRE(sent.has_value());
        REQUIRE(device.wait(*sent, kTestWait).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}
