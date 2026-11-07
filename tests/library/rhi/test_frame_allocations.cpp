#include "allocation/allocation_counter.h"
#include "bfr/core/error.h"
#include "bfr/core/window.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/instance.h"
#include "bfr/rhi/rhi_config.h"
#include "bfr/rhi/swapchain.h"
#include "bfr/rhi/texture.h"
#include "rhi_test_support.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_container_properties.hpp>

#include <array>
#include <cstddef>
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
constexpr int kFrames = 8;
// Lists each frame records, more than the 16 a submit once held in place
// before it went to the heap; the first two render to the swapchain.
constexpr std::size_t kListsPerFrame = 20;
constexpr std::size_t kRenderingLists = 2;
// The frame that destroys a pool the GPU may still be running, so a later
// submit, wait or poll collects it.
constexpr int kRetireFrame = 2;

const bfr::core::window::NativeWindow kHeadless = bfr::core::window::Headless{};

// What the first list after the rendering ones copies each frame.
constexpr std::uint64_t kCopyBytes = 256;
constexpr BufferCopy kCopy{.srcOffset = 0, .dstOffset = 0, .size = kCopyBytes};
// The same bytes copied into one row of a texture: 64 rgba8 texels.
constexpr std::uint32_t kRowTexels = 64;
constexpr BufferTextureCopy kTextureCopy{.bufferOffset = 0,
                                         .bytesPerRow = kCopyBytes,
                                         .rowsPerImage = 1,
                                         .mipLevel = 0,
                                         .layer = 0,
                                         .aspect = TextureAspect::Color,
                                         .x = 0,
                                         .y = 0,
                                         .z = 0,
                                         .width = kRowTexels,
                                         .height = 1,
                                         .depth = 1};

// The swapchain's image, cleared.
constexpr RenderingDesc kClear{.colorAttachmentCount = 1,
                               .colorAttachments = {{{.texture = {},
                                                      .loadOp = LoadOp::Clear,
                                                      .storeOp = StoreOp::Store,
                                                      .clearColor = {0, 0, 0, 1}}}},
                               .depthStencil = {}};

// What one frame's calls failed with, or nothing; it holds no string, so
// checking it allocates nothing.
struct FrameFailure {
    const char* call = nullptr;
    bfr::core::ErrorCode code{};
};

} // namespace

// Validation stays off: its layer allocates through the operator new this
// program counts.
TEST_CASE("Device allocates and frees nothing on the heap frame after frame", "[rhi][agnostic]") {
    if (!bfr::test::AllocationCounter::counts()) {
        SKIP("a sanitizer replaces operator new, so allocations are not counted");
    }
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        auto created = createInstance(recordingInto(backend, messages), RhiConfig{});
        INFO(failureOf(created));
        REQUIRE(created.has_value());
        const std::unique_ptr<Instance> instance = std::move(*created);
        const std::unique_ptr<Device> device = presentingDevice(*instance, kHeadless);
        const auto swapchain = device->createSwapchain(kHeadless, swapchainDesc(*instance));
        INFO(failureOf(swapchain));
        REQUIRE(swapchain.has_value());
        REQUIRE(device->resizeSwapchain(*swapchain, kWidth, kHeight).has_value());
        const auto pool = device->createCommandPool({.queue = 0, .listCount = kListsPerFrame});
        INFO(failureOf(pool));
        REQUIRE(pool.has_value());
        const auto retiring = device->createCommandPool({.queue = 0, .listCount = 1});
        INFO(failureOf(retiring));
        REQUIRE(retiring.has_value());
        const auto source = device->createBuffer(
            {.size = kCopyBytes, .memory = BufferMemory::Upload, .usage = {.copySrc = true}});
        REQUIRE(source.has_value());
        const auto target = device->createBuffer(
            {.size = kCopyBytes, .memory = BufferMemory::Device, .usage = {.copyDst = true}});
        REQUIRE(target.has_value());
        const auto texture = device->createTexture({.dimension = TextureDimension::D2,
                                                    .format = Format::Rgba8Unorm,
                                                    .width = kRowTexels,
                                                    .height = 1,
                                                    .depth = 1,
                                                    .layers = 1,
                                                    .mipLevels = 1,
                                                    .samples = 1,
                                                    .usage = {.copyDst = true}});
        INFO(failureOf(texture));
        REQUIRE(texture.has_value());

        FrameFailure failed;
        std::size_t allocations = 0;
        std::size_t frees = 0;
        {
            const bfr::test::AllocationCounter counter;
            Token last;
            const auto check = [&failed](const char* call, const auto& result) {
                if (!result.has_value() && failed.call == nullptr) {
                    failed = {.call = call, .code = result.error().code};
                }
                return result.has_value();
            };
            for (int frame = 0; frame < kFrames && failed.call == nullptr; ++frame) {
                check("wait", device->wait(last, kTestWait));
                check("resetCommandPool", device->resetCommandPool(*pool));
                check("acquire", device->acquire(*swapchain, kTestWait));
                std::array<CommandListHandle, kListsPerFrame> lists{};
                for (std::size_t index = 0; index < lists.size(); ++index) {
                    const auto commands = device->beginCommands(*pool);
                    if (!check("beginCommands", commands)) {
                        break;
                    }
                    lists.at(index) = *commands;
                    if (index < kRenderingLists) {
                        check("beginRendering", device->beginRendering(*commands, *swapchain, kClear));
                        check("endRendering", device->endRendering(*commands));
                    }
                    if (index == kRenderingLists) {
                        check("copyBuffer", device->copyBuffer(*commands, *source, *target, std::span{&kCopy, 1}));
                        check("barrier", device->barrier(*commands));
                        check("textureBarrier", device->textureBarrier(*commands, *texture, TextureLayout::Undefined,
                                                                       TextureLayout::CopyDst));
                        check("copyBufferToTexture",
                              device->copyBufferToTexture(*commands, *source, *texture, std::span{&kTextureCopy, 1}));
                    }
                }
                const std::array<Token, 2> waits{last, last};
                const auto submitted = device->submit(
                    {.queue = 0, .commandLists = lists, .waits = std::span{waits}.first(frame == 0 ? 0 : 2)});
                if (check("submit", submitted)) {
                    last = *submitted;
                }
                check("present", device->present(*swapchain));
                if (frame == kRetireFrame) {
                    const auto commands = device->beginCommands(*retiring);
                    if (check("beginCommands", commands)) {
                        const auto sent = device->submit(
                            {.queue = 0, .commandLists = std::span{&*commands, 1}, .waits = {}});
                        check("submit", sent);
                    }
                    check("destroyCommandPool", device->destroyCommandPool(*retiring));
                }
                check("poll", device->poll(last));
            }
            check("wait", device->wait(last, kTestWait));
            allocations = counter.allocations();
            frees = counter.frees();
        }
        INFO("first failure: " << (failed.call == nullptr ? "none" : failed.call) << " "
                               << bfr::core::name(failed.code));
        CHECK(failed.call == nullptr);
        CHECK(allocations == 0);
        CHECK(frees == 0);
        CHECK(device->destroyCommandPool(*pool).has_value());
        CHECK(device->destroySwapchain(*swapchain).has_value());
        CHECK(device->destroyBuffer(*source).has_value());
        CHECK(device->destroyBuffer(*target).has_value());
        CHECK(device->destroyTexture(*texture).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

// acquire on a swapchain with no images, as on a minimised window, fails every
// frame; the failure, its message and the backend's log of it allocate nothing.
TEST_CASE("Device allocates nothing when a frame-path call fails frame after frame", "[rhi][agnostic]") {
    if (!bfr::test::AllocationCounter::counts()) {
        SKIP("a sanitizer replaces operator new, so allocations are not counted");
    }
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        auto created = createInstance(recordingInto(backend, messages), RhiConfig{});
        INFO(failureOf(created));
        REQUIRE(created.has_value());
        const std::unique_ptr<Instance> instance = std::move(*created);
        const std::unique_ptr<Device> device = presentingDevice(*instance, kHeadless);
        const auto swapchain = device->createSwapchain(kHeadless, swapchainDesc(*instance));
        INFO(failureOf(swapchain));
        REQUIRE(swapchain.has_value());

        int refused = 0;
        std::size_t allocations = 0;
        std::size_t frees = 0;
        {
            const bfr::test::AllocationCounter counter;
            for (int frame = 0; frame < kFrames; ++frame) {
                const auto acquired = device->acquire(*swapchain, kTestWait);
                refused += !acquired.has_value() && acquired.error().code == bfr::core::ErrorCode::RhiOutOfDate ? 1 : 0;
            }
            allocations = counter.allocations();
            frees = counter.frees();
        }
        CHECK(refused == kFrames);
        CHECK(allocations == 0);
        CHECK(frees == 0);
        CHECK(device->destroySwapchain(*swapchain).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}
