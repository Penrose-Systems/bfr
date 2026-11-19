#include "allocation/allocation_counter.h"
#include "bfr.hpp"
#include "bfr/bfr.h"
#include "bfr_test_support.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_container_properties.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace bfr;
using namespace bfr::test;

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
                               .depthStencil = {},
                               .occlusionQueries = {}};

// What bfr_acquire says of swapchain, which may be stale: bfr.hpp names a
// swapchain only by a live Swapchain.
Result acquireResult(const Device& device, bfr_swapchain swapchain) {
    bfr_swapchain_status status{};
    return resultOf(bfr_acquire(device.get(), swapchain, kTestWait, &status));
}

// What one frame's calls failed with, or nothing; it holds no string, so
// checking it allocates nothing.
struct FrameFailure {
    const char* call = nullptr;
    Result code{};
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
        auto created = createInstance(recordingInto(backend, messages), kValidationOff);
        INFO(failureOf(created));
        REQUIRE(created.has_value());
        const Instance instance = std::move(*created);
        const Device device = presentingDevice(instance, kHeadless);
        auto swapchain = device.createSwapchain(swapchainDesc(instance));
        INFO(failureOf(swapchain));
        REQUIRE(swapchain.has_value());
        REQUIRE(swapchain->resize(kWidth, kHeight).has_value());
        auto pool = device.createCommandPool({.queue = 0, .listCount = kListsPerFrame});
        INFO(failureOf(pool));
        REQUIRE(pool.has_value());
        auto retiring = device.createCommandPool({.queue = 0, .listCount = 1});
        INFO(failureOf(retiring));
        REQUIRE(retiring.has_value());
        auto source = device.createBuffer(
            {.size = kCopyBytes, .memory = BufferMemory::Upload, .usage = {.copySrc = true}});
        REQUIRE(source.has_value());
        auto target = device.createBuffer(
            {.size = kCopyBytes, .memory = BufferMemory::Device, .usage = {.copyDst = true}});
        REQUIRE(target.has_value());
        auto texture = device.createTexture({.dimension = TextureDimension::D2,
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
        // The pool destroyed in frame kRetireFrame, whose next use is checked
        // once the allocations are counted.
        CommandPoolRef retired;
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
                check("wait", device.wait(last, kTestWait));
                check("resetCommandPool", pool->reset());
                check("acquire", swapchain->acquire(kTestWait));
                std::array<CommandList, kListsPerFrame> lists{};
                for (std::size_t index = 0; index < lists.size(); ++index) {
                    const auto commands = pool->beginCommands();
                    if (!check("beginCommands", commands)) {
                        break;
                    }
                    lists.at(index) = *commands;
                    if (index < kRenderingLists) {
                        check("beginRendering", device.beginRendering(*commands, *swapchain, kClear));
                        check("endRendering", device.endRendering(*commands));
                    }
                    if (index == kRenderingLists) {
                        check("copyBuffer", device.copyBuffer(*commands, *source, *target, std::span{&kCopy, 1}));
                        check("barrier", device.barrier(*commands));
                        check("textureBarrier", device.textureBarrier(*commands, *texture, TextureLayout::Undefined,
                                                                      TextureLayout::CopyDst, {}));
                        check("copyBufferToTexture",
                              device.copyBufferToTexture(*commands, *source, *texture, std::span{&kTextureCopy, 1}));
                    }
                }
                const std::array<Token, 2> waits{last, last};
                const auto submitted = device.submit(
                    {.queue = 0, .commandLists = lists, .waits = std::span{waits}.first(frame == 0 ? 0 : 2)});
                if (check("submit", submitted)) {
                    last = *submitted;
                }
                check("present", swapchain->present());
                if (frame == kRetireFrame) {
                    const auto commands = retiring->beginCommands();
                    if (check("beginCommands", commands)) {
                        const auto sent = device.submit(
                            {.queue = 0, .commandLists = std::span{&*commands, 1}, .waits = {}});
                        check("submit", sent);
                    }
                    retired = *retiring;
                    *retiring = CommandPool{};
                }
                check("poll", device.poll(last));
            }
            check("wait", device.wait(last, kTestWait));
            allocations = counter.allocations();
            frees = counter.frees();
        }
        INFO("first failure: " << (failed.call == nullptr ? "none" : failed.call) << " " << name(failed.code));
        CHECK(failed.call == nullptr);
        CHECK(allocations == 0);
        CHECK(frees == 0);
        // bfr.h's destroy calls return nothing, so each destroy is checked by
        // the destroyed object's next use being refused.
        REQUIRE(static_cast<bool>(retired));
        CHECK(resetResult(device, retired.get()) == Result::ErrorInvalidHandle);
        const CommandPoolRef destroyedPool = *pool;
        *pool = CommandPool{};
        CHECK(resetResult(device, destroyedPool.get()) == Result::ErrorInvalidHandle);
        const SwapchainRef destroyedSwapchain = *swapchain;
        *swapchain = Swapchain{};
        CHECK(acquireResult(device, destroyedSwapchain.get()) == Result::ErrorInvalidHandle);
        const BufferRef destroyedSource = *source;
        *source = Buffer{};
        CHECK(slotsResult(device, destroyedSource.get()) == Result::ErrorInvalidHandle);
        const BufferRef destroyedTarget = *target;
        *target = Buffer{};
        CHECK(slotsResult(device, destroyedTarget.get()) == Result::ErrorInvalidHandle);
        const TextureRef destroyedTexture = *texture;
        *texture = Texture{};
        CHECK(slotsResult(device, destroyedTexture.get()) == Result::ErrorInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

// A top structure rebuilt each frame from instances written that frame, as a
// scene that moves does it, and its bottom one with it.
TEST_CASE("Device allocates nothing building acceleration structures frame after frame", "[rhi][agnostic]") {
    if (!bfr::test::AllocationCounter::counts()) {
        SKIP("a sanitizer replaces operator new, so allocations are not counted");
    }
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        auto created = createInstance(recordingInto(backend, messages), kValidationOff);
        INFO(failureOf(created));
        REQUIRE(created.has_value());
        const Instance instance = std::move(*created);
        const Caps caps = capsOf(instance, kFirstAdapter);
        if (!caps.features.accelerationStructure) {
            SKIP("the adapter has no acceleration structures");
        }
        auto made = instance.createDevice(
            testDeviceWith(instance, kFirstAdapter, kNoWindow, [](DeviceDesc& desc) { desc.maxBuildGeometries = 1; }));
        INFO(failureOf(made));
        REQUIRE(made.has_value());
        const Device& device = *made;
        constexpr std::uint32_t kVertexFloats = 3;
        constexpr std::array<float, std::size_t{kVertexFloats} * 3> kTriangle{-1, -1, 0, 1, -1, 0, 0, 1, 0};
        auto vertices = device.createBuffer(
            {.size = sizeof(kTriangle), .memory = BufferMemory::Upload, .usage = {.accelerationStructureInput = true}});
        REQUIRE(vertices.has_value());
        auto instances = device.createBuffer({.size = caps.rayTracing.instanceSize,
                                              .memory = BufferMemory::Upload,
                                              .usage = {.accelerationStructureInput = true}});
        REQUIRE(instances.has_value());
        const TriangleGeometry triangle{.vertices = *vertices,
                                        .vertexFormat = Format::Rgb32Float,
                                        .vertexStride = kVertexFloats * sizeof(float),
                                        .vertexCount = 3,
                                        .triangleCount = 1,
                                        .opaque = true};
        const AccelerationStructureInputs bottomInputs{.triangles = std::span{&triangle, 1}};
        const AccelerationStructureInputs topInputs{.type = AccelerationStructureType::Top,
                                                    .instances = *instances,
                                                    .instanceCount = 1};
        const auto bottomSizes = device.accelerationStructureSizes(bottomInputs);
        const auto topSizes = device.accelerationStructureSizes(topInputs);
        REQUIRE(bottomSizes.has_value());
        REQUIRE(topSizes.has_value());
        auto bottom = device.createAccelerationStructure({.size = bottomSizes->size});
        auto top = device.createAccelerationStructure({.type = AccelerationStructureType::Top, .size = topSizes->size});
        REQUIRE(bottom.has_value());
        REQUIRE(top.has_value());
        auto bottomScratch = device.createBuffer(
            {.size = bottomSizes->buildScratch, .usage = {.accelerationStructureScratch = true}});
        auto topScratch = device.createBuffer(
            {.size = topSizes->buildScratch, .usage = {.accelerationStructureScratch = true}});
        REQUIRE(bottomScratch.has_value());
        REQUIRE(topScratch.has_value());
        auto pool = device.createCommandPool({.queue = 0, .listCount = 1});
        REQUIRE(pool.has_value());
        const auto mapped = instances->map();
        REQUIRE(mapped.has_value());

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
                check("wait", device.wait(last, kTestWait));
                check("resetCommandPool", pool->reset());
                const AccelerationStructureInstance placed{.transform = {1, 0, 0, static_cast<float>(frame), 0, 1, 0, 0,
                                                                         0, 0, 1, 0},
                                                           .mask = 1,
                                                           .bottomStructure = *bottom};
                check("writeInstances",
                      device.writeInstances(std::span{&placed, 1}, *mapped, caps.rayTracing.instanceSize));
                check("accelerationStructureSizes", device.accelerationStructureSizes(topInputs));
                const auto commands = pool->beginCommands();
                if (!check("beginCommands", commands)) {
                    break;
                }
                check("buildAccelerationStructure",
                      device.buildAccelerationStructure(
                          *commands, {.dst = *bottom, .inputs = bottomInputs, .scratch = *bottomScratch}));
                check("barrier", device.barrier(*commands));
                check("buildAccelerationStructure",
                      device.buildAccelerationStructure(*commands,
                                                        {.dst = *top, .inputs = topInputs, .scratch = *topScratch}));
                const auto submitted = device.submit(
                    {.queue = 0, .commandLists = std::span{&*commands, 1}, .waits = {}});
                if (check("submit", submitted)) {
                    last = *submitted;
                }
            }
            check("wait", device.wait(last, kTestWait));
            allocations = counter.allocations();
            frees = counter.frees();
        }
        INFO("first failure: " << (failed.call == nullptr ? "none" : failed.call) << " " << name(failed.code));
        CHECK(failed.call == nullptr);
        CHECK(allocations == 0);
        CHECK(frees == 0);
        // bfr.h's destroy calls return nothing, so each destroy is checked by
        // the destroyed object's next use being refused.
        const CommandPoolRef destroyedPool = *pool;
        *pool = CommandPool{};
        CHECK(resetResult(device, destroyedPool.get()) == Result::ErrorInvalidHandle);
        const AccelerationStructureRef destroyedTop = *top;
        *top = AccelerationStructure{};
        CHECK(captureResult(device, destroyedTop.get()) == Result::ErrorInvalidHandle);
        const AccelerationStructureRef destroyedBottom = *bottom;
        *bottom = AccelerationStructure{};
        CHECK(captureResult(device, destroyedBottom.get()) == Result::ErrorInvalidHandle);
        for (auto* buffer : {&vertices, &instances, &bottomScratch, &topScratch}) {
            const BufferRef destroyedBuffer = **buffer;
            **buffer = Buffer{};
            CHECK(slotsResult(device, destroyedBuffer.get()) == Result::ErrorInvalidHandle);
        }
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
        auto created = createInstance(recordingInto(backend, messages), kValidationOff);
        INFO(failureOf(created));
        REQUIRE(created.has_value());
        const Instance instance = std::move(*created);
        const Device device = presentingDevice(instance, kHeadless);
        auto swapchain = device.createSwapchain(swapchainDesc(instance));
        INFO(failureOf(swapchain));
        REQUIRE(swapchain.has_value());

        int refused = 0;
        std::size_t allocations = 0;
        std::size_t frees = 0;
        {
            const bfr::test::AllocationCounter counter;
            for (int frame = 0; frame < kFrames; ++frame) {
                const auto acquired = swapchain->acquire(kTestWait);
                refused += !acquired.has_value() && acquired.error().code == Result::ErrorOutOfDate ? 1 : 0;
            }
            allocations = counter.allocations();
            frees = counter.frees();
        }
        CHECK(refused == kFrames);
        CHECK(allocations == 0);
        CHECK(frees == 0);
        // bfr.h's destroy returns nothing, so the destroy is checked by the
        // destroyed swapchain's next use being refused.
        const SwapchainRef destroyedSwapchain = *swapchain;
        *swapchain = Swapchain{};
        CHECK(acquireResult(device, destroyedSwapchain.get()) == Result::ErrorInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}
