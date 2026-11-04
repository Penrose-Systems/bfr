// What every backend does with draws and dispatches, through the interface
// alone: the slots resources get, a dispatch that writes a buffer, a draw into
// a texture read back, and the usage checks. Each case runs on every backend
// the build has; the shaders are the SPIR-V in tests/library/shaders, so a case
// that runs code skips on a backend with none checked in.

#include "bfr/core/error.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/instance.h"
#include "bfr/rhi/pipeline.h"
#include "bfr/rhi/texture.h"
#include "rhi_test_support.h"
#include "shaders/spirv.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_container_properties.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace bfr::rhi;
using namespace bfr::rhi::test;
using bfr::core::ErrorCode;

namespace {

constexpr std::uint32_t kWorkgroup = 64;
constexpr std::uint32_t kSide = 8;
constexpr std::uint32_t kTexelBytes = 4;
constexpr std::uint32_t kVertices = 3;
constexpr std::uint32_t kPositionFloats = 4;

// An Instance and a device on its first adapter, presenting nowhere.
struct DrawDevice {
    explicit DrawDevice(const TestInstanceDesc& desc) {
        auto created = createInstance(desc, kValidation);
        INFO(failureOf(created));
        REQUIRE(created.has_value());
        instance = std::move(*created);
        auto made = instance->createDevice(0, kNoWindow, kTestDevice);
        INFO(failureOf(made));
        REQUIRE(made.has_value());
        device = std::move(*made);
    }

    std::unique_ptr<Instance> instance;
    std::unique_ptr<Device> device;
};

// Whether backend takes the SPIR-V checked in, as test_pipeline.cpp says.
bool takesSpirv(const TestBackend& backend) {
    return !backend.backend.has_value() || *backend.backend == Backend::Vulkan;
}

// Whether backend runs what it records: the null backend runs nothing, so
// what it leaves in a buffer is not checked.
bool runs(const TestBackend& backend) {
    return backend.backend.has_value();
}

ShaderCode spirv(std::span<const std::uint8_t> bytes) {
    return {.code = std::as_bytes(bytes), .entryPoint = "main"};
}

BufferHandle created(Device& device, const BufferDesc& desc) {
    const auto buffer = device.createBuffer(desc);
    INFO(failureOf(buffer));
    REQUIRE(buffer.has_value());
    return *buffer;
}

TextureHandle created(Device& device, const TextureDesc& desc) {
    const auto texture = device.createTexture(desc);
    INFO(failureOf(texture));
    REQUIRE(texture.has_value());
    return *texture;
}

PipelineHandle created(Device& device, const ComputePipelineDesc& desc) {
    const auto pipeline = device.createComputePipeline(desc);
    INFO(failureOf(pipeline));
    REQUIRE(pipeline.has_value());
    return *pipeline;
}

PipelineHandle created(Device& device, const GraphicsPipelineDesc& desc) {
    const auto pipeline = device.createGraphicsPipeline(desc);
    INFO(failureOf(pipeline));
    REQUIRE(pipeline.has_value());
    return *pipeline;
}

// The push constants the test shaders declare: the slots they read, then the
// color color.frag writes.
struct Push {
    std::uint32_t positions = 0;
    std::uint32_t textureIndex = 0;
    std::uint32_t samplerIndex = 0;
    std::uint32_t pad = 0;
    std::array<float, 4> color{};
};

std::array<std::byte, sizeof(Push)> bytesOf(const Push& push) {
    return std::bit_cast<std::array<std::byte, sizeof(Push)>>(push);
}

// A pipeline drawing color.frag's color into one Rgba8Unorm target, culling
// nothing, so the test's one triangle shows whichever way it winds.
GraphicsPipelineDesc colorPipeline() {
    GraphicsPipelineDesc desc{.vertex = spirv(bfr::test::spirv::kBindlessVert),
                              .fragment = spirv(bfr::test::spirv::kColorFrag),
                              .topology = TopologyClass::Triangle,
                              .raster = {.cull = CullMode::None},
                              .depthStencil = {},
                              .colorTargetCount = 1,
                              .depthStencilFormat = Format::Undefined,
                              .samples = 1};
    desc.colorTargets[0] = {.format = Format::Rgba8Unorm, .blend = {}};
    return desc;
}

std::uint64_t alignUp(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

} // namespace

TEST_CASE("Device gives a storage buffer and a sampled or storage texture a slot in its table, and no other",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const DrawDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        constexpr std::uint64_t kBytes = 256;
        const BufferHandle first = created(device, {.size = kBytes, .usage = {.storage = true}});
        const BufferHandle second = created(device, {.size = kBytes, .usage = {.storage = true}});
        const BufferHandle plain = created(device, {.size = kBytes, .usage = {.copyDst = true}});
        const auto firstSlot = device.bufferSlot(first);
        const auto secondSlot = device.bufferSlot(second);
        REQUIRE(firstSlot.has_value());
        REQUIRE(secondSlot.has_value());
        CHECK(*firstSlot != *secondSlot);
        CHECK(*firstSlot < kTestDevice.maxBindlessStorageBuffers);
        CHECK(*secondSlot < kTestDevice.maxBindlessStorageBuffers);
        CHECK(device.bufferSlot(plain).error().code == ErrorCode::RhiInvalidCall);

        const TextureDesc square{.dimension = TextureDimension::D2,
                                 .format = Format::Rgba8Unorm,
                                 .width = kSide,
                                 .height = kSide,
                                 .depth = 1,
                                 .layers = 1,
                                 .mipLevels = 1,
                                 .samples = 1,
                                 .usage = {.sampled = true, .storage = true}};
        const TextureHandle both = created(device, square);
        TextureDesc copied = square;
        copied.usage = {.copySrc = true, .copyDst = true};
        const TextureHandle neither = created(device, copied);
        const auto bothSlots = device.textureSlots(both);
        REQUIRE(bothSlots.has_value());
        CHECK(bothSlots->sampled < kTestDevice.maxBindlessTextures);
        CHECK(bothSlots->storage < kTestDevice.maxBindlessStorageTextures);
        const auto neitherSlots = device.textureSlots(neither);
        REQUIRE(neitherSlots.has_value());
        CHECK(neitherSlots->sampled == 0);
        CHECK(neitherSlots->storage == 0);

        for (const BufferHandle buffer : {first, second, plain}) {
            CHECK(device.destroyBuffer(buffer).has_value());
        }
        CHECK(device.destroyTexture(both).has_value());
        CHECK(device.destroyTexture(neither).has_value());
        CHECK(device.bufferSlot(first).error().code == ErrorCode::RhiInvalidHandle);
        CHECK(device.textureSlots(both).error().code == ErrorCode::RhiInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device gives a depth-stencil texture that shaders sample a slot, and renders to it", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    const Format format = GENERATE(Format::D24UnormS8Uint, Format::D32FloatS8Uint);
    std::vector<std::string> messages;
    {
        const DrawDevice made{recordingInto(backend, messages)};
        const auto caps = made.instance->formatCaps(0, format);
        REQUIRE(caps.has_value());
        if (!caps->sampled || !caps->depthStencil) {
            SKIP("the adapter does not sample and render to " << static_cast<int>(format));
        }
        Device& device = *made.device;
        const TextureHandle depth = created(device, {.dimension = TextureDimension::D2,
                                                     .format = format,
                                                     .width = kSide,
                                                     .height = kSide,
                                                     .depth = 1,
                                                     .layers = 1,
                                                     .mipLevels = 1,
                                                     .samples = 1,
                                                     .usage = {.sampled = true, .depthStencil = true}});
        const auto slots = device.textureSlots(depth);
        REQUIRE(slots.has_value());
        CHECK(slots->sampled < kTestDevice.maxBindlessTextures);

        const auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        auto list = device.beginCommands(*pool);
        REQUIRE(list.has_value());
        CHECK(device.textureBarrier(*list, depth, TextureLayout::Undefined, TextureLayout::DepthStencilTarget)
                  .has_value());
        const RenderingDesc rendering{.colorAttachmentCount = 0,
                                      .colorAttachments = {},
                                      .depthStencil = {.texture = depth,
                                                       .loadOp = LoadOp::Clear,
                                                       .storeOp = StoreOp::Store,
                                                       .clearDepth = 1,
                                                       .clearStencil = 0}};
        const auto begun = device.beginRendering(*list, {}, rendering);
        INFO(failureOf(begun));
        REQUIRE(begun.has_value());
        CHECK(device.endRendering(*list).has_value());
        CHECK(
            device.textureBarrier(*list, depth, TextureLayout::DepthStencilTarget, TextureLayout::Sampled).has_value());
        const auto sent = device.submit({.queue = 0, .commandLists = std::span{&*list, 1}, .waits = {}});
        INFO(failureOf(sent));
        REQUIRE(sent.has_value());
        REQUIRE(device.wait(*sent).has_value());
        CHECK(device.destroyTexture(depth).has_value());
        CHECK(device.destroyCommandPool(*pool).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device dispatches a compute pipeline that writes a storage buffer through its slot", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    if (!takesSpirv(backend)) {
        SKIP("no " << backend.name << " bytecode is checked in");
    }
    std::vector<std::string> messages;
    {
        const DrawDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        constexpr std::uint64_t kBytes = std::uint64_t{kWorkgroup} * sizeof(std::uint32_t);
        const BufferHandle target = created(device, {.size = kBytes, .usage = {.copySrc = true, .storage = true}});
        const BufferHandle readback =
            created(device, {.size = kBytes, .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
        const PipelineHandle pipeline = created(device,
                                                ComputePipelineDesc{.compute = spirv(bfr::test::spirv::kBindlessComp),
                                                                    .workgroupSize = {kWorkgroup, 1, 1}});
        const auto slot = device.bufferSlot(target);
        REQUIRE(slot.has_value());
        const auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists, .pushConstantUpdates = 1});
        REQUIRE(pool.has_value());
        auto list = device.beginCommands(*pool);
        REQUIRE(list.has_value());

        CHECK(device.bindPipeline(*list, pipeline).has_value());
        CHECK(device.pushConstants(*list, bytesOf({.positions = *slot})).has_value());
        CHECK(device.dispatch(*list, 1, 1, 1).has_value());
        CHECK(device.barrier(*list).has_value());
        const BufferCopy whole{.srcOffset = 0, .dstOffset = 0, .size = kBytes};
        CHECK(device.copyBuffer(*list, target, readback, std::span{&whole, 1}).has_value());
        const auto sent = device.submit({.queue = 0, .commandLists = std::span{&*list, 1}, .waits = {}});
        INFO(failureOf(sent));
        REQUIRE(sent.has_value());
        REQUIRE(device.wait(*sent).has_value());

        if (runs(backend)) {
            // The shader writes each invocation its own index.
            const auto read = device.mapBuffer(readback);
            REQUIRE(read.has_value());
            std::array<std::uint32_t, kWorkgroup> values{};
            std::memcpy(values.data(), read->data(), kBytes);
            for (std::uint32_t i = 0; i < kWorkgroup; ++i) {
                CHECK(values.at(i) == i);
            }
        }
        CHECK(device.destroyPipeline(pipeline).has_value());
        CHECK(device.destroyBuffer(target).has_value());
        CHECK(device.destroyBuffer(readback).has_value());
        CHECK(device.destroyCommandPool(*pool).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device draws a triangle pulled from a storage buffer into a texture, with y up", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    if (!takesSpirv(backend)) {
        SKIP("no " << backend.name << " bytecode is checked in");
    }
    std::vector<std::string> messages;
    {
        const DrawDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        const Limits& limits = made.instance->adapters()[0].caps.limits;
        const std::uint64_t bytesPerRow = alignUp(std::uint64_t{kSide} * kTexelBytes,
                                                  std::max<std::uint64_t>(limits.copyRowPitchAlignment, 1));
        const std::uint64_t imageBytes = bytesPerRow * kSide;

        // The top half of clip space: y from 0 up to 1, which lands in the
        // top rows of the texture on every backend.
        constexpr std::array<float, std::size_t{kVertices} * kPositionFloats> kTopHalf{-1, 0, 0, 1, 1, 0,
                                                                                       0,  1, 0, 1, 0, 1};
        const BufferHandle positions =
            created(device, {.size = sizeof(kTopHalf), .memory = BufferMemory::Upload, .usage = {.storage = true}});
        const auto written = device.mapBuffer(positions);
        REQUIRE(written.has_value());
        std::memcpy(written->data(), kTopHalf.data(), sizeof(kTopHalf));
        const auto slot = device.bufferSlot(positions);
        REQUIRE(slot.has_value());

        const TextureHandle target = created(device, {.dimension = TextureDimension::D2,
                                                      .format = Format::Rgba8Unorm,
                                                      .width = kSide,
                                                      .height = kSide,
                                                      .depth = 1,
                                                      .layers = 1,
                                                      .mipLevels = 1,
                                                      .samples = 1,
                                                      .usage = {.copySrc = true, .colorAttachment = true}});
        const BufferHandle readback =
            created(device, {.size = imageBytes, .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
        const PipelineHandle pipeline = created(device, colorPipeline());
        const auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists, .pushConstantUpdates = 1});
        REQUIRE(pool.has_value());
        auto list = device.beginCommands(*pool);
        REQUIRE(list.has_value());

        CHECK(device.textureBarrier(*list, target, TextureLayout::Undefined, TextureLayout::ColorTarget).has_value());
        RenderingDesc rendering{.colorAttachmentCount = 1, .colorAttachments = {}, .depthStencil = {}};
        rendering.colorAttachments[0] = {.texture = target,
                                         .loadOp = LoadOp::Clear,
                                         .storeOp = StoreOp::Store,
                                         .clearColor = {0, 0, 0, 1}};
        const auto begun = device.beginRendering(*list, {}, rendering);
        INFO(failureOf(begun));
        REQUIRE(begun.has_value());
        CHECK(device.bindPipeline(*list, pipeline).has_value());
        CHECK(device.pushConstants(*list, bytesOf({.positions = *slot, .color = {1, 0, 0, 1}})).has_value());
        CHECK(device.draw(*list, kVertices, 1, 0, 0).has_value());
        CHECK(device.endRendering(*list).has_value());
        CHECK(device.textureBarrier(*list, target, TextureLayout::ColorTarget, TextureLayout::CopySrc).has_value());
        const BufferTextureCopy whole{.bufferOffset = 0,
                                      .bytesPerRow = bytesPerRow,
                                      .rowsPerImage = kSide,
                                      .mipLevel = 0,
                                      .layer = 0,
                                      .aspect = TextureAspect::Color,
                                      .x = 0,
                                      .y = 0,
                                      .z = 0,
                                      .width = kSide,
                                      .height = kSide,
                                      .depth = 1};
        CHECK(device.copyTextureToBuffer(*list, target, readback, std::span{&whole, 1}).has_value());
        const auto sent = device.submit({.queue = 0, .commandLists = std::span{&*list, 1}, .waits = {}});
        INFO(failureOf(sent));
        REQUIRE(sent.has_value());
        REQUIRE(device.wait(*sent).has_value());

        if (runs(backend)) {
            // Row 0 is the top of the texture: red where the triangle is, the
            // clear below it.
            const auto read = device.mapBuffer(readback);
            REQUIRE(read.has_value());
            const auto texel = [&](std::uint32_t x, std::uint32_t y) {
                std::array<std::uint8_t, kTexelBytes> rgba{};
                std::memcpy(rgba.data(), read->data() + (y * bytesPerRow) + (std::uint64_t{x} * kTexelBytes),
                            kTexelBytes);
                return rgba;
            };
            constexpr std::array<std::uint8_t, kTexelBytes> kRed{255, 0, 0, 255};
            constexpr std::array<std::uint8_t, kTexelBytes> kBlack{0, 0, 0, 255};
            CHECK(texel(kSide / 2, 1) == kRed);
            CHECK(texel(kSide / 2, kSide - 1) == kBlack);
            CHECK(texel(0, kSide - 1) == kBlack);
        }
        CHECK(device.destroyPipeline(pipeline).has_value());
        CHECK(device.destroyTexture(target).has_value());
        CHECK(device.destroyBuffer(positions).has_value());
        CHECK(device.destroyBuffer(readback).has_value());
        CHECK(device.destroyCommandPool(*pool).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device refuses draws, dispatches and state recorded out of place with RhiInvalidCall", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    if (!takesSpirv(backend)) {
        SKIP("no " << backend.name << " bytecode is checked in");
    }
    std::vector<std::string> messages;
    {
        const DrawDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        const TextureHandle target = created(device, {.dimension = TextureDimension::D2,
                                                      .format = Format::Rgba8Unorm,
                                                      .width = kSide,
                                                      .height = kSide,
                                                      .depth = 1,
                                                      .layers = 1,
                                                      .mipLevels = 1,
                                                      .samples = 1,
                                                      .usage = {.colorAttachment = true}});
        const BufferHandle indices = created(device, {.size = kSide, .usage = {.index = true}});
        const PipelineHandle graphics = created(device, colorPipeline());
        const PipelineHandle compute = created(device,
                                               ComputePipelineDesc{.compute = spirv(bfr::test::spirv::kBindlessComp),
                                                                   .workgroupSize = {kWorkgroup, 1, 1}});
        constexpr std::uint32_t kUpdates = 2;
        const auto pool = device.createCommandPool(
            {.queue = 0, .listCount = kPoolLists, .pushConstantUpdates = kUpdates});
        REQUIRE(pool.has_value());
        auto list = device.beginCommands(*pool);
        REQUIRE(list.has_value());
        const Viewport whole{.x = 0, .y = 0, .width = kSide, .height = kSide, .minDepth = 0, .maxDepth = 1};
        const Rect all{.x = 0, .y = 0, .width = kSide, .height = kSide};

        // Outside rendering: no draw or draw state; a dispatch wants a pipeline.
        CHECK(device.setViewport(*list, whole).error().code == ErrorCode::RhiInvalidCall);
        CHECK(device.setScissor(*list, all).error().code == ErrorCode::RhiInvalidCall);
        CHECK(device.draw(*list, kVertices, 1, 0, 0).error().code == ErrorCode::RhiInvalidCall);
        CHECK(device.dispatch(*list, 1, 1, 1).error().code == ErrorCode::RhiInvalidCall);
        CHECK(device.bindPipeline(*list, compute).has_value());
        CHECK(device.dispatch(*list, 1, 1, 1).has_value());
        // Push constants are whole 32-bit values, at most kPushConstantsSize,
        // and at most the pool's updates.
        constexpr std::uint32_t kOddBytes = 6;
        CHECK(device.pushConstants(*list, std::array<std::byte, kOddBytes>{}).error().code ==
              ErrorCode::RhiInvalidCall);
        CHECK(device.pushConstants(*list, std::array<std::byte, kPushConstantsSize + sizeof(std::uint32_t)>{})
                  .error()
                  .code == ErrorCode::RhiInvalidCall);
        for (std::uint32_t update = 0; update < kUpdates; ++update) {
            CHECK(device.pushConstants(*list, bytesOf({})).has_value());
        }
        CHECK(device.pushConstants(*list, bytesOf({})).error().code == ErrorCode::RhiLimitReached);

        // A rendering wants an attachment, and a texture where it says one.
        const RenderingDesc none{};
        CHECK(device.beginRendering(*list, {}, none).error().code == ErrorCode::RhiInvalidCall);
        const RenderingDesc unnamed{.colorAttachmentCount = 1, .colorAttachments = {}, .depthStencil = {}};
        CHECK(device.beginRendering(*list, {}, unnamed).error().code == ErrorCode::RhiInvalidCall);
        RenderingDesc rendering = unnamed;
        rendering.colorAttachments[0].texture = target;
        CHECK(device.textureBarrier(*list, target, TextureLayout::Undefined, TextureLayout::ColorTarget).has_value());
        REQUIRE(device.beginRendering(*list, {}, rendering).has_value());

        // Inside rendering: no dispatch; a draw wants a pipeline, an indexed
        // one an index buffer too.
        CHECK(device.dispatch(*list, 1, 1, 1).error().code == ErrorCode::RhiInvalidCall);
        CHECK(device.draw(*list, kVertices, 1, 0, 0).error().code == ErrorCode::RhiInvalidCall);
        CHECK(device.bindPipeline(*list, graphics).has_value());
        CHECK(device.drawIndexed(*list, kVertices, 1, 0, 0, 0).error().code == ErrorCode::RhiInvalidCall);
        CHECK(device.bindIndexBuffer(*list, indices, 0, IndexType::Uint16).has_value());
        CHECK(device.drawIndexed(*list, 0, 0, 0, 0, 0).has_value());
        CHECK(device.setViewport(*list, whole).has_value());
        CHECK(device.setScissor(*list, all).has_value());
        CHECK(device.endRendering(*list).has_value());
        CHECK(device.endRendering(*list).error().code == ErrorCode::RhiInvalidCall);

        CHECK(device.destroyCommandList(*list).has_value());
        CHECK(device.destroyPipeline(graphics).has_value());
        CHECK(device.destroyPipeline(compute).has_value());
        CHECK(device.destroyBuffer(indices).has_value());
        CHECK(device.destroyTexture(target).has_value());
        CHECK(device.destroyCommandPool(*pool).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}
