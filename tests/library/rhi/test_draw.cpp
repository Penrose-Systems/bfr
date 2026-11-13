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
#include "bfr/rhi/query.h"
#include "bfr/rhi/sampler.h"
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
#include <tuple>
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

// Sets the viewport and scissor to the whole of a kSide by kSide target.
void viewWhole(Device& device, CommandListHandle list) {
    CHECK(device.setViewport(list, {.x = 0, .y = 0, .width = kSide, .height = kSide, .minDepth = 0, .maxDepth = 1})
              .has_value());
    CHECK(device.setScissor(list, {.x = 0, .y = 0, .width = kSide, .height = kSide}).has_value());
}

// An Instance and a device on its first adapter, presenting nowhere.
struct DrawDevice {
    explicit DrawDevice(const InstanceDesc& desc, const DeviceDesc& deviceDesc = kTestDevice) {
        auto created = createInstance(desc, kValidation);
        INFO(failureOf(created));
        REQUIRE(created.has_value());
        instance = std::move(*created);
        auto made = instance->createDevice(0, kNoWindow, deviceDesc);
        INFO(failureOf(made));
        REQUIRE(made.has_value());
        device = std::move(*made);
    }

    std::unique_ptr<Instance> instance;
    std::unique_ptr<Device> device;
};

// Whether backend takes the SPIR-V checked in, as test_pipeline.cpp says.
bool takesSpirv(const TestBackend& backend) {
    return backend.backend == Backend::Vulkan;
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
                              .topology = Topology::TriangleList,
                              .primitiveRestart = PrimitiveRestart::Off,
                              .raster = {.cull = CullMode::None},
                              .depthStencil = {},
                              .colorTargetCount = 1,
                              .depthStencilFormat = Format::Undefined,
                              .samples = 1,
                              .sampleMask = BFR_MASK_ALL};
    desc.colorTargets[0] = {.format = Format::Rgba8Unorm, .blend = {.writeMask = BFR_MASK_ALL}};
    return desc;
}

std::uint64_t alignUp(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

// An Upload buffer of usage holding data.
template <class T, std::size_t N>
BufferHandle uploaded(Device& device, const std::array<T, N>& data, const BufferUsage& usage) {
    const BufferHandle buffer = created(device, {.size = sizeof(data), .memory = BufferMemory::Upload, .usage = usage});
    const auto written = device.mapBuffer(buffer);
    REQUIRE(written.has_value());
    std::memcpy(written->data(), data.data(), sizeof(data));
    return buffer;
}

// first and second, Stride bytes apart from the start of a buffer of two.
template <class Args, std::uint32_t Stride>
std::array<std::byte, std::size_t{2} * Stride> twoAt(const Args& first, const Args& second) {
    static_assert(sizeof(Args) <= Stride);
    std::array<std::byte, std::size_t{2} * Stride> bytes{};
    std::memcpy(bytes.data(), &first, sizeof(Args));
    std::memcpy(bytes.data() + Stride, &second, sizeof(Args));
    return bytes;
}

// How Rendered draws to its target: straight, or through a texture of
// samples resolved into it, as rendering ends or, with byCommand, by
// resolveTexture after it; with a depth attachment of depth, cleared to 1,
// unless it is Undefined, or readDepth, read-only, when it is not null.
struct Targets {
    std::uint32_t samples = 1;
    bool byCommand = false;
    Format depth = Format::Undefined;
    TextureHandle readDepth{};
};

// What a kSide by kSide Rgba8Unorm target cleared to black holds once record
// has recorded into its rendering, read back row by row.
class Rendered {
public:
    template <class Record>
    Rendered(Device& device, const Limits& limits, Record record, const Targets& targets = {})
        : bytesPerRow_(
              alignUp(std::uint64_t{kSide} * kTexelBytes, std::max<std::uint64_t>(limits.copyRowPitchAlignment, 1))),
          image_(bytesPerRow_ * kSide) {
        const TextureDesc single{.dimension = TextureDimension::D2,
                                 .format = Format::Rgba8Unorm,
                                 .width = kSide,
                                 .height = kSide,
                                 .depth = 1,
                                 .layers = 1,
                                 .mipLevels = 1,
                                 .samples = 1,
                                 .usage = {.copySrc = true, .copyDst = true, .colorAttachment = true}};
        const TextureHandle target = created(device, single);
        const bool multisampled = targets.samples > 1;
        TextureDesc samplesDesc = single;
        samplesDesc.samples = targets.samples;
        samplesDesc.usage = {.copySrc = true, .colorAttachment = true};
        const TextureHandle drawn = multisampled ? created(device, samplesDesc) : target;
        TextureDesc depthDesc = samplesDesc;
        depthDesc.format = targets.depth;
        depthDesc.usage = {.depthStencil = true};
        const TextureHandle depth = targets.depth != Format::Undefined ? created(device, depthDesc) : TextureHandle{};
        const BufferHandle readback =
            created(device, {.size = image_.size(), .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
        // A push for each of a record's draws.
        constexpr std::uint32_t kPushes = 2;
        const auto pool = device.createCommandPool(
            {.queue = 0, .listCount = kPoolLists, .pushConstantUpdates = kPushes});
        REQUIRE(pool.has_value());
        auto list = device.beginCommands(*pool);
        REQUIRE(list.has_value());
        CHECK(
            device.textureBarrier(*list, target, TextureLayout::Undefined, TextureLayout::ColorTarget, {}).has_value());
        if (multisampled) {
            CHECK(device.textureBarrier(*list, drawn, TextureLayout::Undefined, TextureLayout::ColorTarget, {})
                      .has_value());
        }
        if (!depth.isNull()) {
            CHECK(device.textureBarrier(*list, depth, TextureLayout::Undefined, TextureLayout::DepthStencilTarget, {})
                      .has_value());
        }
        RenderingDesc rendering{.colorAttachmentCount = 1,
                                .colorAttachments = {},
                                .depthStencil = {.texture = depth,
                                                 .depthLoadOp = LoadOp::Clear,
                                                 .depthStoreOp = StoreOp::DontCare,
                                                 .stencilLoadOp = LoadOp::Clear,
                                                 .stencilStoreOp = StoreOp::DontCare,
                                                 .clearDepth = 1},
                                .occlusionQueries = {}};
        if (!targets.readDepth.isNull()) {
            rendering.depthStencil = {.texture = targets.readDepth, .readOnly = true};
        }
        rendering.colorAttachments[0] = {
            .texture = drawn,
            .loadOp = LoadOp::Clear,
            .storeOp = StoreOp::Store,
            .clearColor = {0, 0, 0, 1},
            .resolveTexture = multisampled && !targets.byCommand ? target : TextureHandle{},
        };
        const auto begun = device.beginRendering(*list, {}, rendering);
        INFO(failureOf(begun));
        REQUIRE(begun.has_value());
        viewWhole(device, *list);
        record(*list);
        CHECK(device.endRendering(*list).has_value());
        if (multisampled && targets.byCommand) {
            CHECK(device.textureBarrier(*list, drawn, TextureLayout::ColorTarget, TextureLayout::ResolveSrc, {})
                      .has_value());
            CHECK(device.textureBarrier(*list, target, TextureLayout::ColorTarget, TextureLayout::ResolveDst, {})
                      .has_value());
            const auto resolved = device.resolveTexture(*list, drawn, 0, 0, target, 0, 0);
            INFO(failureOf(resolved));
            CHECK(resolved.has_value());
            CHECK(device.textureBarrier(*list, target, TextureLayout::ResolveDst, TextureLayout::CopySrc, {})
                      .has_value());
        }
        else {
            CHECK(device.textureBarrier(*list, target, TextureLayout::ColorTarget, TextureLayout::CopySrc, {})
                      .has_value());
        }
        const BufferTextureCopy whole{.bufferOffset = 0,
                                      .bytesPerRow = bytesPerRow_,
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
        REQUIRE(device.wait(*sent, kTestWait).has_value());
        const auto read = device.mapBuffer(readback);
        REQUIRE(read.has_value());
        std::memcpy(image_.data(), read->data(), image_.size());
        CHECK(device.destroyTexture(target).has_value());
        if (multisampled) {
            CHECK(device.destroyTexture(drawn).has_value());
        }
        CHECK(device.destroyTexture(depth).has_value());
        CHECK(device.destroyBuffer(readback).has_value());
        CHECK(device.destroyCommandPool(*pool).has_value());
    }

    // The texel at x, y; row 0 is the top.
    [[nodiscard]] std::array<std::uint8_t, kTexelBytes> at(std::uint32_t x, std::uint32_t y) const {
        std::array<std::uint8_t, kTexelBytes> rgba{};
        std::memcpy(rgba.data(), image_.data() + (y * bytesPerRow_) + (std::uint64_t{x} * kTexelBytes), kTexelBytes);
        return rgba;
    }

private:
    std::uint64_t bytesPerRow_;
    std::vector<std::byte> image_;
};

constexpr std::array<std::uint8_t, kTexelBytes> kRed{255, 0, 0, 255};
constexpr std::array<std::uint8_t, kTexelBytes> kBlack{0, 0, 0, 255};

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
        const auto firstSlot = storageSlot(device, first);
        const auto secondSlot = storageSlot(device, second);
        REQUIRE(firstSlot.has_value());
        REQUIRE(secondSlot.has_value());
        CHECK(*firstSlot != *secondSlot);
        CHECK(*firstSlot < kTestDevice.maxBindlessStorageBuffers);
        CHECK(*secondSlot < kTestDevice.maxBindlessStorageBuffers);
        const auto plainSlots = device.bufferSlots(plain);
        REQUIRE(plainSlots.has_value());
        CHECK(plainSlots->storage == kNoSlot);
        CHECK(plainSlots->uniform == kNoSlot);

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
        CHECK(neitherSlots->sampled == kNoSlot);
        CHECK(neitherSlots->storage == kNoSlot);

        for (const BufferHandle buffer : {first, second, plain}) {
            CHECK(device.destroyBuffer(buffer).has_value());
        }
        CHECK(device.destroyTexture(both).has_value());
        CHECK(device.destroyTexture(neither).has_value());
        CHECK(storageSlot(device, first).error().code == ErrorCode::RhiInvalidHandle);
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
        CHECK(device.textureBarrier(*list, depth, TextureLayout::Undefined, TextureLayout::DepthStencilTarget, {})
                  .has_value());
        const RenderingDesc rendering{.colorAttachmentCount = 0,
                                      .colorAttachments = {},
                                      .depthStencil = {.texture = depth,
                                                       .depthLoadOp = LoadOp::Clear,
                                                       .depthStoreOp = StoreOp::Store,
                                                       .stencilLoadOp = LoadOp::Clear,
                                                       .stencilStoreOp = StoreOp::Store,
                                                       .clearDepth = 1,
                                                       .clearStencil = 0},
                                      .occlusionQueries = {}};
        const auto begun = device.beginRendering(*list, {}, rendering);
        INFO(failureOf(begun));
        REQUIRE(begun.has_value());
        CHECK(device.endRendering(*list).has_value());
        CHECK(device.textureBarrier(*list, depth, TextureLayout::DepthStencilTarget, TextureLayout::Sampled, {})
                  .has_value());
        const auto sent = device.submit({.queue = 0, .commandLists = std::span{&*list, 1}, .waits = {}});
        INFO(failureOf(sent));
        REQUIRE(sent.has_value());
        REQUIRE(device.wait(*sent, kTestWait).has_value());
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
        const auto slot = storageSlot(device, target);
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
        REQUIRE(device.wait(*sent, kTestWait).has_value());

        // The shader writes each invocation its own index.
        const auto read = device.mapBuffer(readback);
        REQUIRE(read.has_value());
        std::array<std::uint32_t, kWorkgroup> values{};
        std::memcpy(values.data(), read->data(), kBytes);
        for (std::uint32_t i = 0; i < kWorkgroup; ++i) {
            CHECK(values.at(i) == i);
        }
        CHECK(device.destroyPipeline(pipeline).has_value());
        CHECK(device.destroyBuffer(target).has_value());
        CHECK(device.destroyBuffer(readback).has_value());
        CHECK(device.destroyCommandPool(*pool).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device gives a uniform buffer a slot in its table, which shaders read, where the adapter has one",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    constexpr std::uint32_t kUniforms = 4;
    constexpr std::uint32_t kValue = 0x5EED;
    std::vector<std::string> messages;
    {
        auto opened = createInstance(recordingInto(backend, messages), kValidation);
        INFO(failureOf(opened));
        REQUIRE(opened.has_value());
        const std::unique_ptr<Instance> instance = std::move(*opened);
        const Caps& caps = instance->adapters()[0].caps;
        if (caps.bindless.maxUniformBuffers == 0) {
            // No table to give slots in.
            const auto refused = instance->createDevice(0, kNoWindow, testDeviceWith([](DeviceDesc& desc) {
                                                            desc.maxBindlessUniformBuffers = 1;
                                                        }));
            CHECK(refused.error().code == ErrorCode::RhiLimitReached);
            SKIP("the adapter has no uniform buffer table");
        }
        auto made = instance->createDevice(0, kNoWindow, testDeviceWith([](DeviceDesc& desc) {
                                               desc.maxBindlessUniformBuffers = kUniforms;
                                           }));
        INFO(failureOf(made));
        REQUIRE(made.has_value());
        Device& device = **made;
        const BufferHandle source = created(device, {.size = caps.limits.minUniformBufferAlignment,
                                                     .memory = BufferMemory::Upload,
                                                     .usage = {.uniform = true}});
        const auto sourceSlots = device.bufferSlots(source);
        REQUIRE(sourceSlots.has_value());
        CHECK(sourceSlots->uniform < kUniforms);
        CHECK(sourceSlots->storage == kNoSlot);
        const auto mapped = device.mapBuffer(source);
        REQUIRE(mapped.has_value());
        std::memcpy(mapped->data(), &kValue, sizeof(kValue));

        if (!takesSpirv(backend)) {
            CHECK(device.destroyBuffer(source).has_value());
            SKIP("no " << backend.name << " bytecode is checked in");
        }
        const BufferHandle target = created(device, {.size = sizeof(std::uint32_t),
                                                     .usage = {.copySrc = true, .storage = true}});
        const BufferHandle readback = created(device, {.size = sizeof(std::uint32_t),
                                                       .memory = BufferMemory::Readback,
                                                       .usage = {.copyDst = true}});
        const PipelineHandle pipeline = created(device,
                                                ComputePipelineDesc{.compute = spirv(bfr::test::spirv::kUniformComp),
                                                                    .workgroupSize = {1, 1, 1}});
        const auto targetSlot = storageSlot(device, target);
        REQUIRE(targetSlot.has_value());
        const auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists, .pushConstantUpdates = 1});
        REQUIRE(pool.has_value());
        auto list = device.beginCommands(*pool);
        REQUIRE(list.has_value());

        CHECK(device.bindPipeline(*list, pipeline).has_value());
        const std::array<std::uint32_t, 2> push{*targetSlot, sourceSlots->uniform};
        CHECK(device.pushConstants(*list, std::as_bytes(std::span{push})).has_value());
        CHECK(device.dispatch(*list, 1, 1, 1).has_value());
        CHECK(device.barrier(*list).has_value());
        const BufferCopy whole{.srcOffset = 0, .dstOffset = 0, .size = sizeof(std::uint32_t)};
        CHECK(device.copyBuffer(*list, target, readback, std::span{&whole, 1}).has_value());
        const auto sent = device.submit({.queue = 0, .commandLists = std::span{&*list, 1}, .waits = {}});
        INFO(failureOf(sent));
        REQUIRE(sent.has_value());
        REQUIRE(device.wait(*sent, kTestWait).has_value());

        const auto read = device.mapBuffer(readback);
        REQUIRE(read.has_value());
        std::uint32_t value = 0;
        std::memcpy(&value, read->data(), sizeof(value));
        CHECK(value == kValue);
        CHECK(device.destroyPipeline(pipeline).has_value());
        CHECK(device.destroyBuffer(source).has_value());
        CHECK(device.destroyBuffer(target).has_value());
        CHECK(device.destroyBuffer(readback).has_value());
        CHECK(device.destroyCommandPool(*pool).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device hands work across queues: compute writes and copy reads it, copy uploads and graphics reads it",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    if (!takesSpirv(backend)) {
        SKIP("no " << backend.name << " bytecode is checked in");
    }
    std::vector<std::string> messages;
    {
        auto instance = createInstance(recordingInto(backend, messages), kValidation);
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

        constexpr std::uint64_t kBytes = std::uint64_t{kWorkgroup} * sizeof(std::uint32_t);
        const BufferHandle written = created(device, {.size = kBytes, .usage = {.copySrc = true, .storage = true}});
        const BufferHandle computed =
            created(device, {.size = kBytes, .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
        const PipelineHandle pipeline = created(device,
                                                ComputePipelineDesc{.compute = spirv(bfr::test::spirv::kBindlessComp),
                                                                    .workgroupSize = {kWorkgroup, 1, 1}});
        const auto slot = storageSlot(device, written);
        REQUIRE(slot.has_value());

        const std::uint64_t bytesPerRow = alignUp(std::uint64_t{kSide} * kTexelBytes,
                                                  std::max<std::uint64_t>(caps.limits.copyRowPitchAlignment, 1));
        const std::uint64_t imageBytes = bytesPerRow * kSide;
        const BufferHandle upload =
            created(device, {.size = imageBytes, .memory = BufferMemory::Upload, .usage = {.copySrc = true}});
        const BufferHandle uploaded =
            created(device, {.size = imageBytes, .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
        const auto texels = device.mapBuffer(upload);
        REQUIRE(texels.has_value());
        for (std::size_t index = 0; index < texels->size(); ++index) {
            (*texels)[index] = static_cast<std::byte>(static_cast<unsigned char>(index));
        }
        const auto texture = device.createTexture({.format = Format::Rgba8Unorm,
                                                   .width = kSide,
                                                   .height = kSide,
                                                   .depth = 1,
                                                   .layers = 1,
                                                   .mipLevels = 1,
                                                   .samples = 1,
                                                   .usage = {.copySrc = true, .copyDst = true}});
        REQUIRE(texture.has_value());
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

        const auto graphicsPool = device.createCommandPool({.queue = kGraphics, .listCount = kPoolLists});
        const auto computePool = device.createCommandPool(
            {.queue = kCompute, .listCount = kPoolLists, .pushConstantUpdates = 1});
        const auto copyPool = device.createCommandPool({.queue = kCopy, .listCount = kPoolLists});
        REQUIRE(graphicsPool.has_value());
        REQUIRE(computePool.has_value());
        REQUIRE(copyPool.has_value());
        const auto send = [&device](std::uint32_t queue, CommandListHandle list, std::span<const Token> waits) {
            const auto token = device.submit({.queue = queue, .commandLists = std::span{&list, 1}, .waits = waits});
            INFO(failureOf(token));
            REQUIRE(token.has_value());
            return *token;
        };

        // Compute writes the buffer; the copy queue waits on it and reads it back.
        const auto dispatching = device.beginCommands(*computePool);
        REQUIRE(dispatching.has_value());
        CHECK(device.bindPipeline(*dispatching, pipeline).has_value());
        CHECK(device.pushConstants(*dispatching, bytesOf({.positions = *slot})).has_value());
        CHECK(device.dispatch(*dispatching, 1, 1, 1).has_value());
        const Token dispatched = send(kCompute, *dispatching, {});
        const auto reading = device.beginCommands(*copyPool);
        REQUIRE(reading.has_value());
        const BufferCopy all{.srcOffset = 0, .dstOffset = 0, .size = kBytes};
        CHECK(device.copyBuffer(*reading, written, computed, std::span{&all, 1}).has_value());
        const Token read = send(kCopy, *reading, std::span{&dispatched, 1});

        // The copy queue uploads the texture in a copy layout; graphics waits,
        // moves it on and reads it back.
        const auto uploading = device.beginCommands(*copyPool);
        REQUIRE(uploading.has_value());
        CHECK(device.textureBarrier(*uploading, *texture, TextureLayout::Undefined, TextureLayout::CopyDst, {})
                  .has_value());
        CHECK(device.copyBufferToTexture(*uploading, upload, *texture, std::span{&whole, 1}).has_value());
        const Token filled = send(kCopy, *uploading, std::span{&read, 1});
        const auto consuming = device.beginCommands(*graphicsPool);
        REQUIRE(consuming.has_value());
        CHECK(device.textureBarrier(*consuming, *texture, TextureLayout::CopyDst, TextureLayout::CopySrc, {})
                  .has_value());
        CHECK(device.copyTextureToBuffer(*consuming, *texture, uploaded, std::span{&whole, 1}).has_value());
        const Token used = send(kGraphics, *consuming, std::span{&filled, 1});
        REQUIRE(device.wait(used, kTestWait).has_value());
        CHECK(device.poll(read).value_or(false));

        // The shader writes each invocation its own index.
        const auto values = device.mapBuffer(computed);
        REQUIRE(values.has_value());
        std::array<std::uint32_t, kWorkgroup> indices{};
        std::memcpy(indices.data(), values->data(), kBytes);
        for (std::uint32_t i = 0; i < kWorkgroup; ++i) {
            CHECK(indices.at(i) == i);
        }
        const auto back = device.mapBuffer(uploaded);
        REQUIRE(back.has_value());
        const std::uint64_t rowBytes = std::uint64_t{kSide} * kTexelBytes;
        for (std::uint64_t row = 0; row < kSide; ++row) {
            INFO("row " << row);
            CHECK(std::ranges::equal(back->subspan(row * bytesPerRow, rowBytes),
                                     texels->subspan(row * bytesPerRow, rowBytes)));
        }
        for (const auto pool : {*graphicsPool, *computePool, *copyPool}) {
            CHECK(device.destroyCommandPool(pool).has_value());
        }
        CHECK(device.destroyPipeline(pipeline).has_value());
        CHECK(device.destroyTexture(*texture).has_value());
        for (const BufferHandle buffer : {written, computed, upload, uploaded}) {
            CHECK(device.destroyBuffer(buffer).has_value());
        }
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
        const auto slot = storageSlot(device, positions);
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

        CHECK(
            device.textureBarrier(*list, target, TextureLayout::Undefined, TextureLayout::ColorTarget, {}).has_value());
        RenderingDesc rendering{.colorAttachmentCount = 1,
                                .colorAttachments = {},
                                .depthStencil = {},
                                .occlusionQueries = {}};
        rendering.colorAttachments[0] = {.texture = target,
                                         .loadOp = LoadOp::Clear,
                                         .storeOp = StoreOp::Store,
                                         .clearColor = {0, 0, 0, 1}};
        const auto begun = device.beginRendering(*list, {}, rendering);
        INFO(failureOf(begun));
        REQUIRE(begun.has_value());
        viewWhole(device, *list);
        CHECK(device.bindPipeline(*list, pipeline).has_value());
        CHECK(device.pushConstants(*list, bytesOf({.positions = *slot, .color = {1, 0, 0, 1}})).has_value());
        CHECK(device.draw(*list, kVertices, 1, 0, 0).has_value());
        CHECK(device.endRendering(*list).has_value());
        CHECK(device.textureBarrier(*list, target, TextureLayout::ColorTarget, TextureLayout::CopySrc, {}).has_value());
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
        REQUIRE(device.wait(*sent, kTestWait).has_value());

        // Row 0 is the top of the texture: red where the triangle is, the
        // clear below it.
        const auto read = device.mapBuffer(readback);
        REQUIRE(read.has_value());
        const auto texel = [&](std::uint32_t x, std::uint32_t y) {
            std::array<std::uint8_t, kTexelBytes> rgba{};
            std::memcpy(rgba.data(), read->data() + (y * bytesPerRow) + (std::uint64_t{x} * kTexelBytes), kTexelBytes);
            return rgba;
        };
        constexpr std::array<std::uint8_t, kTexelBytes> kRed{255, 0, 0, 255};
        constexpr std::array<std::uint8_t, kTexelBytes> kBlack{0, 0, 0, 255};
        CHECK(texel(kSide / 2, 1) == kRed);
        CHECK(texel(kSide / 2, kSide - 1) == kBlack);
        CHECK(texel(0, kSide - 1) == kBlack);
        CHECK(device.destroyPipeline(pipeline).has_value());
        CHECK(device.destroyTexture(target).has_value());
        CHECK(device.destroyBuffer(positions).has_value());
        CHECK(device.destroyBuffer(readback).has_value());
        CHECK(device.destroyCommandPool(*pool).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device blends with the constant color setBlendConstants sets", "[rhi][agnostic]") {
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

        // A triangle over the whole target.
        constexpr std::array<float, std::size_t{kVertices} * kPositionFloats> kWhole{-1, -1, 0,  1, 3, -1,
                                                                                     0,  1,  -1, 3, 0, 1};
        const BufferHandle positions =
            created(device, {.size = sizeof(kWhole), .memory = BufferMemory::Upload, .usage = {.storage = true}});
        const auto written = device.mapBuffer(positions);
        REQUIRE(written.has_value());
        std::memcpy(written->data(), kWhole.data(), sizeof(kWhole));
        const auto slot = storageSlot(device, positions);
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
        const BufferHandle readback = created(device, {.size = bytesPerRow * kSide,
                                                       .memory = BufferMemory::Readback,
                                                       .usage = {.copyDst = true}});
        // The fragment's white times the constant color, plus nothing of the target.
        GraphicsPipelineDesc desc = colorPipeline();
        desc.colorTargets[0].blend = {.enable = true,
                                      .srcColor = BlendFactor::ConstantColor,
                                      .dstColor = BlendFactor::Zero,
                                      .colorOp = BlendOp::Add,
                                      .srcAlpha = BlendFactor::ConstantAlpha,
                                      .dstAlpha = BlendFactor::Zero,
                                      .alphaOp = BlendOp::Add,
                                      .writeMask = BFR_MASK_ALL};
        const PipelineHandle pipeline = created(device, desc);
        const auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists, .pushConstantUpdates = 1});
        REQUIRE(pool.has_value());
        auto list = device.beginCommands(*pool);
        REQUIRE(list.has_value());

        CHECK(
            device.textureBarrier(*list, target, TextureLayout::Undefined, TextureLayout::ColorTarget, {}).has_value());
        RenderingDesc rendering{.colorAttachmentCount = 1,
                                .colorAttachments = {},
                                .depthStencil = {},
                                .occlusionQueries = {}};
        rendering.colorAttachments[0] = {.texture = target,
                                         .loadOp = LoadOp::Clear,
                                         .storeOp = StoreOp::Store,
                                         .clearColor = {0, 0, 0, 0}};
        REQUIRE(device.beginRendering(*list, {}, rendering).has_value());
        viewWhole(device, *list);
        CHECK(device.bindPipeline(*list, pipeline).has_value());
        CHECK(device.setBlendConstants(*list, {0, 1, 0, 1}).has_value());
        CHECK(device.pushConstants(*list, bytesOf({.positions = *slot, .color = {1, 1, 1, 1}})).has_value());
        CHECK(device.draw(*list, kVertices, 1, 0, 0).has_value());
        CHECK(device.endRendering(*list).has_value());
        CHECK(device.textureBarrier(*list, target, TextureLayout::ColorTarget, TextureLayout::CopySrc, {}).has_value());
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
        REQUIRE(sent.has_value());
        REQUIRE(device.wait(*sent, kTestWait).has_value());

        const auto read = device.mapBuffer(readback);
        REQUIRE(read.has_value());
        std::array<std::uint8_t, kTexelBytes> centre{};
        std::memcpy(centre.data(),
                    read->data() + ((kSide / 2) * bytesPerRow) + (std::uint64_t{kSide / 2} * kTexelBytes), kTexelBytes);
        constexpr std::array<std::uint8_t, kTexelBytes> kGreen{0, 255, 0, 255};
        CHECK(centre == kGreen);
        CHECK(device.destroyPipeline(pipeline).has_value());
        CHECK(device.destroyTexture(target).has_value());
        CHECK(device.destroyBuffer(positions).has_value());
        CHECK(device.destroyBuffer(readback).has_value());
        CHECK(device.destroyCommandPool(*pool).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device writes the stencil reference setStencilReference sets", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    if (!takesSpirv(backend)) {
        SKIP("no " << backend.name << " bytecode is checked in");
    }
    const Format format = GENERATE(Format::D24UnormS8Uint, Format::D32FloatS8Uint);
    std::vector<std::string> messages;
    {
        const DrawDevice made{recordingInto(backend, messages)};
        const auto caps = made.instance->formatCaps(0, format);
        REQUIRE(caps.has_value());
        if (!caps->depthStencil || !caps->copySrc) {
            SKIP("the adapter does not render to and copy from " << static_cast<int>(format));
        }
        Device& device = *made.device;
        const Limits& limits = made.instance->adapters()[0].caps.limits;
        constexpr std::uint64_t kStencilBytes = 1;
        const std::uint64_t bytesPerRow = alignUp(std::uint64_t{kSide} * kStencilBytes,
                                                  std::max<std::uint64_t>(limits.copyRowPitchAlignment, 1));

        constexpr std::array<float, std::size_t{kVertices} * kPositionFloats> kWhole{-1, -1, 0,  1, 3, -1,
                                                                                     0,  1,  -1, 3, 0, 1};
        const BufferHandle positions =
            created(device, {.size = sizeof(kWhole), .memory = BufferMemory::Upload, .usage = {.storage = true}});
        const auto written = device.mapBuffer(positions);
        REQUIRE(written.has_value());
        std::memcpy(written->data(), kWhole.data(), sizeof(kWhole));
        const auto slot = storageSlot(device, positions);
        REQUIRE(slot.has_value());
        const TextureDesc texture{.dimension = TextureDimension::D2,
                                  .format = Format::Rgba8Unorm,
                                  .width = kSide,
                                  .height = kSide,
                                  .depth = 1,
                                  .layers = 1,
                                  .mipLevels = 1,
                                  .samples = 1,
                                  .usage = {.colorAttachment = true}};
        const TextureHandle color = created(device, texture);
        TextureDesc depthDesc = texture;
        depthDesc.format = format;
        depthDesc.usage = {.copySrc = true, .depthStencil = true};
        const TextureHandle depth = created(device, depthDesc);
        const BufferHandle readback = created(device, {.size = bytesPerRow * kSide,
                                                       .memory = BufferMemory::Readback,
                                                       .usage = {.copyDst = true}});
        // Every fragment passes and writes the reference.
        GraphicsPipelineDesc desc = colorPipeline();
        const StencilFace replace{.pass = StencilOp::Replace,
                                  .compare = CompareOp::Always,
                                  .readMask = BFR_MASK_ALL,
                                  .writeMask = BFR_MASK_ALL};
        desc.depthStencil = {.stencilTest = true, .front = replace, .back = replace};
        desc.depthStencilFormat = format;
        const PipelineHandle pipeline = created(device, desc);
        const auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists, .pushConstantUpdates = 1});
        REQUIRE(pool.has_value());
        auto list = device.beginCommands(*pool);
        REQUIRE(list.has_value());

        CHECK(
            device.textureBarrier(*list, color, TextureLayout::Undefined, TextureLayout::ColorTarget, {}).has_value());
        CHECK(device.textureBarrier(*list, depth, TextureLayout::Undefined, TextureLayout::DepthStencilTarget, {})
                  .has_value());
        RenderingDesc rendering{.colorAttachmentCount = 1,
                                .colorAttachments = {},
                                .depthStencil = {.texture = depth,
                                                 .depthLoadOp = LoadOp::Clear,
                                                 .depthStoreOp = StoreOp::Store,
                                                 .stencilLoadOp = LoadOp::Clear,
                                                 .stencilStoreOp = StoreOp::Store,
                                                 .clearDepth = 1,
                                                 .clearStencil = 0},
                                .occlusionQueries = {}};
        rendering.colorAttachments[0] = {.texture = color, .loadOp = LoadOp::Clear, .storeOp = StoreOp::Store};
        REQUIRE(device.beginRendering(*list, {}, rendering).has_value());
        viewWhole(device, *list);
        constexpr std::uint32_t kReference = 7;
        CHECK(device.bindPipeline(*list, pipeline).has_value());
        CHECK(device.setStencilReference(*list, kReference).has_value());
        CHECK(device.pushConstants(*list, bytesOf({.positions = *slot, .color = {1, 1, 1, 1}})).has_value());
        CHECK(device.draw(*list, kVertices, 1, 0, 0).has_value());
        CHECK(device.endRendering(*list).has_value());
        CHECK(device.textureBarrier(*list, depth, TextureLayout::DepthStencilTarget, TextureLayout::CopySrc, {})
                  .has_value());
        const BufferTextureCopy stencil{.bufferOffset = 0,
                                        .bytesPerRow = bytesPerRow,
                                        .rowsPerImage = kSide,
                                        .mipLevel = 0,
                                        .layer = 0,
                                        .aspect = TextureAspect::Stencil,
                                        .x = 0,
                                        .y = 0,
                                        .z = 0,
                                        .width = kSide,
                                        .height = kSide,
                                        .depth = 1};
        CHECK(device.copyTextureToBuffer(*list, depth, readback, std::span{&stencil, 1}).has_value());
        const auto sent = device.submit({.queue = 0, .commandLists = std::span{&*list, 1}, .waits = {}});
        REQUIRE(sent.has_value());
        REQUIRE(device.wait(*sent, kTestWait).has_value());

        const auto read = device.mapBuffer(readback);
        REQUIRE(read.has_value());
        const auto centre = static_cast<std::uint8_t>(read->data()[((kSide / 2) * bytesPerRow) + (kSide / 2)]);
        CHECK(centre == kReference);
        CHECK(device.destroyPipeline(pipeline).has_value());
        CHECK(device.destroyTexture(color).has_value());
        CHECK(device.destroyTexture(depth).has_value());
        CHECK(device.destroyBuffer(positions).has_value());
        CHECK(device.destroyBuffer(readback).has_value());
        CHECK(device.destroyCommandPool(*pool).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device clears an integer target to the integers asked for", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    // An unsigned value, and -1, whose bits a signed format reads as int32.
    constexpr std::uint32_t kUnsigned = 7;
    constexpr auto kMinusOne = static_cast<std::uint32_t>(-1);
    const auto [format, bits] = GENERATE_COPY(table<Format, std::uint32_t>({
        {Format::R32Uint, kUnsigned},
        {Format::R32Sint, kMinusOne},
    }));
    std::vector<std::string> messages;
    {
        const DrawDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        const Limits& limits = made.instance->adapters()[0].caps.limits;
        const std::uint64_t bytesPerRow = alignUp(std::uint64_t{kSide} * sizeof(std::uint32_t),
                                                  std::max<std::uint64_t>(limits.copyRowPitchAlignment, 1));
        const TextureHandle target = created(device, {.dimension = TextureDimension::D2,
                                                      .format = format,
                                                      .width = kSide,
                                                      .height = kSide,
                                                      .depth = 1,
                                                      .layers = 1,
                                                      .mipLevels = 1,
                                                      .samples = 1,
                                                      .usage = {.copySrc = true, .colorAttachment = true}});
        const BufferHandle readback = created(device, {.size = bytesPerRow * kSide,
                                                       .memory = BufferMemory::Readback,
                                                       .usage = {.copyDst = true}});
        const auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        auto list = device.beginCommands(*pool);
        REQUIRE(list.has_value());

        CHECK(
            device.textureBarrier(*list, target, TextureLayout::Undefined, TextureLayout::ColorTarget, {}).has_value());
        RenderingDesc rendering{.colorAttachmentCount = 1,
                                .colorAttachments = {},
                                .depthStencil = {},
                                .occlusionQueries = {}};
        rendering.colorAttachments[0] = {.texture = target,
                                         .loadOp = LoadOp::Clear,
                                         .storeOp = StoreOp::Store,
                                         .clearColor = {},
                                         .clearInteger = true,
                                         .clearColorInteger = {bits, 0, 0, 0}};
        const auto begun = device.beginRendering(*list, {}, rendering);
        INFO(failureOf(begun));
        REQUIRE(begun.has_value());
        CHECK(device.endRendering(*list).has_value());
        CHECK(device.textureBarrier(*list, target, TextureLayout::ColorTarget, TextureLayout::CopySrc, {}).has_value());
        const BufferTextureCopy whole{
            .bytesPerRow = bytesPerRow, .rowsPerImage = kSide, .width = kSide, .height = kSide, .depth = 1};
        CHECK(device.copyTextureToBuffer(*list, target, readback, std::span{&whole, 1}).has_value());
        const auto sent = device.submit({.queue = 0, .commandLists = std::span{&*list, 1}, .waits = {}});
        INFO(failureOf(sent));
        REQUIRE(sent.has_value());
        REQUIRE(device.wait(*sent, kTestWait).has_value());

        const auto read = device.mapBuffer(readback);
        REQUIRE(read.has_value());
        std::uint32_t texel = 0;
        std::memcpy(&texel, read->data(), sizeof(texel));
        CHECK(texel == bits);
        CHECK(device.destroyTexture(target).has_value());
        CHECK(device.destroyBuffer(readback).has_value());
        CHECK(device.destroyCommandPool(*pool).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device renders to the mip level and layers an attachment names", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    constexpr std::uint32_t kLayers = 4;
    constexpr std::uint32_t kMips = 2;
    constexpr std::uint32_t kMip = 1;
    constexpr std::uint32_t kFirst = 2;
    constexpr std::uint32_t kCleared = 7;
    std::vector<std::string> messages;
    {
        const DrawDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        const Limits& limits = made.instance->adapters()[0].caps.limits;
        constexpr std::uint32_t kMipSide = kSide >> kMip;
        const std::uint64_t bytesPerRow = alignUp(std::uint64_t{kMipSide} * sizeof(std::uint32_t),
                                                  std::max<std::uint64_t>(limits.copyRowPitchAlignment, 1));
        const std::uint64_t layerBytes = alignUp(bytesPerRow * kMipSide,
                                                 std::max<std::uint64_t>(limits.copyOffsetAlignment, 1));
        const TextureHandle target = created(device, {.dimension = TextureDimension::D2,
                                                      .format = Format::R32Uint,
                                                      .width = kSide,
                                                      .height = kSide,
                                                      .depth = 1,
                                                      .layers = kLayers,
                                                      .mipLevels = kMips,
                                                      .samples = 1,
                                                      .usage = {.copySrc = true, .colorAttachment = true}});
        const BufferHandle readback = created(device, {.size = layerBytes * kLayers,
                                                       .memory = BufferMemory::Readback,
                                                       .usage = {.copyDst = true}});
        const auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        auto list = device.beginCommands(*pool);
        REQUIRE(list.has_value());

        CHECK(
            device.textureBarrier(*list, target, TextureLayout::Undefined, TextureLayout::ColorTarget, {}).has_value());
        // Every layer of the level cleared to 0, then two of them to kCleared.
        for (const auto& [layer, count, value] :
             {std::tuple{0U, kLayers, 0U}, std::tuple{kFirst, kLayers - kFirst, kCleared}}) {
            RenderingDesc rendering{.colorAttachmentCount = 1,
                                    .colorAttachments = {},
                                    .depthStencil = {},
                                    .occlusionQueries = {}};
            rendering.colorAttachments[0] = {.texture = target,
                                             .mipLevel = kMip,
                                             .layer = layer,
                                             .layerCount = count,
                                             .loadOp = LoadOp::Clear,
                                             .storeOp = StoreOp::Store,
                                             .clearColor = {},
                                             .clearInteger = true,
                                             .clearColorInteger = {value, 0, 0, 0}};
            const auto begun = device.beginRendering(*list, {}, rendering);
            INFO(failureOf(begun));
            REQUIRE(begun.has_value());
            CHECK(device.endRendering(*list).has_value());
        }
        CHECK(device.textureBarrier(*list, target, TextureLayout::ColorTarget, TextureLayout::CopySrc, {}).has_value());
        std::array<BufferTextureCopy, kLayers> layers{};
        for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
            layers.at(layer) = {.bufferOffset = layerBytes * layer,
                                .bytesPerRow = bytesPerRow,
                                .rowsPerImage = kMipSide,
                                .mipLevel = kMip,
                                .layer = layer,
                                .width = kMipSide,
                                .height = kMipSide,
                                .depth = 1};
        }
        CHECK(device.copyTextureToBuffer(*list, target, readback, layers).has_value());
        const auto sent = device.submit({.queue = 0, .commandLists = std::span{&*list, 1}, .waits = {}});
        INFO(failureOf(sent));
        REQUIRE(sent.has_value());
        REQUIRE(device.wait(*sent, kTestWait).has_value());

        const auto read = device.mapBuffer(readback);
        REQUIRE(read.has_value());
        for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
            std::uint32_t texel = 0;
            std::memcpy(&texel, read->subspan(layerBytes * layer).data(), sizeof(texel));
            INFO("layer " << layer);
            CHECK(texel == (layer >= kFirst ? kCleared : 0U));
        }
        CHECK(device.destroyTexture(target).has_value());
        CHECK(device.destroyBuffer(readback).has_value());
        CHECK(device.destroyCommandPool(*pool).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device renders to the depth slices of a 3D texture an attachment names, where the adapter can",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    constexpr std::uint32_t kSlices = 4;
    constexpr std::uint32_t kFirst = 2;
    constexpr std::uint32_t kCleared = 7;
    std::vector<std::string> messages;
    {
        const DrawDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        const Caps& caps = made.instance->adapters()[0].caps;
        if (!caps.features.volumeColorTargets) {
            SKIP("the adapter draws to no 3D texture by slice");
        }
        const std::uint64_t bytesPerRow = alignUp(std::uint64_t{kSide} * sizeof(std::uint32_t),
                                                  std::max<std::uint64_t>(caps.limits.copyRowPitchAlignment, 1));
        const std::uint64_t sliceBytes = alignUp(bytesPerRow * kSide,
                                                 std::max<std::uint64_t>(caps.limits.copyOffsetAlignment, 1));
        const TextureHandle target = created(device, {.dimension = TextureDimension::D3,
                                                      .format = Format::R32Uint,
                                                      .width = kSide,
                                                      .height = kSide,
                                                      .depth = kSlices,
                                                      .layers = 1,
                                                      .mipLevels = 1,
                                                      .samples = 1,
                                                      .usage = {.copySrc = true, .colorAttachment = true}});
        const BufferHandle readback = created(device, {.size = sliceBytes * kSlices,
                                                       .memory = BufferMemory::Readback,
                                                       .usage = {.copyDst = true}});
        const auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        auto list = device.beginCommands(*pool);
        REQUIRE(list.has_value());

        CHECK(
            device.textureBarrier(*list, target, TextureLayout::Undefined, TextureLayout::ColorTarget, {}).has_value());
        // Every slice cleared to 0, then two of them to kCleared.
        for (const auto& [slice, count, value] :
             {std::tuple{0U, kSlices, 0U}, std::tuple{kFirst, kSlices - kFirst, kCleared}}) {
            RenderingDesc rendering{.colorAttachmentCount = 1,
                                    .colorAttachments = {},
                                    .depthStencil = {},
                                    .occlusionQueries = {}};
            rendering.colorAttachments[0] = {.texture = target,
                                             .layer = slice,
                                             .layerCount = count,
                                             .loadOp = LoadOp::Clear,
                                             .storeOp = StoreOp::Store,
                                             .clearColor = {},
                                             .clearInteger = true,
                                             .clearColorInteger = {value, 0, 0, 0}};
            const auto begun = device.beginRendering(*list, {}, rendering);
            INFO(failureOf(begun));
            REQUIRE(begun.has_value());
            CHECK(device.endRendering(*list).has_value());
        }
        CHECK(device.textureBarrier(*list, target, TextureLayout::ColorTarget, TextureLayout::CopySrc, {}).has_value());
        std::array<BufferTextureCopy, kSlices> slices{};
        for (std::uint32_t slice = 0; slice < kSlices; ++slice) {
            slices.at(slice) = {.bufferOffset = sliceBytes * slice,
                                .bytesPerRow = bytesPerRow,
                                .rowsPerImage = kSide,
                                .z = slice,
                                .width = kSide,
                                .height = kSide,
                                .depth = 1};
        }
        CHECK(device.copyTextureToBuffer(*list, target, readback, slices).has_value());
        const auto sent = device.submit({.queue = 0, .commandLists = std::span{&*list, 1}, .waits = {}});
        INFO(failureOf(sent));
        REQUIRE(sent.has_value());
        REQUIRE(device.wait(*sent, kTestWait).has_value());

        const auto read = device.mapBuffer(readback);
        REQUIRE(read.has_value());
        for (std::uint32_t slice = 0; slice < kSlices; ++slice) {
            std::uint32_t texel = 0;
            std::memcpy(&texel, read->subspan(sliceBytes * slice).data(), sizeof(texel));
            INFO("slice " << slice);
            CHECK(texel == (slice >= kFirst ? kCleared : 0U));
        }
        CHECK(device.destroyTexture(target).has_value());
        CHECK(device.destroyBuffer(readback).has_value());
        CHECK(device.destroyCommandPool(*pool).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device moves only the mip levels a texture barrier names, so one level renders while another is read",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    constexpr std::uint32_t kMips = 2;
    constexpr std::array<std::uint32_t, kMips> kValues{3, 5};
    std::vector<std::string> messages;
    {
        const DrawDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        const Limits& limits = made.instance->adapters()[0].caps.limits;
        const std::uint64_t bytesPerRow = alignUp(std::uint64_t{kSide} * sizeof(std::uint32_t),
                                                  std::max<std::uint64_t>(limits.copyRowPitchAlignment, 1));
        const std::uint64_t mipBytes = alignUp(bytesPerRow * kSide,
                                               std::max<std::uint64_t>(limits.copyOffsetAlignment, 1));
        const TextureHandle target = created(device, {.dimension = TextureDimension::D2,
                                                      .format = Format::R32Uint,
                                                      .width = kSide,
                                                      .height = kSide,
                                                      .depth = 1,
                                                      .layers = 1,
                                                      .mipLevels = kMips,
                                                      .samples = 1,
                                                      .usage = {.copySrc = true, .colorAttachment = true}});
        const BufferHandle readback =
            created(device, {.size = mipBytes * kMips, .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
        const auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        auto list = device.beginCommands(*pool);
        REQUIRE(list.has_value());

        // Each level is cleared while the one before it waits to be copied.
        std::array<BufferTextureCopy, kMips> levels{};
        for (std::uint32_t mip = 0; mip < kMips; ++mip) {
            const TextureRange level{.mipLevel = mip, .mipCount = 1, .layer = 0, .layerCount = 1};
            CHECK(device.textureBarrier(*list, target, TextureLayout::Undefined, TextureLayout::ColorTarget, level)
                      .has_value());
            RenderingDesc rendering{.colorAttachmentCount = 1,
                                    .colorAttachments = {},
                                    .depthStencil = {},
                                    .occlusionQueries = {}};
            rendering.colorAttachments[0] = {.texture = target,
                                             .mipLevel = mip,
                                             .loadOp = LoadOp::Clear,
                                             .storeOp = StoreOp::Store,
                                             .clearColor = {},
                                             .clearInteger = true,
                                             .clearColorInteger = {kValues.at(mip), 0, 0, 0}};
            const auto begun = device.beginRendering(*list, {}, rendering);
            INFO(failureOf(begun));
            REQUIRE(begun.has_value());
            CHECK(device.endRendering(*list).has_value());
            CHECK(device.textureBarrier(*list, target, TextureLayout::ColorTarget, TextureLayout::CopySrc, level)
                      .has_value());
            const std::uint32_t side = kSide >> mip;
            levels.at(mip) = {.bufferOffset = mipBytes * mip,
                              .bytesPerRow = bytesPerRow,
                              .rowsPerImage = side,
                              .mipLevel = mip,
                              .width = side,
                              .height = side,
                              .depth = 1};
        }
        CHECK(device.copyTextureToBuffer(*list, target, readback, levels).has_value());
        const auto sent = device.submit({.queue = 0, .commandLists = std::span{&*list, 1}, .waits = {}});
        INFO(failureOf(sent));
        REQUIRE(sent.has_value());
        REQUIRE(device.wait(*sent, kTestWait).has_value());

        const auto read = device.mapBuffer(readback);
        REQUIRE(read.has_value());
        for (std::uint32_t mip = 0; mip < kMips; ++mip) {
            std::uint32_t texel = 0;
            std::memcpy(&texel, read->subspan(mipBytes * mip).data(), sizeof(texel));
            INFO("mip " << mip);
            CHECK(texel == kValues.at(mip));
        }
        CHECK(device.destroyTexture(target).has_value());
        CHECK(device.destroyBuffer(readback).has_value());
        CHECK(device.destroyCommandPool(*pool).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device refuses an enum value BFR doesn't have with RhiInvalidCall", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    if (!takesSpirv(backend)) {
        SKIP("no " << backend.name << " bytecode is checked in");
    }
    // A number one past each enum's values, as a C++ caller may cast in.
    constexpr std::uint8_t kPast = 99;
    std::vector<std::string> messages;
    {
        const DrawDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        const TextureDesc textureDesc{.dimension = TextureDimension::D2,
                                      .format = Format::Rgba8Unorm,
                                      .width = kSide,
                                      .height = kSide,
                                      .depth = 1,
                                      .layers = 1,
                                      .mipLevels = 1,
                                      .samples = 1,
                                      .usage = {.copySrc = true, .colorAttachment = true}};
        const TextureHandle target = created(device, textureDesc);
        const BufferHandle indices = created(device, {.size = kSide, .usage = {.copyDst = true, .index = true}});
        const auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        const auto list = device.beginCommands(*pool);
        REQUIRE(list.has_value());

        CHECK(
            device.createBuffer({.size = kSide, .memory = static_cast<BufferMemory>(kPast), .usage = {.copyDst = true}})
                .error()
                .code == ErrorCode::RhiInvalidCall);
        TextureDesc badTexture = textureDesc;
        badTexture.dimension = static_cast<TextureDimension>(kPast);
        CHECK(device.createTexture(badTexture).error().code == ErrorCode::RhiInvalidCall);
        CHECK(device.createSampler({.magFilter = static_cast<Filter>(kPast)}).error().code ==
              ErrorCode::RhiInvalidCall);
        GraphicsPipelineDesc pipeline = colorPipeline();
        pipeline.depthStencil.depthCompare = static_cast<CompareOp>(kPast);
        CHECK(device.createGraphicsPipeline(pipeline).error().code == ErrorCode::RhiInvalidCall);
        pipeline = colorPipeline();
        pipeline.primitiveRestart = static_cast<PrimitiveRestart>(kPast);
        CHECK(device.createGraphicsPipeline(pipeline).error().code == ErrorCode::RhiInvalidCall);
        CHECK(device.setTopology(*list, static_cast<Topology>(kPast)).error().code == ErrorCode::RhiInvalidCall);
        pipeline = colorPipeline();
        pipeline.colorTargets[0].blend.srcColor = static_cast<BlendFactor>(kPast);
        CHECK(device.createGraphicsPipeline(pipeline).error().code == ErrorCode::RhiInvalidCall);
        CHECK(device.textureBarrier(*list, target, static_cast<TextureLayout>(kPast), TextureLayout::ColorTarget, {})
                  .error()
                  .code == ErrorCode::RhiInvalidCall);
        CHECK(device.bindIndexBuffer(*list, indices, 0, static_cast<IndexType>(kPast)).error().code ==
              ErrorCode::RhiInvalidCall);
        RenderingDesc rendering{.colorAttachmentCount = 1,
                                .colorAttachments = {},
                                .depthStencil = {},
                                .occlusionQueries = {}};
        rendering.colorAttachments[0] = {.texture = target, .loadOp = static_cast<LoadOp>(kPast)};
        CHECK(device.beginRendering(*list, {}, rendering).error().code == ErrorCode::RhiInvalidCall);
        const BufferTextureCopy region{.bytesPerRow = std::uint64_t{kSide} * kTexelBytes,
                                       .rowsPerImage = kSide,
                                       .aspect = static_cast<TextureAspect>(kPast),
                                       .width = 1,
                                       .height = 1,
                                       .depth = 1};
        CHECK(device.copyTextureToBuffer(*list, target, indices, std::span{&region, 1}).error().code ==
              ErrorCode::RhiInvalidCall);
        CHECK(device.destroyCommandList(*list).has_value());
        CHECK(device.destroyCommandPool(*pool).has_value());
        CHECK(device.destroyBuffer(indices).has_value());
        CHECK(device.destroyTexture(target).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE(
    "Device takes push constants to the layout's size, and refuses renderings it can't pass on with RhiInvalidCall",
    "[rhi][agnostic]") {
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

        // Push constants are as many bytes as the layout BFR made holds.
        const std::uint32_t pushSize = made.instance->adapters()[0].caps.limits.maxPushConstantsSize;
        const std::vector<std::byte> pushed(pushSize);
        CHECK(device.pushConstants(*list, pushed).has_value());
        for (std::uint32_t update = 1; update < kUpdates; ++update) {
            CHECK(device.pushConstants(*list, bytesOf({})).has_value());
        }

        // A rendering wants an attachment, and a texture where it says one.
        const RenderingDesc none{};
        CHECK(device.beginRendering(*list, {}, none).error().code == ErrorCode::RhiInvalidCall);
        const RenderingDesc unnamed{.colorAttachmentCount = 1,
                                    .colorAttachments = {},
                                    .depthStencil = {},
                                    .occlusionQueries = {}};
        CHECK(device.beginRendering(*list, {}, unnamed).error().code == ErrorCode::RhiInvalidCall);
        RenderingDesc rendering = unnamed;
        rendering.colorAttachments[0].texture = target;
        CHECK(
            device.textureBarrier(*list, target, TextureLayout::Undefined, TextureLayout::ColorTarget, {}).has_value());
        REQUIRE(device.beginRendering(*list, {}, rendering).has_value());
        CHECK(device.setViewport(*list, whole).has_value());
        CHECK(device.setScissor(*list, all).has_value());
        CHECK(device.bindPipeline(*list, graphics).has_value());
        CHECK(device.bindIndexBuffer(*list, indices, 0, IndexType::Uint16).has_value());
        CHECK(device.drawIndexed(*list, 0, 0, 0, 0, 0).has_value());
        CHECK(device.endRendering(*list).has_value());

        CHECK(device.destroyCommandList(*list).has_value());
        CHECK(device.destroyPipeline(graphics).has_value());
        CHECK(device.destroyPipeline(compute).has_value());
        CHECK(device.destroyBuffer(indices).has_value());
        CHECK(device.destroyTexture(target).has_value());
        CHECK(device.destroyCommandPool(*pool).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device draws indirectly the arguments a buffer holds, many at a stride, and as many as a count says",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    if (!takesSpirv(backend)) {
        SKIP("no " << backend.name << " bytecode is checked in");
    }
    // Two draws' arguments, a padded stride apart.
    constexpr std::uint32_t kStride = 32;
    static constexpr std::array kStrides{kStride};
    std::vector<std::string> messages;
    {
        const DrawDevice made{recordingInto(backend, messages),
                              testDeviceWith([](DeviceDesc& desc) { desc.indirectStrides = kStrides; })};
        Device& device = *made.device;
        const Caps& caps = made.instance->adapters()[0].caps;
        // The top half of clip space, then the bottom half.
        constexpr std::array<float, std::size_t{2} * kVertices * kPositionFloats> kHalves{-1, 0, 0, 1, 1,  0,  0, 1,
                                                                                          0,  1, 0, 1, -1, 0,  0, 1,
                                                                                          1,  0, 0, 1, 0,  -1, 0, 1};
        const BufferHandle positions = uploaded(device, kHalves, {.storage = true});
        const auto slot = storageSlot(device, positions);
        REQUIRE(slot.has_value());
        const PipelineHandle pipeline = created(device, colorPipeline());
        const auto draws =
            twoAt<DrawIndirectArgs, kStride>({.vertexCount = kVertices, .instanceCount = 1},
                                             {.vertexCount = kVertices, .instanceCount = 1, .firstVertex = kVertices});
        const auto indexedDraws = twoAt<DrawIndexedIndirectArgs, kStride>({.indexCount = kVertices, .instanceCount = 1},
                                                                          {.indexCount = kVertices,
                                                                           .instanceCount = 1,
                                                                           .firstIndex = kVertices});
        constexpr std::array<std::uint16_t, std::size_t{2} * kVertices> kIndices{0, 1, 2, 3, 4, 5};
        constexpr std::array<std::uint32_t, 1> kOneDraw{1};
        const BufferHandle args = uploaded(device, draws, {.indirect = true});
        const BufferHandle indexedArgs = uploaded(device, indexedDraws, {.indirect = true});
        const BufferHandle indices = uploaded(device, kIndices, {.index = true});
        const BufferHandle count = uploaded(device, kOneDraw, {.indirect = true});

        const bool indexed = GENERATE(false, true);
        const auto bind = [&](CommandListHandle list) {
            CHECK(device.bindPipeline(list, pipeline).has_value());
            CHECK(device.pushConstants(list, bytesOf({.positions = *slot, .color = {1, 0, 0, 1}})).has_value());
            CHECK(device.bindIndexBuffer(list, indices, 0, IndexType::Uint16).has_value());
        };
        const auto drawIndirect = [&](CommandListHandle list, std::uint32_t drawCount) {
            return indexed ? device.drawIndexedIndirect(list, indexedArgs, 0, drawCount, kStride)
                           : device.drawIndirect(list, args, 0, drawCount, kStride);
        };
        const auto drawCounted = [&](CommandListHandle list, std::uint32_t maxDrawCount) {
            return indexed ? device.drawIndexedIndirectCount(list, indexedArgs, 0, count, 0, maxDrawCount, kStride)
                           : device.drawIndirectCount(list, args, 0, count, 0, maxDrawCount, kStride);
        };
        constexpr std::uint32_t kTopRow = 1;
        constexpr std::uint32_t kBottomRow = kSide - 2;

        SECTION("one draw draws the first arguments") {
            const Rendered image{device, caps.limits, [&](CommandListHandle list) {
                                     bind(list);
                                     CHECK(drawIndirect(list, 1).has_value());
                                 }};
            CHECK(image.at(kSide / 2, kTopRow) == kRed);
            CHECK(image.at(kSide / 2, kBottomRow) == kBlack);
        }
        SECTION("two draws read the second arguments a stride on, where the device draws several") {
            if (!caps.features.multiDrawIndirect) {
                SKIP("the adapter draws one indirect draw per call");
            }
            const Rendered image{device, caps.limits, [&](CommandListHandle list) {
                                     bind(list);
                                     CHECK(drawIndirect(list, 2).has_value());
                                 }};
            CHECK(image.at(kSide / 2, kTopRow) == kRed);
            CHECK(image.at(kSide / 2, kBottomRow) == kRed);
        }
        SECTION("a count buffer holding 1 draws one of at most two, where the device reads a count") {
            if (!caps.features.drawIndirectCount) {
                SKIP("the adapter reads no draw count from a buffer");
            }
            const Rendered image{device, caps.limits, [&](CommandListHandle list) {
                                     bind(list);
                                     CHECK(drawCounted(list, 2).has_value());
                                 }};
            CHECK(image.at(kSide / 2, kTopRow) == kRed);
            CHECK(image.at(kSide / 2, kBottomRow) == kBlack);
        }
        for (const BufferHandle buffer : {positions, args, indexedArgs, indices, count}) {
            CHECK(device.destroyBuffer(buffer).has_value());
        }
        CHECK(device.destroyPipeline(pipeline).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device dispatches indirectly the workgroups a buffer holds", "[rhi][agnostic]") {
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
        constexpr std::array kOneGroup{DispatchIndirectArgs{.x = 1, .y = 1, .z = 1}};
        const BufferHandle args = uploaded(device, kOneGroup, {.indirect = true});
        const PipelineHandle pipeline = created(device,
                                                ComputePipelineDesc{.compute = spirv(bfr::test::spirv::kBindlessComp),
                                                                    .workgroupSize = {kWorkgroup, 1, 1}});
        const auto slot = storageSlot(device, target);
        REQUIRE(slot.has_value());
        const auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists, .pushConstantUpdates = 1});
        REQUIRE(pool.has_value());
        auto list = device.beginCommands(*pool);
        REQUIRE(list.has_value());
        CHECK(device.bindPipeline(*list, pipeline).has_value());
        CHECK(device.pushConstants(*list, bytesOf({.positions = *slot})).has_value());
        CHECK(device.dispatchIndirect(*list, args, 0).has_value());
        CHECK(device.barrier(*list).has_value());
        const BufferCopy whole{.srcOffset = 0, .dstOffset = 0, .size = kBytes};
        CHECK(device.copyBuffer(*list, target, readback, std::span{&whole, 1}).has_value());
        const auto sent = device.submit({.queue = 0, .commandLists = std::span{&*list, 1}, .waits = {}});
        INFO(failureOf(sent));
        REQUIRE(sent.has_value());
        REQUIRE(device.wait(*sent, kTestWait).has_value());
        // The shader writes each invocation its own index.
        const auto read = device.mapBuffer(readback);
        REQUIRE(read.has_value());
        std::array<std::uint32_t, kWorkgroup> values{};
        std::memcpy(values.data(), read->data(), kBytes);
        for (std::uint32_t i = 0; i < kWorkgroup; ++i) {
            CHECK(values.at(i) == i);
        }
        CHECK(device.destroyPipeline(pipeline).has_value());
        for (const BufferHandle buffer : {target, readback, args}) {
            CHECK(device.destroyBuffer(buffer).has_value());
        }
        CHECK(device.destroyCommandPool(*pool).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device assembles the exact topology a pipeline or setTopology names, restarting strips where asked",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    if (!takesSpirv(backend)) {
        SKIP("no " << backend.name << " bytecode is checked in");
    }
    std::vector<std::string> messages;
    {
        const DrawDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        const Caps& caps = made.instance->adapters()[0].caps;
        // The corners of clip space: bottom left, bottom right, top left, top
        // right. As a strip they cover it; their first three as a list cover
        // the bottom-left half.
        constexpr std::uint32_t kCorners = 4;
        constexpr std::array<float, std::size_t{kCorners} * kPositionFloats> kQuad{-1, -1, 0, 1, 1, -1, 0, 1,
                                                                                   -1, 1,  0, 1, 1, 1,  0, 1};
        const BufferHandle positions = uploaded(device, kQuad, {.storage = true});
        const auto slot = storageSlot(device, positions);
        REQUIRE(slot.has_value());
        const PipelineHandle list = created(device, colorPipeline());
        GraphicsPipelineDesc stripDesc = colorPipeline();
        stripDesc.topology = Topology::TriangleStrip;
        stripDesc.primitiveRestart = PrimitiveRestart::Index16;
        const PipelineHandle strip = created(device, stripDesc);
        // A strip of one vertex, a restart, and a strip of the first three.
        constexpr std::uint16_t kRestart = 0xFFFF;
        constexpr std::array<std::uint16_t, 5> kRestarted{3, kRestart, 0, 1, 2};
        const BufferHandle indices = uploaded(device, kRestarted, {.index = true});
        // Strips and fans are drawn indexed: lavapipe (Mesa 25.2.8) records
        // nothing at all for a non-indexed one, clear included, with
        // validation silent, where a GPU driver draws it.
        constexpr std::array<std::uint16_t, kCorners> kInOrder{0, 1, 2, 3};
        const BufferHandle inOrder = uploaded(device, kInOrder, {.index = true});
        const auto bind = [&](CommandListHandle commands, PipelineHandle pipeline) {
            CHECK(device.bindPipeline(commands, pipeline).has_value());
            CHECK(device.pushConstants(commands, bytesOf({.positions = *slot, .color = {1, 0, 0, 1}})).has_value());
        };
        constexpr std::uint32_t kLeft = 0;
        constexpr std::uint32_t kRight = kSide - 1;
        constexpr std::uint32_t kTop = 0;
        constexpr std::uint32_t kBottom = kSide - 1;

        SECTION("a list pipeline draws a list") {
            const Rendered image{device, caps.limits, [&](CommandListHandle commands) {
                                     bind(commands, list);
                                     CHECK(device.draw(commands, kCorners, 1, 0, 0).has_value());
                                 }};
            CHECK(image.at(kLeft, kBottom) == kRed);
            CHECK(image.at(kRight, kTop) == kBlack);
        }
        SECTION("setTopology makes a list pipeline's draws a strip, and fails where the device sets none") {
            if (caps.features.dynamicTopology) {
                const Rendered image{
                    device, caps.limits, [&](CommandListHandle commands) {
                        bind(commands, list);
                        CHECK(device.setTopology(commands, Topology::TriangleStrip).has_value());
                        CHECK(device.bindIndexBuffer(commands, inOrder, 0, IndexType::Uint16).has_value());
                        CHECK(device.drawIndexed(commands, kCorners, 1, 0, 0, 0).has_value());
                    }};
                CHECK(image.at(kLeft, kBottom) == kRed);
                CHECK(image.at(kRight, kTop) == kRed);
            }
            else {
                const Rendered image{device, caps.limits, [&](CommandListHandle commands) {
                                         const auto set = device.setTopology(commands, Topology::TriangleStrip);
                                         REQUIRE_FALSE(set.has_value());
                                         CHECK(set.error().code == ErrorCode::RhiUnsupported);
                                     }};
            }
        }
        SECTION("a strip pipeline restarts at an index of all ones") {
            const Rendered image{device, caps.limits, [&](CommandListHandle commands) {
                                     bind(commands, strip);
                                     CHECK(device.bindIndexBuffer(commands, indices, 0, IndexType::Uint16).has_value());
                                     CHECK(device.drawIndexed(commands, kRestarted.size(), 1, 0, 0, 0).has_value());
                                 }};
            CHECK(image.at(kLeft, kBottom) == kRed);
            CHECK(image.at(kRight, kTop) == kBlack);
        }
        SECTION("a fan shares its first vertex, where the device has fans and sets topology") {
            if (!caps.features.triangleFans || !caps.features.dynamicTopology) {
                SKIP("the adapter has no triangle fans, or sets no topology");
            }
            // Bottom left, bottom right, top right, top left: a fan covers it.
            constexpr std::array<float, std::size_t{kCorners} * kPositionFloats> kRound{-1, -1, 0, 1, 1,  -1, 0, 1,
                                                                                        1,  1,  0, 1, -1, 1,  0, 1};
            const BufferHandle round = uploaded(device, kRound, {.storage = true});
            const auto roundSlot = storageSlot(device, round);
            REQUIRE(roundSlot.has_value());
            const Rendered image{device, caps.limits, [&](CommandListHandle commands) {
                                     CHECK(device.bindPipeline(commands, list).has_value());
                                     CHECK(device
                                               .pushConstants(commands,
                                                              bytesOf({.positions = *roundSlot, .color = {1, 0, 0, 1}}))
                                               .has_value());
                                     CHECK(device.setTopology(commands, Topology::TriangleFan).has_value());
                                     CHECK(device.bindIndexBuffer(commands, inOrder, 0, IndexType::Uint16).has_value());
                                     CHECK(device.drawIndexed(commands, kCorners, 1, 0, 0, 0).has_value());
                                 }};
            CHECK(image.at(kLeft, kTop) == kRed);
            CHECK(image.at(kRight, kBottom) == kRed);
            CHECK(device.destroyBuffer(round).has_value());
        }
        CHECK(device.destroyBuffer(positions).has_value());
        CHECK(device.destroyBuffer(indices).has_value());
        CHECK(device.destroyBuffer(inOrder).has_value());
        CHECK(device.destroyPipeline(list).has_value());
        CHECK(device.destroyPipeline(strip).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

namespace {

constexpr std::array<std::uint8_t, kTexelBytes> kGreen{0, 255, 0, 255};
constexpr std::uint32_t kFourSamples = 4;

// Whether the adapter renders format with kFourSamples and resolves it.
bool resolvesFour(const Instance& instance, Format format) {
    const auto caps = instance.formatCaps(0, format);
    REQUIRE(caps.has_value());
    return (caps->sampleCounts & kFourSamples) != 0 && caps->resolve;
}

// One rendering of only a depth-stencil attachment, as attachment says,
// recorded, submitted and waited on, with depth moved from layout before to
// after around it.
void renderDepth(Device& device, const DepthStencilAttachment& attachment, TextureLayout before, TextureLayout after,
                 TextureHandle resolve = {}) {
    const auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
    REQUIRE(pool.has_value());
    auto list = device.beginCommands(*pool);
    REQUIRE(list.has_value());
    CHECK(device.textureBarrier(*list, attachment.texture, before, TextureLayout::DepthStencilTarget, {}).has_value());
    if (!resolve.isNull()) {
        CHECK(device.textureBarrier(*list, resolve, TextureLayout::Undefined, TextureLayout::DepthStencilTarget, {})
                  .has_value());
    }
    const RenderingDesc rendering{.colorAttachmentCount = 0,
                                  .colorAttachments = {},
                                  .depthStencil = attachment,
                                  .occlusionQueries = {}};
    const auto begun = device.beginRendering(*list, {}, rendering);
    INFO(failureOf(begun));
    REQUIRE(begun.has_value());
    CHECK(device.endRendering(*list).has_value());
    CHECK(device.textureBarrier(*list, attachment.texture, TextureLayout::DepthStencilTarget, after, {}).has_value());
    if (!resolve.isNull()) {
        CHECK(device.textureBarrier(*list, resolve, TextureLayout::DepthStencilTarget, TextureLayout::CopySrc, {})
                  .has_value());
    }
    const auto sent = device.submit({.queue = 0, .commandLists = std::span{&*list, 1}, .waits = {}});
    INFO(failureOf(sent));
    REQUIRE(sent.has_value());
    REQUIRE(device.wait(*sent, kTestWait).has_value());
    CHECK(device.destroyCommandPool(*pool).has_value());
}

// The first texel of aspect of texture, in layout CopySrc, as bytes, copied
// as a row of the pitch limits asks for.
std::array<std::byte, kTexelBytes> firstTexel(Device& device, const Limits& limits, TextureHandle texture,
                                              TextureAspect aspect) {
    const std::uint64_t bytesPerRow = alignUp(kTexelBytes, std::max<std::uint64_t>(limits.copyRowPitchAlignment, 1));
    const BufferHandle readback =
        created(device, {.size = bytesPerRow, .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
    const auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
    REQUIRE(pool.has_value());
    auto list = device.beginCommands(*pool);
    REQUIRE(list.has_value());
    const BufferTextureCopy first{.bufferOffset = 0,
                                  .bytesPerRow = bytesPerRow,
                                  .rowsPerImage = 1,
                                  .mipLevel = 0,
                                  .layer = 0,
                                  .aspect = aspect,
                                  .x = 0,
                                  .y = 0,
                                  .z = 0,
                                  .width = 1,
                                  .height = 1,
                                  .depth = 1};
    CHECK(device.copyTextureToBuffer(*list, texture, readback, std::span{&first, 1}).has_value());
    const auto sent = device.submit({.queue = 0, .commandLists = std::span{&*list, 1}, .waits = {}});
    REQUIRE(sent.has_value());
    REQUIRE(device.wait(*sent, kTestWait).has_value());
    const auto read = device.mapBuffer(readback);
    REQUIRE(read.has_value());
    std::array<std::byte, kTexelBytes> bytes{};
    std::memcpy(bytes.data(), read->data(), bytes.size());
    CHECK(device.destroyBuffer(readback).has_value());
    CHECK(device.destroyCommandPool(*pool).has_value());
    return bytes;
}

TextureDesc depthTexture(Format format, std::uint32_t samples, const TextureUsage& usage) {
    return {.dimension = TextureDimension::D2,
            .format = format,
            .width = kSide,
            .height = kSide,
            .depth = 1,
            .layers = 1,
            .mipLevels = 1,
            .samples = samples,
            .usage = usage};
}

} // namespace

TEST_CASE("Device tests depth, and resolves a multisampled attachment as rendering ends or by resolveTexture",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    if (!takesSpirv(backend)) {
        SKIP("no " << backend.name << " bytecode is checked in");
    }
    std::vector<std::string> messages;
    {
        const DrawDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        const Caps& caps = made.instance->adapters()[0].caps;
        // The whole of clip space at depth z.
        const auto whole = [](float z) {
            return std::array<float, std::size_t{kVertices} * kPositionFloats>{-1, -1, z, 1, 3, -1, z, 1, -1, 3, z, 1};
        };
        const BufferHandle near = uploaded(device, whole(0.25F), {.storage = true});
        const BufferHandle far = uploaded(device, whole(0.75F), {.storage = true});
        // Bottom left, bottom right and top right: the bottom-right half, its
        // edge through the pixels on the diagonal.
        constexpr std::array<float, std::size_t{kVertices} * kPositionFloats> kHalf{-1, -1, 0, 1, 1, -1,
                                                                                    0,  1,  1, 1, 0, 1};
        const BufferHandle half = uploaded(device, kHalf, {.storage = true});
        const auto slotOf = [&](BufferHandle buffer) {
            const auto slot = storageSlot(device, buffer);
            REQUIRE(slot.has_value());
            return *slot;
        };
        const auto draw = [&](CommandListHandle commands, PipelineHandle pipeline, BufferHandle positions,
                              const std::array<float, 4>& color) {
            CHECK(device.bindPipeline(commands, pipeline).has_value());
            CHECK(
                device.pushConstants(commands, bytesOf({.positions = slotOf(positions), .color = color})).has_value());
            CHECK(device.draw(commands, kVertices, 1, 0, 0).has_value());
        };
        constexpr std::uint32_t kRight = kSide - 1;
        constexpr std::uint32_t kBottom = kSide - 1;

        SECTION("the nearer of two triangles stays in front, drawn first") {
            const auto depthCaps = made.instance->formatCaps(0, Format::D32Float);
            REQUIRE(depthCaps.has_value());
            if (!depthCaps->depthStencil) {
                SKIP("the adapter does not render to D32Float");
            }
            GraphicsPipelineDesc desc = colorPipeline();
            desc.depthStencil = {.depthTest = true,
                                 .depthWrite = true,
                                 .depthCompare = CompareOp::Less,
                                 .stencilTest = false,
                                 .front = {},
                                 .back = {}};
            desc.depthStencilFormat = Format::D32Float;
            const PipelineHandle tested = created(device, desc);
            const Rendered image{device,
                                 caps.limits,
                                 [&](CommandListHandle commands) {
                                     draw(commands, tested, near, {1, 0, 0, 1});
                                     draw(commands, tested, far, {0, 1, 0, 1});
                                 },
                                 {.depth = Format::D32Float}};
            CHECK(image.at(kSide / 2, kSide / 2) == kRed);
            CHECK(device.destroyPipeline(tested).has_value());
        }
        SECTION("an edge pixel's samples average into its resolved texel") {
            if (!resolvesFour(*made.instance, Format::Rgba8Unorm)) {
                SKIP("the adapter does not resolve Rgba8Unorm from 4 samples");
            }
            GraphicsPipelineDesc desc = colorPipeline();
            desc.samples = kFourSamples;
            const PipelineHandle multisampled = created(device, desc);
            const bool byCommand = GENERATE(false, true);
            INFO((byCommand ? "by resolveTexture" : "as rendering ends"));
            const Rendered image{device,
                                 caps.limits,
                                 [&](CommandListHandle commands) { draw(commands, multisampled, half, {1, 0, 0, 1}); },
                                 {.samples = kFourSamples, .byCommand = byCommand}};
            CHECK(image.at(kRight, kBottom) == kRed);
            CHECK(image.at(0, 0) == kBlack);
            // Partly covered: neither the clear nor the triangle.
            const std::uint8_t edge = image.at(kSide / 2, (kSide / 2) - 1).front();
            CHECK(edge > 0);
            CHECK(edge < kRed.front());
            CHECK(device.destroyPipeline(multisampled).has_value());
        }
        SECTION("a sample mask of none writes nothing, where the device takes one") {
            if (!caps.features.sampleMask) {
                SKIP("the adapter writes every sample");
            }
            GraphicsPipelineDesc desc = colorPipeline();
            desc.sampleMask = 0;
            const PipelineHandle masked = created(device, desc);
            const Rendered image{device, caps.limits, [&](CommandListHandle commands) {
                                     draw(commands, masked, near, {1, 0, 0, 1});
                                 }};
            CHECK(image.at(kSide / 2, kSide / 2) == kBlack);
            CHECK(device.destroyPipeline(masked).has_value());
        }
        CHECK(device.destroyBuffer(near).has_value());
        CHECK(device.destroyBuffer(far).has_value());
        CHECK(device.destroyBuffer(half).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device loads and stores depth and stencil apart, and resolves depth by a mode Caps::resolve has",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const DrawDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        const Caps& caps = made.instance->adapters()[0].caps;

        SECTION("stencil kept while depth is cleared") {
            const Format format = GENERATE(Format::D24UnormS8Uint, Format::D32FloatS8Uint);
            const auto formatCaps = made.instance->formatCaps(0, format);
            REQUIRE(formatCaps.has_value());
            if (!formatCaps->depthStencil || !formatCaps->copySrc) {
                SKIP("the adapter does not render to and copy from " << static_cast<int>(format));
            }
            const TextureHandle depth = created(device,
                                                depthTexture(format, 1, {.copySrc = true, .depthStencil = true}));
            constexpr std::uint32_t kKept = 7;
            renderDepth(device,
                        {.texture = depth,
                         .depthLoadOp = LoadOp::Clear,
                         .stencilLoadOp = LoadOp::Clear,
                         .clearDepth = 1,
                         .clearStencil = kKept},
                        TextureLayout::Undefined, TextureLayout::DepthStencilTarget);
            renderDepth(device,
                        {.texture = depth,
                         .depthLoadOp = LoadOp::Clear,
                         .stencilLoadOp = LoadOp::Load,
                         .clearDepth = 0,
                         .clearStencil = 0},
                        TextureLayout::DepthStencilTarget, TextureLayout::CopySrc);
            CHECK(std::to_integer<std::uint32_t>(
                      firstTexel(device, caps.limits, depth, TextureAspect::Stencil).front()) == kKept);
            CHECK(device.destroyTexture(depth).has_value());
        }
        SECTION("a multisampled depth attachment resolves by sample zero, where the device has it") {
            if ((caps.resolve.depthModes & (1U << std::to_underlying(ResolveMode::SampleZero))) == 0 ||
                !resolvesFour(*made.instance, Format::D32Float)) {
                SKIP("the adapter does not resolve D32Float depth from 4 samples by sample zero");
            }
            const TextureHandle depth = created(device,
                                                depthTexture(Format::D32Float, kFourSamples, {.depthStencil = true}));
            const TextureHandle resolved = created(device, depthTexture(Format::D32Float, 1,
                                                                        {.copySrc = true, .depthStencil = true}));
            constexpr float kCleared = 0.5F;
            renderDepth(device,
                        {.texture = depth,
                         .depthLoadOp = LoadOp::Clear,
                         .depthStoreOp = StoreOp::DontCare,
                         .clearDepth = kCleared,
                         .resolveTexture = resolved,
                         .depthResolve = ResolveMode::SampleZero},
                        TextureLayout::Undefined, TextureLayout::DepthStencilTarget, resolved);
            CHECK(std::bit_cast<float>(firstTexel(device, caps.limits, resolved, TextureAspect::Depth)) == kCleared);
            CHECK(device.destroyTexture(depth).has_value());
            CHECK(device.destroyTexture(resolved).has_value());
        }
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device samples a depth attachment it tests read-only", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    if (!takesSpirv(backend)) {
        SKIP("no " << backend.name << " bytecode is checked in");
    }
    std::vector<std::string> messages;
    {
        const DrawDevice made{recordingInto(backend, messages)};
        const auto formatCaps = made.instance->formatCaps(0, Format::D32Float);
        REQUIRE(formatCaps.has_value());
        if (!formatCaps->sampled || !formatCaps->depthStencil) {
            SKIP("the adapter does not sample and render to D32Float");
        }
        Device& device = *made.device;
        const Caps& caps = made.instance->adapters()[0].caps;
        const TextureHandle depth = created(device,
                                            depthTexture(Format::D32Float, 1, {.sampled = true, .depthStencil = true}));
        constexpr float kCleared = 0.25F;
        renderDepth(device, {.texture = depth, .depthLoadOp = LoadOp::Clear, .clearDepth = kCleared},
                    TextureLayout::Undefined, TextureLayout::DepthStencilRead);
        const auto textureSlot = device.textureSlots(depth);
        REQUIRE(textureSlot.has_value());
        const auto sampler = device.createSampler({});
        REQUIRE(sampler.has_value());
        const auto samplerSlot = device.samplerSlot(*sampler);
        REQUIRE(samplerSlot.has_value());
        // In front of what depth holds, so the test passes.
        constexpr std::array<float, std::size_t{kVertices} * kPositionFloats> kNear{-1, -1, 0,  1, 3, -1,
                                                                                    0,  1,  -1, 3, 0, 1};
        const BufferHandle positions = uploaded(device, kNear, {.storage = true});
        const auto positionSlot = storageSlot(device, positions);
        REQUIRE(positionSlot.has_value());
        GraphicsPipelineDesc desc = colorPipeline();
        desc.fragment = spirv(bfr::test::spirv::kBindlessFrag);
        desc.depthStencil = {.depthTest = true,
                             .depthWrite = false,
                             .depthCompare = CompareOp::Less,
                             .stencilTest = false,
                             .front = {},
                             .back = {}};
        desc.depthStencilFormat = Format::D32Float;
        const PipelineHandle sampling = created(device, desc);
        const Rendered image{device,
                             caps.limits,
                             [&](CommandListHandle commands) {
                                 CHECK(device.bindPipeline(commands, sampling).has_value());
                                 CHECK(device
                                           .pushConstants(commands, bytesOf({.positions = *positionSlot,
                                                                             .textureIndex = textureSlot->sampled,
                                                                             .samplerIndex = *samplerSlot}))
                                           .has_value());
                                 CHECK(device.draw(commands, kVertices, 1, 0, 0).has_value());
                             },
                             {.readDepth = depth}};
        // 0.25 of 255, rounded either way.
        const std::uint8_t red = image.at(kSide / 2, kSide / 2).front();
        CHECK(red >= 63);
        CHECK(red <= 64);
        CHECK(device.destroyPipeline(sampling).has_value());
        CHECK(device.destroySampler(*sampler).has_value());
        CHECK(device.destroyBuffer(positions).has_value());
        CHECK(device.destroyTexture(depth).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device draws with mesh shaders, directly and indirectly, and with a task shader, where it has them",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const DrawDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        const Caps& caps = made.instance->adapters()[0].caps;
        GraphicsPipelineDesc meshDesc = colorPipeline();
        meshDesc.vertex = {};
        meshDesc.mesh = spirv(bfr::test::spirv::kTriangleMesh);
        meshDesc.meshWorkgroupSize = {1, 1, 1};
        if (!caps.features.meshShader) {
            const auto refused = device.createGraphicsPipeline(meshDesc);
            REQUIRE_FALSE(refused.has_value());
            CHECK(refused.error().code == ErrorCode::RhiUnsupported);
            SKIP("the adapter has no mesh shaders");
        }
        if (!takesSpirv(backend)) {
            SKIP("no " << backend.name << " bytecode is checked in");
        }
        // The first three corners of clip space, then the other three: two
        // triangles covering it, one a workgroup.
        constexpr std::uint32_t kTriangleCorners = 6;
        constexpr std::array<float, std::size_t{kTriangleCorners} * kPositionFloats> kHalves{-1, -1, 0,  1, 1, -1,
                                                                                             0,  1,  -1, 1, 0, 1,
                                                                                             1,  -1, 0,  1, 1, 1,
                                                                                             0,  1,  -1, 1, 0, 1};
        const BufferHandle positions = uploaded(device, kHalves, {.storage = true});
        const auto slot = storageSlot(device, positions);
        REQUIRE(slot.has_value());
        const PipelineHandle mesh = created(device, meshDesc);
        // Two workgroups; then, for the count draw, one, and two again, which
        // only a draw count past 1 reaches.
        constexpr std::array kGroups{DispatchIndirectArgs{.x = 2, .y = 1, .z = 1},
                                     DispatchIndirectArgs{.x = 1, .y = 1, .z = 1},
                                     DispatchIndirectArgs{.x = 2, .y = 1, .z = 1}};
        constexpr std::array<std::uint32_t, 1> kOneDraw{1};
        const BufferHandle args = uploaded(device, kGroups, {.indirect = true});
        const BufferHandle count = uploaded(device, kOneDraw, {.indirect = true});
        const auto bind = [&](CommandListHandle commands, PipelineHandle pipeline) {
            CHECK(device.bindPipeline(commands, pipeline).has_value());
            CHECK(device.pushConstants(commands, bytesOf({.positions = *slot, .color = {1, 0, 0, 1}})).has_value());
        };
        constexpr std::uint32_t kLeft = 0;
        constexpr std::uint32_t kRight = kSide - 1;
        constexpr std::uint32_t kTop = 0;
        constexpr std::uint32_t kBottom = kSide - 1;

        SECTION("one workgroup draws the first triangle") {
            const Rendered image{device, caps.limits, [&](CommandListHandle commands) {
                                     bind(commands, mesh);
                                     CHECK(device.drawMesh(commands, 1, 1, 1).has_value());
                                 }};
            CHECK(image.at(kLeft, kBottom) == kRed);
            CHECK(image.at(kRight, kTop) == kBlack);
        }
        SECTION("two workgroups draw both, directly and indirectly") {
            const auto draw = GENERATE(0, 1);
            const Rendered image{
                device, caps.limits, [&](CommandListHandle commands) {
                    bind(commands, mesh);
                    if (draw == 0) {
                        CHECK(device.drawMesh(commands, 2, 1, 1).has_value());
                    }
                    else {
                        CHECK(device.drawMeshIndirect(commands, args, 0, 1, sizeof(DispatchIndirectArgs)).has_value());
                    }
                }};
            INFO("draw " << draw);
            CHECK(image.at(kLeft, kBottom) == kRed);
            CHECK(image.at(kRight, kTop) == kRed);
        }
        SECTION("a count draw draws as many as the count buffer holds, where the device reads a count") {
            if (!caps.features.drawIndirectCount) {
                SKIP("the adapter reads no draw count from a buffer");
            }
            // From the second record: one workgroup, then two, cut at the
            // count of 1.
            constexpr std::uint32_t kMaxDraws = 2;
            const Rendered image{device, caps.limits, [&](CommandListHandle commands) {
                                     bind(commands, mesh);
                                     CHECK(device
                                               .drawMeshIndirectCount(commands, args, sizeof(DispatchIndirectArgs),
                                                                      count, 0, kMaxDraws, sizeof(DispatchIndirectArgs))
                                               .has_value());
                                 }};
            CHECK(image.at(kLeft, kBottom) == kRed);
            CHECK(image.at(kRight, kTop) == kBlack);
        }
        SECTION("a task shader's workgroup launches two mesh workgroups, where the device has task shaders") {
            if (!caps.features.taskShader) {
                SKIP("the adapter has no task shaders");
            }
            GraphicsPipelineDesc taskDesc = meshDesc;
            taskDesc.task = spirv(bfr::test::spirv::kTriangleTask);
            taskDesc.taskWorkgroupSize = {1, 1, 1};
            const PipelineHandle task = created(device, taskDesc);
            const Rendered image{device, caps.limits, [&](CommandListHandle commands) {
                                     bind(commands, task);
                                     CHECK(device.drawMesh(commands, 1, 1, 1).has_value());
                                 }};
            CHECK(image.at(kLeft, kBottom) == kRed);
            CHECK(image.at(kRight, kTop) == kRed);
            CHECK(device.destroyPipeline(task).has_value());
        }
        CHECK(device.destroyBuffer(positions).has_value());
        CHECK(device.destroyBuffer(args).has_value());
        CHECK(device.destroyBuffer(count).has_value());
        CHECK(device.destroyPipeline(mesh).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

namespace {

// The 64-bit value at offset bytes into bytes.
std::uint64_t valueAt(std::span<const std::byte> bytes, std::uint64_t offset) {
    std::uint64_t value = 0;
    std::memcpy(&value, bytes.subspan(offset).data(), sizeof(value));
    return value;
}

// The top half of clip space: y from 0 up to 1, the top rows of the target.
constexpr std::array<float, std::size_t{kVertices} * kPositionFloats> kTopHalf{-1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1};

} // namespace

TEST_CASE("Device counts the samples a draw passes in an occlusion query, exactly where precise, and none when depth "
          "rejects them",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const DrawDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        const Caps& caps = made.instance->adapters()[0].caps;
        constexpr std::uint32_t kQueries = 2;
        if (!caps.features.queries) {
            CHECK(device.createQueryPool({.type = QueryType::Occlusion, .count = kQueries, .statistics = {}})
                      .error()
                      .code == ErrorCode::RhiUnsupported);
            SKIP("the adapter makes no query pools");
        }
        if (!takesSpirv(backend)) {
            SKIP("no " << backend.name << " bytecode is checked in");
        }
        const auto pool = device.createQueryPool({.type = QueryType::Occlusion, .count = kQueries, .statistics = {}});
        INFO(failureOf(pool));
        REQUIRE(pool.has_value());
        const auto layout = device.queryLayout(*pool);
        REQUIRE(layout.has_value());
        REQUIRE(layout->value != kNoOffset);
        REQUIRE(layout->stride >= layout->value + sizeof(std::uint64_t));
        const std::uint64_t resultBytes = std::uint64_t{layout->stride} * kQueries;
        const BufferHandle results =
            created(device, {.size = resultBytes, .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
        const bool precise = caps.features.occlusionQueryPrecise;
        const BufferHandle positions = uploaded(device, kTopHalf, {.storage = true});
        const auto slot = storageSlot(device, positions);
        REQUIRE(slot.has_value());
        const auto depthCaps = made.instance->formatCaps(0, Format::D32Float);
        REQUIRE(depthCaps.has_value());
        if (!depthCaps->depthStencil) {
            SKIP("the adapter does not render to D32Float");
        }
        // Over depth cleared to 0, the triangle at depth 0 passes Always and
        // fails Less.
        GraphicsPipelineDesc depthDesc = colorPipeline();
        depthDesc.depthStencilFormat = Format::D32Float;
        depthDesc.depthStencil = {.depthTest = true,
                                  .depthWrite = false,
                                  .depthCompare = CompareOp::Always,
                                  .stencilTest = false,
                                  .front = {},
                                  .back = {}};
        const PipelineHandle passes = created(device, depthDesc);
        depthDesc.depthStencil.depthCompare = CompareOp::Less;
        const PipelineHandle rejected = created(device, depthDesc);
        const TextureHandle depth = created(device, {.dimension = TextureDimension::D2,
                                                     .format = Format::D32Float,
                                                     .width = kSide,
                                                     .height = kSide,
                                                     .depth = 1,
                                                     .layers = 1,
                                                     .mipLevels = 1,
                                                     .samples = 1,
                                                     .usage = {.depthStencil = true}});
        const std::uint64_t bytesPerRow = alignUp(std::uint64_t{kSide} * kTexelBytes,
                                                  std::max<std::uint64_t>(caps.limits.copyRowPitchAlignment, 1));
        const TextureHandle target = created(device, {.dimension = TextureDimension::D2,
                                                      .format = Format::Rgba8Unorm,
                                                      .width = kSide,
                                                      .height = kSide,
                                                      .depth = 1,
                                                      .layers = 1,
                                                      .mipLevels = 1,
                                                      .samples = 1,
                                                      .usage = {.copySrc = true, .colorAttachment = true}});
        const BufferHandle image = created(device, {.size = bytesPerRow * kSide,
                                                    .memory = BufferMemory::Readback,
                                                    .usage = {.copyDst = true}});
        constexpr std::uint32_t kPushes = 2;
        const auto commands = device.createCommandPool(
            {.queue = 0, .listCount = kPoolLists, .pushConstantUpdates = kPushes});
        REQUIRE(commands.has_value());
        auto list = device.beginCommands(*commands);
        REQUIRE(list.has_value());

        CHECK(device.resetQueries(*list, *pool, 0, kQueries).has_value());
        CHECK(
            device.textureBarrier(*list, target, TextureLayout::Undefined, TextureLayout::ColorTarget, {}).has_value());
        CHECK(device.textureBarrier(*list, depth, TextureLayout::Undefined, TextureLayout::DepthStencilTarget, {})
                  .has_value());
        RenderingDesc rendering{.colorAttachmentCount = 1,
                                .colorAttachments = {},
                                .depthStencil = {.texture = depth,
                                                 .depthLoadOp = LoadOp::Clear,
                                                 .depthStoreOp = StoreOp::DontCare,
                                                 .stencilLoadOp = LoadOp::Clear,
                                                 .stencilStoreOp = StoreOp::DontCare,
                                                 .clearDepth = 0},
                                .occlusionQueries = *pool};
        rendering.colorAttachments[0] = {.texture = target,
                                         .loadOp = LoadOp::Clear,
                                         .storeOp = StoreOp::Store,
                                         .clearColor = {0, 0, 0, 1}};
        REQUIRE(device.beginRendering(*list, {}, rendering).has_value());
        viewWhole(device, *list);
        // Query 0 counts the triangle depth passes; query 1 the same triangle
        // depth rejects.
        for (const auto& [query, pipeline] : {std::pair{0U, passes}, std::pair{1U, rejected}}) {
            CHECK(device.bindPipeline(*list, pipeline).has_value());
            CHECK(device.pushConstants(*list, bytesOf({.positions = *slot, .color = {1, 0, 0, 1}})).has_value());
            CHECK(device.beginQuery(*list, *pool, query, precise).has_value());
            CHECK(device.draw(*list, kVertices, 1, 0, 0).has_value());
            CHECK(device.endQuery(*list, *pool, query).has_value());
        }
        CHECK(device.endRendering(*list).has_value());
        CHECK(device.resolveQueries(*list, *pool, 0, kQueries, results, 0).has_value());
        CHECK(device.textureBarrier(*list, target, TextureLayout::ColorTarget, TextureLayout::CopySrc, {}).has_value());
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
        CHECK(device.copyTextureToBuffer(*list, target, image, std::span{&whole, 1}).has_value());
        const auto sent = device.submit({.queue = 0, .commandLists = std::span{&*list, 1}, .waits = {}});
        INFO(failureOf(sent));
        REQUIRE(sent.has_value());
        REQUIRE(device.wait(*sent, kTestWait).has_value());

        // One sample a pixel: the precise count is the pixels the triangle
        // turned red.
        const auto texels = device.mapBuffer(image);
        REQUIRE(texels.has_value());
        std::uint64_t red = 0;
        for (std::uint32_t y = 0; y < kSide; ++y) {
            for (std::uint32_t x = 0; x < kSide; ++x) {
                std::array<std::uint8_t, kTexelBytes> rgba{};
                std::memcpy(rgba.data(), texels->data() + (y * bytesPerRow) + (std::uint64_t{x} * kTexelBytes),
                            kTexelBytes);
                red += rgba == kRed ? 1U : 0U;
            }
        }
        REQUIRE(red != 0);
        const auto read = device.mapBuffer(results);
        REQUIRE(read.has_value());
        const std::uint64_t drawn = valueAt(*read, layout->value);
        const std::uint64_t none = valueAt(*read, layout->stride + std::uint64_t{layout->value});
        INFO("passed " << drawn << ", rejected " << none << ", red pixels " << red);
        if (precise) {
            CHECK(drawn == red);
        }
        else {
            CHECK(drawn != 0);
        }
        CHECK(none == 0);

        constexpr std::uint8_t kPast = 99;
        CHECK(device.createQueryPool({.type = static_cast<QueryType>(kPast), .count = 1, .statistics = {}})
                  .error()
                  .code == ErrorCode::RhiInvalidCall);
        CHECK(device.destroyCommandPool(*commands).has_value());
        CHECK(device.destroyTexture(target).has_value());
        CHECK(device.destroyBuffer(image).has_value());
        CHECK(device.destroyBuffer(results).has_value());
        CHECK(device.destroyBuffer(positions).has_value());
        CHECK(device.destroyPipeline(passes).has_value());
        CHECK(device.destroyPipeline(rejected).has_value());
        CHECK(device.destroyTexture(depth).has_value());
        CHECK(device.destroyQueryPool(*pool).has_value());
        CHECK(device.destroyQueryPool(*pool).error().code == ErrorCode::RhiInvalidHandle);
        CHECK(device.queryLayout(*pool).error().code == ErrorCode::RhiInvalidHandle);
        CHECK(device.destroyQueryPool({}).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device counts the pipeline statistics a pool selects, a dispatch's and a draw's, where queryLayout says",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const DrawDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        const Caps& caps = made.instance->adapters()[0].caps;
        if (!caps.features.queries || !caps.features.pipelineStatistics) {
            SKIP("the adapter counts no pipeline statistics");
        }
        if (!takesSpirv(backend)) {
            SKIP("no " << backend.name << " bytecode is checked in");
        }
        constexpr std::uint32_t kQueries = 2;
        const auto pool = device.createQueryPool({.type = QueryType::PipelineStatistics,
                                                  .count = kQueries,
                                                  .statistics = {.inputVertices = true,
                                                                 .inputPrimitives = true,
                                                                 .vertexInvocations = true,
                                                                 .computeInvocations = true}});
        INFO(failureOf(pool));
        REQUIRE(pool.has_value());
        const auto layout = device.queryLayout(*pool);
        REQUIRE(layout.has_value());
        CHECK(layout->value == kNoOffset);
        const auto offsetOf = [&](PipelineStatistic statistic) {
            return layout->statistics.at(static_cast<std::size_t>(statistic));
        };
        for (const PipelineStatistic statistic :
             {PipelineStatistic::InputVertices, PipelineStatistic::VertexInvocations,
              PipelineStatistic::ComputeInvocations}) {
            INFO("statistic " << static_cast<int>(statistic));
            REQUIRE(offsetOf(statistic) != kNoOffset);
            REQUIRE(offsetOf(statistic) + sizeof(std::uint64_t) <= layout->stride);
        }
        const std::uint64_t resultBytes = std::uint64_t{layout->stride} * kQueries;
        const BufferHandle results =
            created(device, {.size = resultBytes, .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
        constexpr std::uint64_t kStorageBytes = std::uint64_t{kWorkgroup} * sizeof(std::uint32_t);
        const BufferHandle storage = created(device, {.size = kStorageBytes, .usage = {.storage = true}});
        const auto storageAt = storageSlot(device, storage);
        REQUIRE(storageAt.has_value());
        const PipelineHandle compute = created(device,
                                               ComputePipelineDesc{.compute = spirv(bfr::test::spirv::kBindlessComp),
                                                                   .workgroupSize = {kWorkgroup, 1, 1}});
        const BufferHandle positions = uploaded(device, kTopHalf, {.storage = true});
        const auto positionsAt = storageSlot(device, positions);
        REQUIRE(positionsAt.has_value());
        const PipelineHandle draw = created(device, colorPipeline());
        const TextureHandle target = created(device, {.dimension = TextureDimension::D2,
                                                      .format = Format::Rgba8Unorm,
                                                      .width = kSide,
                                                      .height = kSide,
                                                      .depth = 1,
                                                      .layers = 1,
                                                      .mipLevels = 1,
                                                      .samples = 1,
                                                      .usage = {.colorAttachment = true}});
        constexpr std::uint32_t kPushes = 2;
        const auto commands = device.createCommandPool(
            {.queue = 0, .listCount = kPoolLists, .pushConstantUpdates = kPushes});
        REQUIRE(commands.has_value());
        auto list = device.beginCommands(*commands);
        REQUIRE(list.has_value());

        CHECK(device.resetQueries(*list, *pool, 0, kQueries).has_value());
        // Query 0 counts one workgroup of compute; query 1 one triangle.
        CHECK(device.beginQuery(*list, *pool, 0, false).has_value());
        CHECK(device.bindPipeline(*list, compute).has_value());
        CHECK(device.pushConstants(*list, bytesOf({.positions = *storageAt})).has_value());
        CHECK(device.dispatch(*list, 1, 1, 1).has_value());
        CHECK(device.endQuery(*list, *pool, 0).has_value());
        CHECK(
            device.textureBarrier(*list, target, TextureLayout::Undefined, TextureLayout::ColorTarget, {}).has_value());
        RenderingDesc rendering{.colorAttachmentCount = 1,
                                .colorAttachments = {},
                                .depthStencil = {},
                                .occlusionQueries = {}};
        rendering.colorAttachments[0] = {.texture = target, .loadOp = LoadOp::Clear, .storeOp = StoreOp::Store};
        REQUIRE(device.beginRendering(*list, {}, rendering).has_value());
        viewWhole(device, *list);
        CHECK(device.beginQuery(*list, *pool, 1, false).has_value());
        CHECK(device.bindPipeline(*list, draw).has_value());
        CHECK(device.pushConstants(*list, bytesOf({.positions = *positionsAt, .color = {1, 0, 0, 1}})).has_value());
        CHECK(device.draw(*list, kVertices, 1, 0, 0).has_value());
        CHECK(device.endQuery(*list, *pool, 1).has_value());
        CHECK(device.endRendering(*list).has_value());
        CHECK(device.resolveQueries(*list, *pool, 0, kQueries, results, 0).has_value());
        const auto sent = device.submit({.queue = 0, .commandLists = std::span{&*list, 1}, .waits = {}});
        INFO(failureOf(sent));
        REQUIRE(sent.has_value());
        REQUIRE(device.wait(*sent, kTestWait).has_value());

        const auto read = device.mapBuffer(results);
        REQUIRE(read.has_value());
        const auto statistic = [&](std::uint32_t query, PipelineStatistic which) {
            return valueAt(*read, (std::uint64_t{layout->stride} * query) + offsetOf(which));
        };
        CHECK(statistic(0, PipelineStatistic::ComputeInvocations) == kWorkgroup);
        CHECK(statistic(0, PipelineStatistic::InputVertices) == 0);
        CHECK(statistic(1, PipelineStatistic::InputVertices) == kVertices);
        CHECK(statistic(1, PipelineStatistic::InputPrimitives) == 1);
        CHECK(statistic(1, PipelineStatistic::VertexInvocations) != 0);
        CHECK(statistic(1, PipelineStatistic::ComputeInvocations) == 0);

        CHECK(device.destroyCommandPool(*commands).has_value());
        CHECK(device.destroyTexture(target).has_value());
        CHECK(device.destroyBuffer(results).has_value());
        CHECK(device.destroyBuffer(storage).has_value());
        CHECK(device.destroyBuffer(positions).has_value());
        CHECK(device.destroyPipeline(compute).has_value());
        CHECK(device.destroyPipeline(draw).has_value());
        CHECK(device.destroyQueryPool(*pool).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device counts the primitives and invocations of a mesh draw, where it has mesh statistics",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const DrawDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        const Caps& caps = made.instance->adapters()[0].caps;
        if (!caps.features.queries || !caps.features.meshShader || !caps.features.meshStatistics ||
            !caps.features.pipelineStatistics) {
            SKIP("the adapter counts no mesh work");
        }
        if (!takesSpirv(backend)) {
            SKIP("no " << backend.name << " bytecode is checked in");
        }
        const auto primitives = device.createQueryPool(
            {.type = QueryType::MeshPrimitivesGenerated, .count = 1, .statistics = {}});
        INFO(failureOf(primitives));
        REQUIRE(primitives.has_value());
        const auto invocations = device.createQueryPool(
            {.type = QueryType::PipelineStatistics, .count = 1, .statistics = {.meshInvocations = true}});
        INFO(failureOf(invocations));
        REQUIRE(invocations.has_value());
        const auto primitivesLayout = device.queryLayout(*primitives);
        const auto invocationsLayout = device.queryLayout(*invocations);
        REQUIRE(primitivesLayout.has_value());
        REQUIRE(invocationsLayout.has_value());
        const std::uint32_t meshAt = invocationsLayout->statistics.at(
            static_cast<std::size_t>(PipelineStatistic::MeshInvocations));
        REQUIRE(primitivesLayout->value != kNoOffset);
        REQUIRE(meshAt != kNoOffset);
        const BufferHandle results = created(device, {.size = std::uint64_t{primitivesLayout->stride} +
                                                              invocationsLayout->stride,
                                                      .memory = BufferMemory::Readback,
                                                      .usage = {.copyDst = true}});
        // Two triangles covering clip space, one a workgroup.
        constexpr std::uint32_t kTriangleCorners = 6;
        constexpr std::array<float, std::size_t{kTriangleCorners} * kPositionFloats> kHalves{-1, -1, 0,  1, 1, -1,
                                                                                             0,  1,  -1, 1, 0, 1,
                                                                                             1,  -1, 0,  1, 1, 1,
                                                                                             0,  1,  -1, 1, 0, 1};
        constexpr std::uint32_t kGroups = 2;
        const BufferHandle positions = uploaded(device, kHalves, {.storage = true});
        const auto slot = storageSlot(device, positions);
        REQUIRE(slot.has_value());
        GraphicsPipelineDesc meshDesc = colorPipeline();
        meshDesc.vertex = {};
        meshDesc.mesh = spirv(bfr::test::spirv::kTriangleMesh);
        meshDesc.meshWorkgroupSize = {1, 1, 1};
        const PipelineHandle mesh = created(device, meshDesc);
        const TextureHandle target = created(device, {.dimension = TextureDimension::D2,
                                                      .format = Format::Rgba8Unorm,
                                                      .width = kSide,
                                                      .height = kSide,
                                                      .depth = 1,
                                                      .layers = 1,
                                                      .mipLevels = 1,
                                                      .samples = 1,
                                                      .usage = {.colorAttachment = true}});
        const auto commands = device.createCommandPool({.queue = 0, .listCount = kPoolLists, .pushConstantUpdates = 1});
        REQUIRE(commands.has_value());
        auto list = device.beginCommands(*commands);
        REQUIRE(list.has_value());

        CHECK(device.resetQueries(*list, *primitives, 0, 1).has_value());
        CHECK(device.resetQueries(*list, *invocations, 0, 1).has_value());
        CHECK(
            device.textureBarrier(*list, target, TextureLayout::Undefined, TextureLayout::ColorTarget, {}).has_value());
        RenderingDesc rendering{.colorAttachmentCount = 1,
                                .colorAttachments = {},
                                .depthStencil = {},
                                .occlusionQueries = {}};
        rendering.colorAttachments[0] = {.texture = target, .loadOp = LoadOp::Clear, .storeOp = StoreOp::Store};
        REQUIRE(device.beginRendering(*list, {}, rendering).has_value());
        viewWhole(device, *list);
        CHECK(device.bindPipeline(*list, mesh).has_value());
        CHECK(device.pushConstants(*list, bytesOf({.positions = *slot, .color = {1, 0, 0, 1}})).has_value());
        CHECK(device.beginQuery(*list, *primitives, 0, false).has_value());
        CHECK(device.beginQuery(*list, *invocations, 0, false).has_value());
        CHECK(device.drawMesh(*list, kGroups, 1, 1).has_value());
        CHECK(device.endQuery(*list, *invocations, 0).has_value());
        CHECK(device.endQuery(*list, *primitives, 0).has_value());
        CHECK(device.endRendering(*list).has_value());
        CHECK(device.resolveQueries(*list, *primitives, 0, 1, results, 0).has_value());
        CHECK(device.resolveQueries(*list, *invocations, 0, 1, results, primitivesLayout->stride).has_value());
        const auto sent = device.submit({.queue = 0, .commandLists = std::span{&*list, 1}, .waits = {}});
        INFO(failureOf(sent));
        REQUIRE(sent.has_value());
        REQUIRE(device.wait(*sent, kTestWait).has_value());

        const auto read = device.mapBuffer(results);
        REQUIRE(read.has_value());
        CHECK(valueAt(*read, primitivesLayout->value) == kGroups);
        CHECK(valueAt(*read, std::uint64_t{primitivesLayout->stride} + meshAt) != 0);

        CHECK(device.destroyCommandPool(*commands).has_value());
        CHECK(device.destroyTexture(target).has_value());
        CHECK(device.destroyBuffer(results).has_value());
        CHECK(device.destroyBuffer(positions).has_value());
        CHECK(device.destroyPipeline(mesh).has_value());
        CHECK(device.destroyQueryPool(*primitives).has_value());
        CHECK(device.destroyQueryPool(*invocations).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}
