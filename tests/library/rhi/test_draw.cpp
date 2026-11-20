// What every backend does with draws and dispatches, through the public API
// alone: the slots resources get, a dispatch that writes a buffer, a draw into
// a texture read back, and the usage checks. Each case runs on every backend
// the build has; the shaders are the SPIR-V in tests/library/shaders, so a case
// that runs code skips on a backend with none checked in.

#include "bfr.hpp"
#include "bfr/bfr.h"
#include "bfr_test_support.h"
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
#include <expected>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace bfr;
using namespace bfr::test;

namespace {

constexpr std::uint32_t kWorkgroup = 64;
constexpr std::uint32_t kSide = 8;
constexpr std::uint32_t kTexelBytes = 4;
constexpr std::uint32_t kVertices = 3;
constexpr std::uint32_t kPositionFloats = 4;

// Sets the viewport and scissor to the whole of a kSide by kSide target.
void viewWhole(const Device& device, CommandList list) {
    CHECK(device.setViewport(list, {.x = 0, .y = 0, .width = kSide, .height = kSide, .minDepth = 0, .maxDepth = 1})
              .has_value());
    CHECK(device.setScissor(list, {.x = 0, .y = 0, .width = kSide, .height = kSide}).has_value());
}

// The overloads below would hide the shared ones for buffers and textures.
using bfr::test::created;

Pipeline created(const Device& device, const ComputePipelineDesc& desc) {
    auto pipeline = device.createComputePipeline(desc);
    INFO(failureOf(pipeline));
    REQUIRE(pipeline.has_value());
    return std::move(*pipeline);
}

Pipeline created(const Device& device, const GraphicsPipelineDesc& desc) {
    auto pipeline = device.createGraphicsPipeline(desc);
    INFO(failureOf(pipeline));
    REQUIRE(pipeline.has_value());
    return std::move(*pipeline);
}

// What bfr_get_query_layout says of a query pool handle, which may be stale:
// bfr.hpp names a pool only by a live QueryPool.
Result layoutResult(const Device& device, bfr_query_pool pool) {
    bfr_query_layout layout{};
    return resultOf(bfr_get_query_layout(device.get(), pool, &layout));
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

std::array<std::uint8_t, sizeof(Push)> bytesOf(const Push& push) {
    return std::bit_cast<std::array<std::uint8_t, sizeof(Push)>>(push);
}

// A pipeline drawing color.frag's color into one Rgba8Unorm target, culling
// nothing, so the test's one triangle shows whichever way it winds.
GraphicsPipelineDesc colorPipeline() {
    GraphicsPipelineDesc desc{.vertex = shaderCode(bfr::test::spirv::kBindlessVert),
                              .fragment = shaderCode(bfr::test::spirv::kColorFrag),
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
    // The {} is not redundant: without it, gcc's -Wmissing-field-initializers
    // fails each designated initialiser that leaves readDepth out.
    TextureRef readDepth{}; // NOLINT(readability-redundant-member-init)
};

// What a kSide by kSide Rgba8Unorm target cleared to black holds once record
// has recorded into its rendering, read back row by row.
class Rendered {
public:
    template <class Record>
    Rendered(const Device& device, const CapsLimits& limits, Record record, const Targets& targets = {})
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
        Texture target = created(device, single);
        const bool multisampled = targets.samples > 1;
        TextureDesc samplesDesc = single;
        samplesDesc.samples = targets.samples;
        samplesDesc.usage = {.copySrc = true, .colorAttachment = true};
        Texture samples = multisampled ? created(device, samplesDesc) : Texture{};
        const Texture& drawn = multisampled ? samples : target;
        TextureDesc depthDesc = samplesDesc;
        depthDesc.format = targets.depth;
        depthDesc.usage = {.depthStencil = true};
        Texture depth = targets.depth != Format::Undefined ? created(device, depthDesc) : Texture{};
        Buffer readback =
            created(device, {.size = image_.size(), .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
        // A push for each of a record's draws.
        constexpr std::uint32_t kPushes = 2;
        auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists, .pushConstantUpdates = kPushes});
        REQUIRE(pool.has_value());
        auto list = pool->beginCommands();
        REQUIRE(list.has_value());
        CHECK(
            device.textureBarrier(*list, target, TextureLayout::Undefined, TextureLayout::ColorTarget, {}).has_value());
        if (multisampled) {
            CHECK(device.textureBarrier(*list, drawn, TextureLayout::Undefined, TextureLayout::ColorTarget, {})
                      .has_value());
        }
        if (depth) {
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
        if (targets.readDepth) {
            rendering.depthStencil = {.texture = targets.readDepth, .readOnly = true};
        }
        rendering.colorAttachments[0] = {
            .texture = drawn,
            .loadOp = LoadOp::Clear,
            .storeOp = StoreOp::Store,
            .clearColor = {0, 0, 0, 1},
            .resolveTexture = multisampled && !targets.byCommand ? TextureRef{target} : TextureRef{},
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
        const auto read = readback.map();
        REQUIRE(read.has_value());
        std::memcpy(image_.data(), *read, image_.size());
        CHECK(slotsResult(device, destroyed(target)) == Result::ErrorInvalidHandle);
        if (multisampled) {
            CHECK(slotsResult(device, destroyed(samples)) == Result::ErrorInvalidHandle);
        }
        if (depth) {
            CHECK(slotsResult(device, destroyed(depth)) == Result::ErrorInvalidHandle);
        }
        else {
            // d1: null destroy: no message (track B)
            bfr_destroy_texture(device.get(), bfr_texture{});
        }
        CHECK(slotsResult(device, destroyed(readback)) == Result::ErrorInvalidHandle);
        CHECK(resetResult(device, destroyed(*pool)) == Result::ErrorInvalidHandle);
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
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        constexpr std::uint64_t kBytes = 256;
        Buffer first = created(device, {.size = kBytes, .usage = {.storage = true}});
        Buffer second = created(device, {.size = kBytes, .usage = {.storage = true}});
        Buffer plain = created(device, {.size = kBytes, .usage = {.copyDst = true}});
        const auto firstSlot = storageSlot(first);
        const auto secondSlot = storageSlot(second);
        REQUIRE(firstSlot.has_value());
        REQUIRE(secondSlot.has_value());
        CHECK(*firstSlot != *secondSlot);
        CHECK(*firstSlot < kTestDevice.maxBindlessStorageBuffers);
        CHECK(*secondSlot < kTestDevice.maxBindlessStorageBuffers);
        const auto plainSlots = plain.bufferSlots();
        REQUIRE(plainSlots.has_value());
        CHECK(plainSlots->storage == BFR_NO_SLOT);
        CHECK(plainSlots->uniform == BFR_NO_SLOT);

        const TextureDesc square{.dimension = TextureDimension::D2,
                                 .format = Format::Rgba8Unorm,
                                 .width = kSide,
                                 .height = kSide,
                                 .depth = 1,
                                 .layers = 1,
                                 .mipLevels = 1,
                                 .samples = 1,
                                 .usage = {.sampled = true, .storage = true}};
        Texture both = created(device, square);
        TextureDesc copied = square;
        copied.usage = {.copySrc = true, .copyDst = true};
        Texture neither = created(device, copied);
        const auto bothSlots = both.textureSlots();
        REQUIRE(bothSlots.has_value());
        CHECK(bothSlots->sampled < kTestDevice.maxBindlessTextures);
        CHECK(bothSlots->storage < kTestDevice.maxBindlessStorageTextures);
        const auto neitherSlots = neither.textureSlots();
        REQUIRE(neitherSlots.has_value());
        CHECK(neitherSlots->sampled == BFR_NO_SLOT);
        CHECK(neitherSlots->storage == BFR_NO_SLOT);

        const bfr_buffer goneFirst = first.get();
        for (Buffer* buffer : std::array{&first, &second, &plain}) {
            CHECK(slotsResult(device, destroyed(*buffer)) == Result::ErrorInvalidHandle);
        }
        const bfr_texture goneBoth = destroyed(both);
        CHECK(slotsResult(device, goneBoth) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(neither)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, goneFirst) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, goneBoth) == Result::ErrorInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device gives a depth-stencil texture that shaders sample a slot, and renders to it", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    const Format format = GENERATE(Format::D24UnormS8Uint, Format::D32FloatS8Uint);
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const auto caps = formatCapsOf(made.instance, kFirstAdapter, format);
        INFO(failureOf(caps));
        REQUIRE(caps.has_value());
        if (!caps->sampled || !caps->depthStencil) {
            SKIP("the adapter does not sample and render to " << static_cast<int>(format));
        }
        const Device& device = made.device;
        Texture depth = created(device, {.dimension = TextureDimension::D2,
                                         .format = format,
                                         .width = kSide,
                                         .height = kSide,
                                         .depth = 1,
                                         .layers = 1,
                                         .mipLevels = 1,
                                         .samples = 1,
                                         .usage = {.sampled = true, .depthStencil = true}});
        const auto slots = depth.textureSlots();
        REQUIRE(slots.has_value());
        CHECK(slots->sampled < kTestDevice.maxBindlessTextures);

        auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        auto list = pool->beginCommands();
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
        CHECK(slotsResult(device, destroyed(depth)) == Result::ErrorInvalidHandle);
        CHECK(resetResult(device, destroyed(*pool)) == Result::ErrorInvalidHandle);
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
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        constexpr std::uint64_t kBytes = std::uint64_t{kWorkgroup} * sizeof(std::uint32_t);
        Buffer target = created(device, {.size = kBytes, .usage = {.copySrc = true, .storage = true}});
        Buffer readback = created(device,
                                  {.size = kBytes, .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
        Pipeline pipeline = created(device, ComputePipelineDesc{.compute = shaderCode(bfr::test::spirv::kBindlessComp),
                                                                .workgroupSize = {kWorkgroup, 1, 1}});
        const auto slot = storageSlot(target);
        REQUIRE(slot.has_value());
        auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists, .pushConstantUpdates = 1});
        REQUIRE(pool.has_value());
        auto list = pool->beginCommands();
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
        const auto read = readback.map();
        REQUIRE(read.has_value());
        std::array<std::uint32_t, kWorkgroup> values{};
        std::memcpy(values.data(), *read, kBytes);
        for (std::uint32_t i = 0; i < kWorkgroup; ++i) {
            CHECK(values.at(i) == i);
        }
        CHECK(bindResult(device, 0, destroyed(pipeline)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(target)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(readback)) == Result::ErrorInvalidHandle);
        CHECK(resetResult(device, destroyed(*pool)) == Result::ErrorInvalidHandle);
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
        const Instance instance = std::move(*opened);
        const Caps caps = capsOf(instance, kFirstAdapter);
        if (caps.bindless.maxUniformBuffers == 0) {
            // No table to give slots in.
            const auto refused = instance.createDevice(
                testDeviceWith(instance, kFirstAdapter, kNoWindow,
                               [](DeviceDesc& desc) { desc.maxBindlessUniformBuffers = 1; }));
            CHECK(refused.error().code == Result::ErrorLimitReached);
            SKIP("the adapter has no uniform buffer table");
        }
        auto made = instance.createDevice(testDeviceWith(instance, kFirstAdapter, kNoWindow, [](DeviceDesc& desc) {
            desc.maxBindlessUniformBuffers = kUniforms;
        }));
        INFO(failureOf(made));
        REQUIRE(made.has_value());
        const Device& device = *made;
        Buffer source = created(device, {.size = caps.limits.minUniformBufferAlignment,
                                         .memory = BufferMemory::Upload,
                                         .usage = {.uniform = true}});
        const auto sourceSlots = source.bufferSlots();
        REQUIRE(sourceSlots.has_value());
        CHECK(sourceSlots->uniform < kUniforms);
        CHECK(sourceSlots->storage == BFR_NO_SLOT);
        const auto mapped = source.map();
        REQUIRE(mapped.has_value());
        std::memcpy(*mapped, &kValue, sizeof(kValue));

        if (!takesSpirv(backend)) {
            CHECK(slotsResult(device, destroyed(source)) == Result::ErrorInvalidHandle);
            SKIP("no " << backend.name << " bytecode is checked in");
        }
        Buffer target = created(device, {.size = sizeof(std::uint32_t), .usage = {.copySrc = true, .storage = true}});
        Buffer readback = created(device, {.size = sizeof(std::uint32_t),
                                           .memory = BufferMemory::Readback,
                                           .usage = {.copyDst = true}});
        Pipeline pipeline = created(device, ComputePipelineDesc{.compute = shaderCode(bfr::test::spirv::kUniformComp),
                                                                .workgroupSize = {1, 1, 1}});
        const auto targetSlot = storageSlot(target);
        REQUIRE(targetSlot.has_value());
        auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists, .pushConstantUpdates = 1});
        REQUIRE(pool.has_value());
        auto list = pool->beginCommands();
        REQUIRE(list.has_value());

        CHECK(device.bindPipeline(*list, pipeline).has_value());
        const std::array<std::uint32_t, 2> push{*targetSlot, sourceSlots->uniform};
        CHECK(device.pushConstants(*list, std::bit_cast<std::array<std::uint8_t, sizeof(push)>>(push)).has_value());
        CHECK(device.dispatch(*list, 1, 1, 1).has_value());
        CHECK(device.barrier(*list).has_value());
        const BufferCopy whole{.srcOffset = 0, .dstOffset = 0, .size = sizeof(std::uint32_t)};
        CHECK(device.copyBuffer(*list, target, readback, std::span{&whole, 1}).has_value());
        const auto sent = device.submit({.queue = 0, .commandLists = std::span{&*list, 1}, .waits = {}});
        INFO(failureOf(sent));
        REQUIRE(sent.has_value());
        REQUIRE(device.wait(*sent, kTestWait).has_value());

        const auto read = readback.map();
        REQUIRE(read.has_value());
        std::uint32_t value = 0;
        std::memcpy(&value, *read, sizeof(value));
        CHECK(value == kValue);
        CHECK(bindResult(device, 0, destroyed(pipeline)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(source)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(target)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(readback)) == Result::ErrorInvalidHandle);
        CHECK(resetResult(device, destroyed(*pool)) == Result::ErrorInvalidHandle);
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
        const Caps caps = capsOf(*instance, kFirstAdapter);
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
        auto opened = instance->createDevice(
            testDeviceWith(*instance, kFirstAdapter, kNoWindow, [&queues](DeviceDesc& desc) { desc.queues = queues; }));
        INFO(failureOf(opened));
        REQUIRE(opened.has_value());
        const Device& device = *opened;

        constexpr std::uint64_t kBytes = std::uint64_t{kWorkgroup} * sizeof(std::uint32_t);
        Buffer written = created(device, {.size = kBytes, .usage = {.copySrc = true, .storage = true}});
        Buffer computed = created(device,
                                  {.size = kBytes, .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
        Pipeline pipeline = created(device, ComputePipelineDesc{.compute = shaderCode(bfr::test::spirv::kBindlessComp),
                                                                .workgroupSize = {kWorkgroup, 1, 1}});
        const auto slot = storageSlot(written);
        REQUIRE(slot.has_value());

        const std::uint64_t bytesPerRow = alignUp(std::uint64_t{kSide} * kTexelBytes,
                                                  std::max<std::uint64_t>(caps.limits.copyRowPitchAlignment, 1));
        const std::uint64_t imageBytes = bytesPerRow * kSide;
        Buffer upload = created(device,
                                {.size = imageBytes, .memory = BufferMemory::Upload, .usage = {.copySrc = true}});
        Buffer uploaded = created(device,
                                  {.size = imageBytes, .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
        const auto mappedTexels = upload.map();
        REQUIRE(mappedTexels.has_value());
        const std::span<std::byte> texels = mappedBytes(*mappedTexels, imageBytes);
        for (std::size_t index = 0; index < texels.size(); ++index) {
            texels[index] = static_cast<std::byte>(static_cast<unsigned char>(index));
        }
        auto texture = device.createTexture({.format = Format::Rgba8Unorm,
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

        auto graphicsPool = device.createCommandPool({.queue = kGraphics, .listCount = kPoolLists});
        auto computePool = device.createCommandPool(
            {.queue = kCompute, .listCount = kPoolLists, .pushConstantUpdates = 1});
        auto copyPool = device.createCommandPool({.queue = kCopy, .listCount = kPoolLists});
        REQUIRE(graphicsPool.has_value());
        REQUIRE(computePool.has_value());
        REQUIRE(copyPool.has_value());
        const auto send = [&device](std::uint32_t queue, CommandList list, std::span<const Token> waits) {
            const auto token = device.submit({.queue = queue, .commandLists = std::span{&list, 1}, .waits = waits});
            INFO(failureOf(token));
            REQUIRE(token.has_value());
            return *token;
        };

        // Compute writes the buffer; the copy queue waits on it and reads it back.
        const auto dispatching = computePool->beginCommands();
        REQUIRE(dispatching.has_value());
        CHECK(device.bindPipeline(*dispatching, pipeline).has_value());
        CHECK(device.pushConstants(*dispatching, bytesOf({.positions = *slot})).has_value());
        CHECK(device.dispatch(*dispatching, 1, 1, 1).has_value());
        const Token dispatched = send(kCompute, *dispatching, {});
        const auto reading = copyPool->beginCommands();
        REQUIRE(reading.has_value());
        const BufferCopy all{.srcOffset = 0, .dstOffset = 0, .size = kBytes};
        CHECK(device.copyBuffer(*reading, written, computed, std::span{&all, 1}).has_value());
        const Token read = send(kCopy, *reading, std::span{&dispatched, 1});

        // The copy queue uploads the texture in a copy layout; graphics waits,
        // moves it on and reads it back.
        const auto uploading = copyPool->beginCommands();
        REQUIRE(uploading.has_value());
        CHECK(device.textureBarrier(*uploading, *texture, TextureLayout::Undefined, TextureLayout::CopyDst, {})
                  .has_value());
        CHECK(device.copyBufferToTexture(*uploading, upload, *texture, std::span{&whole, 1}).has_value());
        const Token filled = send(kCopy, *uploading, std::span{&read, 1});
        const auto consuming = graphicsPool->beginCommands();
        REQUIRE(consuming.has_value());
        CHECK(device.textureBarrier(*consuming, *texture, TextureLayout::CopyDst, TextureLayout::CopySrc, {})
                  .has_value());
        CHECK(device.copyTextureToBuffer(*consuming, *texture, uploaded, std::span{&whole, 1}).has_value());
        const Token used = send(kGraphics, *consuming, std::span{&filled, 1});
        REQUIRE(device.wait(used, kTestWait).has_value());
        CHECK(device.poll(read).value_or(false));

        // The shader writes each invocation its own index.
        const auto values = computed.map();
        REQUIRE(values.has_value());
        std::array<std::uint32_t, kWorkgroup> indices{};
        std::memcpy(indices.data(), *values, kBytes);
        for (std::uint32_t i = 0; i < kWorkgroup; ++i) {
            CHECK(indices.at(i) == i);
        }
        const auto mappedBack = uploaded.map();
        REQUIRE(mappedBack.has_value());
        const std::span<const std::byte> back = mappedBytes(*mappedBack, imageBytes);
        const std::uint64_t rowBytes = std::uint64_t{kSide} * kTexelBytes;
        for (std::uint64_t row = 0; row < kSide; ++row) {
            INFO("row " << row);
            CHECK(std::ranges::equal(back.subspan(row * bytesPerRow, rowBytes),
                                     texels.subspan(row * bytesPerRow, rowBytes)));
        }
        for (CommandPool* pool : std::array{&*graphicsPool, &*computePool, &*copyPool}) {
            CHECK(resetResult(device, destroyed(*pool)) == Result::ErrorInvalidHandle);
        }
        CHECK(bindResult(device, kGraphics, destroyed(pipeline)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(*texture)) == Result::ErrorInvalidHandle);
        for (Buffer* buffer : std::array{&written, &computed, &upload, &uploaded}) {
            CHECK(slotsResult(device, destroyed(*buffer)) == Result::ErrorInvalidHandle);
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
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        const CapsLimits limits = capsOf(made.instance, kFirstAdapter).limits;
        const std::uint64_t bytesPerRow = alignUp(std::uint64_t{kSide} * kTexelBytes,
                                                  std::max<std::uint64_t>(limits.copyRowPitchAlignment, 1));
        const std::uint64_t imageBytes = bytesPerRow * kSide;

        // The top half of clip space: y from 0 up to 1, which lands in the
        // top rows of the texture on every backend.
        constexpr std::array<float, std::size_t{kVertices} * kPositionFloats> kTopHalf{-1, 0, 0, 1, 1, 0,
                                                                                       0,  1, 0, 1, 0, 1};
        Buffer positions =
            created(device, {.size = sizeof(kTopHalf), .memory = BufferMemory::Upload, .usage = {.storage = true}});
        const auto written = positions.map();
        REQUIRE(written.has_value());
        std::memcpy(*written, kTopHalf.data(), sizeof(kTopHalf));
        const auto slot = storageSlot(positions);
        REQUIRE(slot.has_value());

        Texture target = created(device, {.dimension = TextureDimension::D2,
                                          .format = Format::Rgba8Unorm,
                                          .width = kSide,
                                          .height = kSide,
                                          .depth = 1,
                                          .layers = 1,
                                          .mipLevels = 1,
                                          .samples = 1,
                                          .usage = {.copySrc = true, .colorAttachment = true}});
        Buffer readback = created(device,
                                  {.size = imageBytes, .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
        Pipeline pipeline = created(device, colorPipeline());
        auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists, .pushConstantUpdates = 1});
        REQUIRE(pool.has_value());
        auto list = pool->beginCommands();
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
        const auto mapped = readback.map();
        REQUIRE(mapped.has_value());
        const std::span<const std::byte> read = mappedBytes(*mapped, imageBytes);
        const auto texel = [&](std::uint32_t x, std::uint32_t y) {
            std::array<std::uint8_t, kTexelBytes> rgba{};
            std::memcpy(rgba.data(), read.subspan((y * bytesPerRow) + (std::uint64_t{x} * kTexelBytes)).data(),
                        kTexelBytes);
            return rgba;
        };
        constexpr std::array<std::uint8_t, kTexelBytes> kRed{255, 0, 0, 255};
        constexpr std::array<std::uint8_t, kTexelBytes> kBlack{0, 0, 0, 255};
        CHECK(texel(kSide / 2, 1) == kRed);
        CHECK(texel(kSide / 2, kSide - 1) == kBlack);
        CHECK(texel(0, kSide - 1) == kBlack);
        CHECK(bindResult(device, 0, destroyed(pipeline)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(target)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(positions)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(readback)) == Result::ErrorInvalidHandle);
        CHECK(resetResult(device, destroyed(*pool)) == Result::ErrorInvalidHandle);
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
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        const CapsLimits limits = capsOf(made.instance, kFirstAdapter).limits;
        const std::uint64_t bytesPerRow = alignUp(std::uint64_t{kSide} * kTexelBytes,
                                                  std::max<std::uint64_t>(limits.copyRowPitchAlignment, 1));

        // A triangle over the whole target.
        constexpr std::array<float, std::size_t{kVertices} * kPositionFloats> kWhole{-1, -1, 0,  1, 3, -1,
                                                                                     0,  1,  -1, 3, 0, 1};
        Buffer positions =
            created(device, {.size = sizeof(kWhole), .memory = BufferMemory::Upload, .usage = {.storage = true}});
        const auto written = positions.map();
        REQUIRE(written.has_value());
        std::memcpy(*written, kWhole.data(), sizeof(kWhole));
        const auto slot = storageSlot(positions);
        REQUIRE(slot.has_value());
        Texture target = created(device, {.dimension = TextureDimension::D2,
                                          .format = Format::Rgba8Unorm,
                                          .width = kSide,
                                          .height = kSide,
                                          .depth = 1,
                                          .layers = 1,
                                          .mipLevels = 1,
                                          .samples = 1,
                                          .usage = {.copySrc = true, .colorAttachment = true}});
        Buffer readback = created(device, {.size = bytesPerRow * kSide,
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
        Pipeline pipeline = created(device, desc);
        auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists, .pushConstantUpdates = 1});
        REQUIRE(pool.has_value());
        auto list = pool->beginCommands();
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
        CHECK(device.setBlendConstants(*list, 0, 1, 0, 1).has_value());
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

        const auto mapped = readback.map();
        REQUIRE(mapped.has_value());
        const std::span<const std::byte> read = mappedBytes(*mapped, bytesPerRow * kSide);
        std::array<std::uint8_t, kTexelBytes> centre{};
        std::memcpy(centre.data(),
                    read.subspan(((kSide / 2) * bytesPerRow) + (std::uint64_t{kSide / 2} * kTexelBytes)).data(),
                    kTexelBytes);
        constexpr std::array<std::uint8_t, kTexelBytes> kGreen{0, 255, 0, 255};
        CHECK(centre == kGreen);
        CHECK(bindResult(device, 0, destroyed(pipeline)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(target)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(positions)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(readback)) == Result::ErrorInvalidHandle);
        CHECK(resetResult(device, destroyed(*pool)) == Result::ErrorInvalidHandle);
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
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const auto caps = formatCapsOf(made.instance, kFirstAdapter, format);
        INFO(failureOf(caps));
        REQUIRE(caps.has_value());
        if (!caps->depthStencil || !caps->copySrc) {
            SKIP("the adapter does not render to and copy from " << static_cast<int>(format));
        }
        const Device& device = made.device;
        const CapsLimits limits = capsOf(made.instance, kFirstAdapter).limits;
        constexpr std::uint64_t kStencilBytes = 1;
        const std::uint64_t bytesPerRow = alignUp(std::uint64_t{kSide} * kStencilBytes,
                                                  std::max<std::uint64_t>(limits.copyRowPitchAlignment, 1));

        constexpr std::array<float, std::size_t{kVertices} * kPositionFloats> kWhole{-1, -1, 0,  1, 3, -1,
                                                                                     0,  1,  -1, 3, 0, 1};
        Buffer positions =
            created(device, {.size = sizeof(kWhole), .memory = BufferMemory::Upload, .usage = {.storage = true}});
        const auto written = positions.map();
        REQUIRE(written.has_value());
        std::memcpy(*written, kWhole.data(), sizeof(kWhole));
        const auto slot = storageSlot(positions);
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
        Texture color = created(device, texture);
        TextureDesc depthDesc = texture;
        depthDesc.format = format;
        depthDesc.usage = {.copySrc = true, .depthStencil = true};
        Texture depth = created(device, depthDesc);
        Buffer readback = created(device, {.size = bytesPerRow * kSide,
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
        Pipeline pipeline = created(device, desc);
        auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists, .pushConstantUpdates = 1});
        REQUIRE(pool.has_value());
        auto list = pool->beginCommands();
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

        const auto read = readback.map();
        REQUIRE(read.has_value());
        const auto centre = static_cast<std::uint8_t>(
            mappedBytes(*read, bytesPerRow * kSide)[((kSide / 2) * bytesPerRow) + (kSide / 2)]);
        CHECK(centre == kReference);
        CHECK(bindResult(device, 0, destroyed(pipeline)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(color)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(depth)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(positions)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(readback)) == Result::ErrorInvalidHandle);
        CHECK(resetResult(device, destroyed(*pool)) == Result::ErrorInvalidHandle);
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
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        const CapsLimits limits = capsOf(made.instance, kFirstAdapter).limits;
        const std::uint64_t bytesPerRow = alignUp(std::uint64_t{kSide} * sizeof(std::uint32_t),
                                                  std::max<std::uint64_t>(limits.copyRowPitchAlignment, 1));
        Texture target = created(device, {.dimension = TextureDimension::D2,
                                          .format = format,
                                          .width = kSide,
                                          .height = kSide,
                                          .depth = 1,
                                          .layers = 1,
                                          .mipLevels = 1,
                                          .samples = 1,
                                          .usage = {.copySrc = true, .colorAttachment = true}});
        Buffer readback = created(device, {.size = bytesPerRow * kSide,
                                           .memory = BufferMemory::Readback,
                                           .usage = {.copyDst = true}});
        auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        auto list = pool->beginCommands();
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

        const auto read = readback.map();
        REQUIRE(read.has_value());
        std::uint32_t texel = 0;
        std::memcpy(&texel, *read, sizeof(texel));
        CHECK(texel == bits);
        CHECK(slotsResult(device, destroyed(target)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(readback)) == Result::ErrorInvalidHandle);
        CHECK(resetResult(device, destroyed(*pool)) == Result::ErrorInvalidHandle);
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
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        const CapsLimits limits = capsOf(made.instance, kFirstAdapter).limits;
        constexpr std::uint32_t kMipSide = kSide >> kMip;
        const std::uint64_t bytesPerRow = alignUp(std::uint64_t{kMipSide} * sizeof(std::uint32_t),
                                                  std::max<std::uint64_t>(limits.copyRowPitchAlignment, 1));
        const std::uint64_t layerBytes = alignUp(bytesPerRow * kMipSide,
                                                 std::max<std::uint64_t>(limits.copyOffsetAlignment, 1));
        Texture target = created(device, {.dimension = TextureDimension::D2,
                                          .format = Format::R32Uint,
                                          .width = kSide,
                                          .height = kSide,
                                          .depth = 1,
                                          .layers = kLayers,
                                          .mipLevels = kMips,
                                          .samples = 1,
                                          .usage = {.copySrc = true, .colorAttachment = true}});
        Buffer readback = created(device, {.size = layerBytes * kLayers,
                                           .memory = BufferMemory::Readback,
                                           .usage = {.copyDst = true}});
        auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        auto list = pool->beginCommands();
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

        const auto mapped = readback.map();
        REQUIRE(mapped.has_value());
        const std::span<const std::byte> read = mappedBytes(*mapped, layerBytes * kLayers);
        for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
            std::uint32_t texel = 0;
            std::memcpy(&texel, read.subspan(layerBytes * layer).data(), sizeof(texel));
            INFO("layer " << layer);
            CHECK(texel == (layer >= kFirst ? kCleared : 0U));
        }
        CHECK(slotsResult(device, destroyed(target)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(readback)) == Result::ErrorInvalidHandle);
        CHECK(resetResult(device, destroyed(*pool)) == Result::ErrorInvalidHandle);
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
        const FirstAdapterDevice made{recordingInto(backend, messages), [](DeviceDesc& desc, const Caps& caps) {
                                          desc.features.volumeColorTargets = caps.features.volumeColorTargets;
                                      }};
        const Device& device = made.device;
        const Caps caps = capsOf(made.instance, kFirstAdapter);
        if (!caps.features.volumeColorTargets) {
            SKIP("the adapter draws to no 3D texture by slice");
        }
        const std::uint64_t bytesPerRow = alignUp(std::uint64_t{kSide} * sizeof(std::uint32_t),
                                                  std::max<std::uint64_t>(caps.limits.copyRowPitchAlignment, 1));
        const std::uint64_t sliceBytes = alignUp(bytesPerRow * kSide,
                                                 std::max<std::uint64_t>(caps.limits.copyOffsetAlignment, 1));
        Texture target = created(device, {.dimension = TextureDimension::D3,
                                          .format = Format::R32Uint,
                                          .width = kSide,
                                          .height = kSide,
                                          .depth = kSlices,
                                          .layers = 1,
                                          .mipLevels = 1,
                                          .samples = 1,
                                          .usage = {.copySrc = true, .colorAttachment = true}});
        Buffer readback = created(device, {.size = sliceBytes * kSlices,
                                           .memory = BufferMemory::Readback,
                                           .usage = {.copyDst = true}});
        auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        auto list = pool->beginCommands();
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

        const auto mapped = readback.map();
        REQUIRE(mapped.has_value());
        const std::span<const std::byte> read = mappedBytes(*mapped, sliceBytes * kSlices);
        for (std::uint32_t slice = 0; slice < kSlices; ++slice) {
            std::uint32_t texel = 0;
            std::memcpy(&texel, read.subspan(sliceBytes * slice).data(), sizeof(texel));
            INFO("slice " << slice);
            CHECK(texel == (slice >= kFirst ? kCleared : 0U));
        }
        CHECK(slotsResult(device, destroyed(target)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(readback)) == Result::ErrorInvalidHandle);
        CHECK(resetResult(device, destroyed(*pool)) == Result::ErrorInvalidHandle);
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
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        const CapsLimits limits = capsOf(made.instance, kFirstAdapter).limits;
        const std::uint64_t bytesPerRow = alignUp(std::uint64_t{kSide} * sizeof(std::uint32_t),
                                                  std::max<std::uint64_t>(limits.copyRowPitchAlignment, 1));
        const std::uint64_t mipBytes = alignUp(bytesPerRow * kSide,
                                               std::max<std::uint64_t>(limits.copyOffsetAlignment, 1));
        Texture target = created(device, {.dimension = TextureDimension::D2,
                                          .format = Format::R32Uint,
                                          .width = kSide,
                                          .height = kSide,
                                          .depth = 1,
                                          .layers = 1,
                                          .mipLevels = kMips,
                                          .samples = 1,
                                          .usage = {.copySrc = true, .colorAttachment = true}});
        Buffer readback =
            created(device, {.size = mipBytes * kMips, .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
        auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        auto list = pool->beginCommands();
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

        const auto mapped = readback.map();
        REQUIRE(mapped.has_value());
        const std::span<const std::byte> read = mappedBytes(*mapped, mipBytes * kMips);
        for (std::uint32_t mip = 0; mip < kMips; ++mip) {
            std::uint32_t texel = 0;
            std::memcpy(&texel, read.subspan(mipBytes * mip).data(), sizeof(texel));
            INFO("mip " << mip);
            CHECK(texel == kValues.at(mip));
        }
        CHECK(slotsResult(device, destroyed(target)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(readback)) == Result::ErrorInvalidHandle);
        CHECK(resetResult(device, destroyed(*pool)) == Result::ErrorInvalidHandle);
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
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        const TextureDesc textureDesc{.dimension = TextureDimension::D2,
                                      .format = Format::Rgba8Unorm,
                                      .width = kSide,
                                      .height = kSide,
                                      .depth = 1,
                                      .layers = 1,
                                      .mipLevels = 1,
                                      .samples = 1,
                                      .usage = {.copySrc = true, .colorAttachment = true}};
        Texture target = created(device, textureDesc);
        Buffer indices = created(device, {.size = kSide, .usage = {.copyDst = true, .index = true}});
        auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        const auto list = pool->beginCommands();
        REQUIRE(list.has_value());

        CHECK(
            device.createBuffer({.size = kSide, .memory = static_cast<BufferMemory>(kPast), .usage = {.copyDst = true}})
                .error()
                .code == Result::ErrorInvalidArgument);
        TextureDesc badTexture = textureDesc;
        badTexture.dimension = static_cast<TextureDimension>(kPast);
        CHECK(device.createTexture(badTexture).error().code == Result::ErrorInvalidArgument);
        CHECK(device.createSampler({.magFilter = static_cast<Filter>(kPast)}).error().code ==
              Result::ErrorInvalidArgument);
        GraphicsPipelineDesc pipeline = colorPipeline();
        pipeline.depthStencil.depthCompare = static_cast<CompareOp>(kPast);
        CHECK(device.createGraphicsPipeline(pipeline).error().code == Result::ErrorInvalidArgument);
        pipeline = colorPipeline();
        pipeline.primitiveRestart = static_cast<PrimitiveRestart>(kPast);
        CHECK(device.createGraphicsPipeline(pipeline).error().code == Result::ErrorInvalidArgument);
        CHECK(device.setTopology(*list, static_cast<Topology>(kPast)).error().code == Result::ErrorInvalidArgument);
        pipeline = colorPipeline();
        pipeline.colorTargets[0].blend.srcColor = static_cast<BlendFactor>(kPast);
        CHECK(device.createGraphicsPipeline(pipeline).error().code == Result::ErrorInvalidArgument);
        CHECK(device.textureBarrier(*list, target, static_cast<TextureLayout>(kPast), TextureLayout::ColorTarget, {})
                  .error()
                  .code == Result::ErrorInvalidArgument);
        CHECK(device.bindIndexBuffer(*list, indices, 0, static_cast<IndexType>(kPast)).error().code ==
              Result::ErrorInvalidArgument);
        RenderingDesc rendering{.colorAttachmentCount = 1,
                                .colorAttachments = {},
                                .depthStencil = {},
                                .occlusionQueries = {}};
        rendering.colorAttachments[0] = {.texture = target, .loadOp = static_cast<LoadOp>(kPast)};
        CHECK(device.beginRendering(*list, {}, rendering).error().code == Result::ErrorInvalidArgument);
        const BufferTextureCopy region{.bytesPerRow = std::uint64_t{kSide} * kTexelBytes,
                                       .rowsPerImage = kSide,
                                       .aspect = static_cast<TextureAspect>(kPast),
                                       .width = 1,
                                       .height = 1,
                                       .depth = 1};
        CHECK(device.copyTextureToBuffer(*list, target, indices, std::span{&region, 1}).error().code ==
              Result::ErrorInvalidArgument);
        device.destroyCommandList(*list);
        CHECK(device.barrier(*list).error().code == Result::ErrorInvalidHandle);
        CHECK(resetResult(device, destroyed(*pool)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(indices)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(target)) == Result::ErrorInvalidHandle);
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
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        Texture target = created(device, {.dimension = TextureDimension::D2,
                                          .format = Format::Rgba8Unorm,
                                          .width = kSide,
                                          .height = kSide,
                                          .depth = 1,
                                          .layers = 1,
                                          .mipLevels = 1,
                                          .samples = 1,
                                          .usage = {.colorAttachment = true}});
        Buffer indices = created(device, {.size = kSide, .usage = {.index = true}});
        Pipeline graphics = created(device, colorPipeline());
        Pipeline compute = created(device, ComputePipelineDesc{.compute = shaderCode(bfr::test::spirv::kBindlessComp),
                                                               .workgroupSize = {kWorkgroup, 1, 1}});
        constexpr std::uint32_t kUpdates = 2;
        auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists, .pushConstantUpdates = kUpdates});
        REQUIRE(pool.has_value());
        auto list = pool->beginCommands();
        REQUIRE(list.has_value());
        const Viewport whole{.x = 0, .y = 0, .width = kSide, .height = kSide, .minDepth = 0, .maxDepth = 1};
        const Rect all{.x = 0, .y = 0, .width = kSide, .height = kSide};

        // Push constants are as many bytes as the layout BFR made holds.
        const std::uint32_t pushSize = capsOf(made.instance, kFirstAdapter).limits.maxPushConstantsSize;
        const std::vector<std::uint8_t> pushed(pushSize);
        CHECK(device.pushConstants(*list, pushed).has_value());
        for (std::uint32_t update = 1; update < kUpdates; ++update) {
            CHECK(device.pushConstants(*list, bytesOf({})).has_value());
        }

        // A rendering wants an attachment, and a texture where it says one.
        const RenderingDesc none{};
        CHECK(device.beginRendering(*list, {}, none).error().code == Result::ErrorInvalidArgument);
        const RenderingDesc unnamed{.colorAttachmentCount = 1,
                                    .colorAttachments = {},
                                    .depthStencil = {},
                                    .occlusionQueries = {}};
        CHECK(device.beginRendering(*list, {}, unnamed).error().code == Result::ErrorInvalidArgument);
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

        device.destroyCommandList(*list);
        CHECK(device.barrier(*list).error().code == Result::ErrorInvalidHandle);
        CHECK(bindResult(device, 0, destroyed(graphics)) == Result::ErrorInvalidHandle);
        CHECK(bindResult(device, 0, destroyed(compute)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(indices)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(target)) == Result::ErrorInvalidHandle);
        CHECK(resetResult(device, destroyed(*pool)) == Result::ErrorInvalidHandle);
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
        const FirstAdapterDevice made{recordingInto(backend, messages), [](DeviceDesc& desc, const Caps& caps) {
                                          desc.features.multiDrawIndirect = caps.features.multiDrawIndirect;
                                          desc.features.drawIndirectCount = caps.features.drawIndirectCount;
                                          desc.features.drawIndirectFirstInstance =
                                              caps.features.drawIndirectFirstInstance;
                                          desc.indirectStrides = kStrides;
                                      }};
        const Device& device = made.device;
        const Caps caps = capsOf(made.instance, kFirstAdapter);
        // The top half of clip space, then the bottom half.
        constexpr std::array<float, std::size_t{2} * kVertices * kPositionFloats> kHalves{-1, 0, 0, 1, 1,  0,  0, 1,
                                                                                          0,  1, 0, 1, -1, 0,  0, 1,
                                                                                          1,  0, 0, 1, 0,  -1, 0, 1};
        Buffer positions = uploaded(device, kHalves, {.storage = true});
        const auto slot = storageSlot(positions);
        REQUIRE(slot.has_value());
        Pipeline pipeline = created(device, colorPipeline());
        const auto draws = twoAt<bfr_draw_indirect_args, kStride>(
            {.vertex_count = kVertices, .instance_count = 1, .first_vertex = 0, .first_instance = 0},
            {.vertex_count = kVertices, .instance_count = 1, .first_vertex = kVertices, .first_instance = 0});
        const auto indexedDraws = twoAt<bfr_draw_indexed_indirect_args, kStride>(
            {.index_count = kVertices, .instance_count = 1, .first_index = 0, .vertex_offset = 0, .first_instance = 0},
            {.index_count = kVertices,
             .instance_count = 1,
             .first_index = kVertices,
             .vertex_offset = 0,
             .first_instance = 0});
        constexpr std::array<std::uint16_t, std::size_t{2} * kVertices> kIndices{0, 1, 2, 3, 4, 5};
        constexpr std::array<std::uint32_t, 1> kOneDraw{1};
        Buffer args = uploaded(device, draws, {.indirect = true});
        Buffer indexedArgs = uploaded(device, indexedDraws, {.indirect = true});
        Buffer indices = uploaded(device, kIndices, {.index = true});
        Buffer count = uploaded(device, kOneDraw, {.indirect = true});

        const bool indexed = GENERATE(false, true);
        const auto bind = [&](CommandList list) {
            CHECK(device.bindPipeline(list, pipeline).has_value());
            CHECK(device.pushConstants(list, bytesOf({.positions = *slot, .color = {1, 0, 0, 1}})).has_value());
            CHECK(device.bindIndexBuffer(list, indices, 0, IndexType::Uint16).has_value());
        };
        const auto drawIndirect = [&](CommandList list, std::uint32_t drawCount) {
            return indexed ? device.drawIndexedIndirect(list, indexedArgs, 0, drawCount, kStride)
                           : device.drawIndirect(list, args, 0, drawCount, kStride);
        };
        const auto drawCounted = [&](CommandList list, std::uint32_t maxDrawCount) {
            return indexed ? device.drawIndexedIndirectCount(list, indexedArgs, 0, count, 0, maxDrawCount, kStride)
                           : device.drawIndirectCount(list, args, 0, count, 0, maxDrawCount, kStride);
        };
        constexpr std::uint32_t kTopRow = 1;
        constexpr std::uint32_t kBottomRow = kSide - 2;

        SECTION("one draw draws the first arguments") {
            const Rendered image{device, caps.limits, [&](CommandList list) {
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
            const Rendered image{device, caps.limits, [&](CommandList list) {
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
            const Rendered image{device, caps.limits, [&](CommandList list) {
                                     bind(list);
                                     CHECK(drawCounted(list, 2).has_value());
                                 }};
            CHECK(image.at(kSide / 2, kTopRow) == kRed);
            CHECK(image.at(kSide / 2, kBottomRow) == kBlack);
        }
        for (Buffer* buffer : std::array{&positions, &args, &indexedArgs, &indices, &count}) {
            CHECK(slotsResult(device, destroyed(*buffer)) == Result::ErrorInvalidHandle);
        }
        CHECK(bindResult(device, 0, destroyed(pipeline)) == Result::ErrorInvalidHandle);
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
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        constexpr std::uint64_t kBytes = std::uint64_t{kWorkgroup} * sizeof(std::uint32_t);
        Buffer target = created(device, {.size = kBytes, .usage = {.copySrc = true, .storage = true}});
        Buffer readback = created(device,
                                  {.size = kBytes, .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
        constexpr std::array kOneGroup{bfr_dispatch_indirect_args{.x = 1, .y = 1, .z = 1}};
        Buffer args = uploaded(device, kOneGroup, {.indirect = true});
        Pipeline pipeline = created(device, ComputePipelineDesc{.compute = shaderCode(bfr::test::spirv::kBindlessComp),
                                                                .workgroupSize = {kWorkgroup, 1, 1}});
        const auto slot = storageSlot(target);
        REQUIRE(slot.has_value());
        auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists, .pushConstantUpdates = 1});
        REQUIRE(pool.has_value());
        auto list = pool->beginCommands();
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
        const auto read = readback.map();
        REQUIRE(read.has_value());
        std::array<std::uint32_t, kWorkgroup> values{};
        std::memcpy(values.data(), *read, kBytes);
        for (std::uint32_t i = 0; i < kWorkgroup; ++i) {
            CHECK(values.at(i) == i);
        }
        CHECK(bindResult(device, 0, destroyed(pipeline)) == Result::ErrorInvalidHandle);
        for (Buffer* buffer : std::array{&target, &readback, &args}) {
            CHECK(slotsResult(device, destroyed(*buffer)) == Result::ErrorInvalidHandle);
        }
        CHECK(resetResult(device, destroyed(*pool)) == Result::ErrorInvalidHandle);
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
        const FirstAdapterDevice made{recordingInto(backend, messages), [](DeviceDesc& desc, const Caps& caps) {
                                          desc.features.dynamicTopology = caps.features.dynamicTopology;
                                          desc.features.triangleFans = caps.features.triangleFans;
                                      }};
        const Device& device = made.device;
        const Caps caps = capsOf(made.instance, kFirstAdapter);
        // The corners of clip space: bottom left, bottom right, top left, top
        // right. As a strip they cover it; their first three as a list cover
        // the bottom-left half.
        constexpr std::uint32_t kCorners = 4;
        constexpr std::array<float, std::size_t{kCorners} * kPositionFloats> kQuad{-1, -1, 0, 1, 1, -1, 0, 1,
                                                                                   -1, 1,  0, 1, 1, 1,  0, 1};
        Buffer positions = uploaded(device, kQuad, {.storage = true});
        const auto slot = storageSlot(positions);
        REQUIRE(slot.has_value());
        Pipeline list = created(device, colorPipeline());
        GraphicsPipelineDesc stripDesc = colorPipeline();
        stripDesc.topology = Topology::TriangleStrip;
        stripDesc.primitiveRestart = PrimitiveRestart::Index16;
        Pipeline strip = created(device, stripDesc);
        // A strip of one vertex, a restart, and a strip of the first three.
        constexpr std::uint16_t kRestart = 0xFFFF;
        constexpr std::array<std::uint16_t, 5> kRestarted{3, kRestart, 0, 1, 2};
        Buffer indices = uploaded(device, kRestarted, {.index = true});
        // Strips and fans are drawn indexed: lavapipe (Mesa 25.2.8) records
        // nothing at all for a non-indexed one, clear included, with
        // validation silent, where a GPU driver draws it.
        constexpr std::array<std::uint16_t, kCorners> kInOrder{0, 1, 2, 3};
        Buffer inOrder = uploaded(device, kInOrder, {.index = true});
        const auto bind = [&](CommandList commands, const Pipeline& pipeline) {
            CHECK(device.bindPipeline(commands, pipeline).has_value());
            CHECK(device.pushConstants(commands, bytesOf({.positions = *slot, .color = {1, 0, 0, 1}})).has_value());
        };
        constexpr std::uint32_t kLeft = 0;
        constexpr std::uint32_t kRight = kSide - 1;
        constexpr std::uint32_t kTop = 0;
        constexpr std::uint32_t kBottom = kSide - 1;

        SECTION("a list pipeline draws a list") {
            const Rendered image{device, caps.limits, [&](CommandList commands) {
                                     bind(commands, list);
                                     CHECK(device.draw(commands, kCorners, 1, 0, 0).has_value());
                                 }};
            CHECK(image.at(kLeft, kBottom) == kRed);
            CHECK(image.at(kRight, kTop) == kBlack);
        }
        SECTION("setTopology makes a list pipeline's draws a strip, and fails where the device sets none") {
            if (caps.features.dynamicTopology) {
                const Rendered image{
                    device, caps.limits, [&](CommandList commands) {
                        bind(commands, list);
                        CHECK(device.setTopology(commands, Topology::TriangleStrip).has_value());
                        CHECK(device.bindIndexBuffer(commands, inOrder, 0, IndexType::Uint16).has_value());
                        CHECK(device.drawIndexed(commands, kCorners, 1, 0, 0, 0).has_value());
                    }};
                CHECK(image.at(kLeft, kBottom) == kRed);
                CHECK(image.at(kRight, kTop) == kRed);
            }
            else {
                const Rendered image{device, caps.limits, [&](CommandList commands) {
                                         const auto set = device.setTopology(commands, Topology::TriangleStrip);
                                         REQUIRE_FALSE(set.has_value());
                                         CHECK(set.error().code == Result::ErrorUnsupported);
                                     }};
            }
        }
        SECTION("a strip pipeline restarts at an index of all ones") {
            const Rendered image{device, caps.limits, [&](CommandList commands) {
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
            Buffer round = uploaded(device, kRound, {.storage = true});
            const auto roundSlot = storageSlot(round);
            REQUIRE(roundSlot.has_value());
            const Rendered image{device, caps.limits, [&](CommandList commands) {
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
            CHECK(slotsResult(device, destroyed(round)) == Result::ErrorInvalidHandle);
        }
        CHECK(slotsResult(device, destroyed(positions)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(indices)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(inOrder)) == Result::ErrorInvalidHandle);
        CHECK(bindResult(device, 0, destroyed(list)) == Result::ErrorInvalidHandle);
        CHECK(bindResult(device, 0, destroyed(strip)) == Result::ErrorInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

namespace {

constexpr std::array<std::uint8_t, kTexelBytes> kGreen{0, 255, 0, 255};
constexpr std::uint32_t kFourSamples = 4;

// Whether the adapter renders format with kFourSamples and resolves it.
bool resolvesFour(const Instance& instance, Format format) {
    const auto caps = formatCapsOf(instance, kFirstAdapter, format);
    REQUIRE(caps.has_value());
    return (caps->sampleCounts & kFourSamples) != 0 && caps->resolve;
}

// One rendering of only depth as a depth-stencil attachment, as attachment
// says, recorded, submitted and waited on, with depth moved from layout before
// to after around it, and resolve, where not null, moved to CopySrc after it.
void renderDepth(const Device& device, const Texture& depth, DepthStencilAttachment attachment, TextureLayout before,
                 TextureLayout after, const Texture* resolve = nullptr) {
    attachment.texture = depth;
    auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
    REQUIRE(pool.has_value());
    auto list = pool->beginCommands();
    REQUIRE(list.has_value());
    CHECK(device.textureBarrier(*list, depth, before, TextureLayout::DepthStencilTarget, {}).has_value());
    if (resolve != nullptr) {
        CHECK(device.textureBarrier(*list, *resolve, TextureLayout::Undefined, TextureLayout::DepthStencilTarget, {})
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
    CHECK(device.textureBarrier(*list, depth, TextureLayout::DepthStencilTarget, after, {}).has_value());
    if (resolve != nullptr) {
        CHECK(device.textureBarrier(*list, *resolve, TextureLayout::DepthStencilTarget, TextureLayout::CopySrc, {})
                  .has_value());
    }
    const auto sent = device.submit({.queue = 0, .commandLists = std::span{&*list, 1}, .waits = {}});
    INFO(failureOf(sent));
    REQUIRE(sent.has_value());
    REQUIRE(device.wait(*sent, kTestWait).has_value());
    CHECK(resetResult(device, destroyed(*pool)) == Result::ErrorInvalidHandle);
}

// The first texel of aspect of texture, in layout CopySrc, as bytes, copied
// as a row of the pitch limits asks for.
std::array<std::byte, kTexelBytes> firstTexel(const Device& device, const CapsLimits& limits, const Texture& texture,
                                              TextureAspect aspect) {
    const std::uint64_t bytesPerRow = alignUp(kTexelBytes, std::max<std::uint64_t>(limits.copyRowPitchAlignment, 1));
    Buffer readback = created(device,
                              {.size = bytesPerRow, .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
    auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
    REQUIRE(pool.has_value());
    auto list = pool->beginCommands();
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
    const auto read = readback.map();
    REQUIRE(read.has_value());
    std::array<std::byte, kTexelBytes> bytes{};
    std::memcpy(bytes.data(), *read, bytes.size());
    CHECK(slotsResult(device, destroyed(readback)) == Result::ErrorInvalidHandle);
    CHECK(resetResult(device, destroyed(*pool)) == Result::ErrorInvalidHandle);
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
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        const Caps caps = capsOf(made.instance, kFirstAdapter);
        // The whole of clip space at depth z.
        const auto whole = [](float z) {
            return std::array<float, std::size_t{kVertices} * kPositionFloats>{-1, -1, z, 1, 3, -1, z, 1, -1, 3, z, 1};
        };
        constexpr float kNearDepth = 0.25F;
        constexpr float kFarDepth = 0.75F;
        Buffer near = uploaded(device, whole(kNearDepth), {.storage = true});
        Buffer far = uploaded(device, whole(kFarDepth), {.storage = true});
        // Bottom left, bottom right and top right: the bottom-right half, its
        // edge through the pixels on the diagonal.
        constexpr std::array<float, std::size_t{kVertices} * kPositionFloats> kHalf{-1, -1, 0, 1, 1, -1,
                                                                                    0,  1,  1, 1, 0, 1};
        Buffer half = uploaded(device, kHalf, {.storage = true});
        const auto slotOf = [&](Buffer& buffer) {
            const auto slot = storageSlot(buffer);
            REQUIRE(slot.has_value());
            return *slot;
        };
        const auto draw = [&](CommandList commands, const Pipeline& pipeline, Buffer& positions,
                              const std::array<float, 4>& color) {
            CHECK(device.bindPipeline(commands, pipeline).has_value());
            CHECK(
                device.pushConstants(commands, bytesOf({.positions = slotOf(positions), .color = color})).has_value());
            CHECK(device.draw(commands, kVertices, 1, 0, 0).has_value());
        };
        constexpr std::uint32_t kRight = kSide - 1;
        constexpr std::uint32_t kBottom = kSide - 1;

        SECTION("the nearer of two triangles stays in front, drawn first") {
            const auto depthCaps = formatCapsOf(made.instance, kFirstAdapter, Format::D32Float);
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
            Pipeline tested = created(device, desc);
            const Rendered image{device,
                                 caps.limits,
                                 [&](CommandList commands) {
                                     draw(commands, tested, near, {1, 0, 0, 1});
                                     draw(commands, tested, far, {0, 1, 0, 1});
                                 },
                                 {.depth = Format::D32Float}};
            CHECK(image.at(kSide / 2, kSide / 2) == kRed);
            CHECK(bindResult(device, 0, destroyed(tested)) == Result::ErrorInvalidHandle);
        }
        SECTION("an edge pixel's samples average into its resolved texel") {
            if (!resolvesFour(made.instance, Format::Rgba8Unorm)) {
                SKIP("the adapter does not resolve Rgba8Unorm from 4 samples");
            }
            GraphicsPipelineDesc desc = colorPipeline();
            desc.samples = kFourSamples;
            Pipeline multisampled = created(device, desc);
            const bool byCommand = GENERATE(false, true);
            INFO((byCommand ? "by resolveTexture" : "as rendering ends"));
            if (byCommand && !caps.resolve.texture) {
                SKIP("the device resolves only as a rendering ends (Caps::resolve.texture)");
            }
            const Rendered image{device,
                                 caps.limits,
                                 [&](CommandList commands) { draw(commands, multisampled, half, {1, 0, 0, 1}); },
                                 {.samples = kFourSamples, .byCommand = byCommand}};
            CHECK(image.at(kRight, kBottom) == kRed);
            CHECK(image.at(0, 0) == kBlack);
            // Partly covered: neither the clear nor the triangle.
            const std::uint8_t edge = image.at(kSide / 2, (kSide / 2) - 1).front();
            CHECK(edge > 0);
            CHECK(edge < kRed.front());
            CHECK(bindResult(device, 0, destroyed(multisampled)) == Result::ErrorInvalidHandle);
        }
        SECTION("a sample mask of none writes nothing, where the device takes one") {
            if (!caps.features.sampleMask) {
                SKIP("the adapter writes every sample");
            }
            GraphicsPipelineDesc desc = colorPipeline();
            desc.sampleMask = 0;
            Pipeline masked = created(device, desc);
            const Rendered image{device, caps.limits, [&](CommandList commands) {
                                     draw(commands, masked, near, {1, 0, 0, 1});
                                 }};
            CHECK(image.at(kSide / 2, kSide / 2) == kBlack);
            CHECK(bindResult(device, 0, destroyed(masked)) == Result::ErrorInvalidHandle);
        }
        CHECK(slotsResult(device, destroyed(near)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(far)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(half)) == Result::ErrorInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device loads and stores depth and stencil apart, and resolves depth by a mode Caps::resolve has",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        const Caps caps = capsOf(made.instance, kFirstAdapter);

        SECTION("stencil kept while depth is cleared") {
            const Format format = GENERATE(Format::D24UnormS8Uint, Format::D32FloatS8Uint);
            const auto formatCaps = formatCapsOf(made.instance, kFirstAdapter, format);
            REQUIRE(formatCaps.has_value());
            if (!formatCaps->depthStencil || !formatCaps->copySrc) {
                SKIP("the adapter does not render to and copy from " << static_cast<int>(format));
            }
            Texture depth = created(device, depthTexture(format, 1, {.copySrc = true, .depthStencil = true}));
            constexpr std::uint32_t kKept = 7;
            renderDepth(
                device, depth,
                {.depthLoadOp = LoadOp::Clear, .stencilLoadOp = LoadOp::Clear, .clearDepth = 1, .clearStencil = kKept},
                TextureLayout::Undefined, TextureLayout::DepthStencilTarget);
            renderDepth(
                device, depth,
                {.depthLoadOp = LoadOp::Clear, .stencilLoadOp = LoadOp::Load, .clearDepth = 0, .clearStencil = 0},
                TextureLayout::DepthStencilTarget, TextureLayout::CopySrc);
            CHECK(std::to_integer<std::uint32_t>(
                      firstTexel(device, caps.limits, depth, TextureAspect::Stencil).front()) == kKept);
            CHECK(slotsResult(device, destroyed(depth)) == Result::ErrorInvalidHandle);
        }
        SECTION("a multisampled depth attachment resolves by sample zero, where the device has it") {
            if ((caps.resolve.depthModes & (1U << std::to_underlying(ResolveMode::SampleZero))) == 0 ||
                !resolvesFour(made.instance, Format::D32Float)) {
                SKIP("the adapter does not resolve D32Float depth from 4 samples by sample zero");
            }
            Texture depth = created(device, depthTexture(Format::D32Float, kFourSamples, {.depthStencil = true}));
            Texture resolved = created(device,
                                       depthTexture(Format::D32Float, 1, {.copySrc = true, .depthStencil = true}));
            constexpr float kCleared = 0.5F;
            renderDepth(device, depth,
                        {.depthLoadOp = LoadOp::Clear,
                         .depthStoreOp = StoreOp::DontCare,
                         .clearDepth = kCleared,
                         .resolveTexture = resolved,
                         .depthResolve = ResolveMode::SampleZero},
                        TextureLayout::Undefined, TextureLayout::DepthStencilTarget, &resolved);
            CHECK(std::bit_cast<float>(firstTexel(device, caps.limits, resolved, TextureAspect::Depth)) == kCleared);
            CHECK(slotsResult(device, destroyed(depth)) == Result::ErrorInvalidHandle);
            CHECK(slotsResult(device, destroyed(resolved)) == Result::ErrorInvalidHandle);
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
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const auto formatCaps = formatCapsOf(made.instance, kFirstAdapter, Format::D32Float);
        REQUIRE(formatCaps.has_value());
        if (!formatCaps->sampled || !formatCaps->depthStencil) {
            SKIP("the adapter does not sample and render to D32Float");
        }
        const Device& device = made.device;
        const Caps caps = capsOf(made.instance, kFirstAdapter);
        Texture depth = created(device, depthTexture(Format::D32Float, 1, {.sampled = true, .depthStencil = true}));
        constexpr float kCleared = 0.25F;
        renderDepth(device, depth, {.depthLoadOp = LoadOp::Clear, .clearDepth = kCleared}, TextureLayout::Undefined,
                    TextureLayout::DepthStencilRead);
        const auto textureSlot = depth.textureSlots();
        REQUIRE(textureSlot.has_value());
        auto sampler = device.createSampler({});
        REQUIRE(sampler.has_value());
        const auto samplerSlot = sampler->samplerSlot();
        REQUIRE(samplerSlot.has_value());
        // In front of what depth holds, so the test passes.
        constexpr std::array<float, std::size_t{kVertices} * kPositionFloats> kNear{-1, -1, 0,  1, 3, -1,
                                                                                    0,  1,  -1, 3, 0, 1};
        Buffer positions = uploaded(device, kNear, {.storage = true});
        const auto positionSlot = storageSlot(positions);
        REQUIRE(positionSlot.has_value());
        GraphicsPipelineDesc desc = colorPipeline();
        desc.fragment = shaderCode(bfr::test::spirv::kBindlessFrag);
        desc.depthStencil = {.depthTest = true,
                             .depthWrite = false,
                             .depthCompare = CompareOp::Less,
                             .stencilTest = false,
                             .front = {},
                             .back = {}};
        desc.depthStencilFormat = Format::D32Float;
        Pipeline sampling = created(device, desc);
        const Rendered image{device,
                             caps.limits,
                             [&](CommandList commands) {
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
        CHECK(bindResult(device, 0, destroyed(sampling)) == Result::ErrorInvalidHandle);
        CHECK(slotResult(device, destroyed(*sampler)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(positions)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(depth)) == Result::ErrorInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device draws with mesh shaders, directly and indirectly, and with a task shader, where it has them",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages), [](DeviceDesc& desc, const Caps& caps) {
                                          desc.features.meshShader = caps.features.meshShader;
                                          desc.features.taskShader = caps.features.taskShader;
                                          desc.features.multiDrawIndirect = caps.features.multiDrawIndirect;
                                          desc.features.drawIndirectCount = caps.features.drawIndirectCount;
                                      }};
        const Device& device = made.device;
        const Caps caps = capsOf(made.instance, kFirstAdapter);
        GraphicsPipelineDesc meshDesc = colorPipeline();
        meshDesc.vertex = {};
        meshDesc.mesh = shaderCode(bfr::test::spirv::kTriangleMesh);
        meshDesc.meshWorkgroupSize = {1, 1, 1};
        if (!caps.features.meshShader) {
            const auto refused = device.createGraphicsPipeline(meshDesc);
            REQUIRE_FALSE(refused.has_value());
            CHECK(refused.error().code == Result::ErrorUnsupported);
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
        Buffer positions = uploaded(device, kHalves, {.storage = true});
        const auto slot = storageSlot(positions);
        REQUIRE(slot.has_value());
        Pipeline mesh = created(device, meshDesc);
        // Two workgroups; then, for the count draw, one, and two again, which
        // only a draw count past 1 reaches.
        constexpr std::array kGroups{bfr_dispatch_indirect_args{.x = 2, .y = 1, .z = 1},
                                     bfr_dispatch_indirect_args{.x = 1, .y = 1, .z = 1},
                                     bfr_dispatch_indirect_args{.x = 2, .y = 1, .z = 1}};
        constexpr std::array<std::uint32_t, 1> kOneDraw{1};
        Buffer args = uploaded(device, kGroups, {.indirect = true});
        Buffer count = uploaded(device, kOneDraw, {.indirect = true});
        const auto bind = [&](CommandList commands, const Pipeline& pipeline) {
            CHECK(device.bindPipeline(commands, pipeline).has_value());
            CHECK(device.pushConstants(commands, bytesOf({.positions = *slot, .color = {1, 0, 0, 1}})).has_value());
        };
        constexpr std::uint32_t kLeft = 0;
        constexpr std::uint32_t kRight = kSide - 1;
        constexpr std::uint32_t kTop = 0;
        constexpr std::uint32_t kBottom = kSide - 1;

        SECTION("one workgroup draws the first triangle") {
            const Rendered image{device, caps.limits, [&](CommandList commands) {
                                     bind(commands, mesh);
                                     CHECK(device.drawMesh(commands, 1, 1, 1).has_value());
                                 }};
            CHECK(image.at(kLeft, kBottom) == kRed);
            CHECK(image.at(kRight, kTop) == kBlack);
        }
        SECTION("two workgroups draw both, directly and indirectly") {
            const auto draw = GENERATE(0, 1);
            const Rendered image{device, caps.limits, [&](CommandList commands) {
                                     bind(commands, mesh);
                                     if (draw == 0) {
                                         CHECK(device.drawMesh(commands, 2, 1, 1).has_value());
                                     }
                                     else {
                                         CHECK(device
                                                   .drawMeshIndirect(commands, args, 0, 1,
                                                                     sizeof(bfr_dispatch_indirect_args))
                                                   .has_value());
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
            const Rendered image{device, caps.limits, [&](CommandList commands) {
                                     bind(commands, mesh);
                                     CHECK(device
                                               .drawMeshIndirectCount(commands, args,
                                                                      sizeof(bfr_dispatch_indirect_args), count, 0,
                                                                      kMaxDraws, sizeof(bfr_dispatch_indirect_args))
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
            taskDesc.task = shaderCode(bfr::test::spirv::kTriangleTask);
            taskDesc.taskWorkgroupSize = {1, 1, 1};
            Pipeline task = created(device, taskDesc);
            const Rendered image{device, caps.limits, [&](CommandList commands) {
                                     bind(commands, task);
                                     CHECK(device.drawMesh(commands, 1, 1, 1).has_value());
                                 }};
            CHECK(image.at(kLeft, kBottom) == kRed);
            CHECK(image.at(kRight, kTop) == kRed);
            CHECK(bindResult(device, 0, destroyed(task)) == Result::ErrorInvalidHandle);
        }
        CHECK(slotsResult(device, destroyed(positions)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(args)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(count)) == Result::ErrorInvalidHandle);
        CHECK(bindResult(device, 0, destroyed(mesh)) == Result::ErrorInvalidHandle);
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
        const FirstAdapterDevice made{recordingInto(backend, messages), [](DeviceDesc& desc, const Caps& caps) {
                                          desc.features.occlusionQueryPrecise = caps.features.occlusionQueryPrecise;
                                      }};
        const Device& device = made.device;
        const Caps caps = capsOf(made.instance, kFirstAdapter);
        constexpr std::uint32_t kQueries = 2;
        if (!caps.features.queries) {
            CHECK(device.createQueryPool({.type = QueryType::Occlusion, .count = kQueries, .statistics = {}})
                      .error()
                      .code == Result::ErrorUnsupported);
            SKIP("the adapter makes no query pools");
        }
        if (!takesSpirv(backend)) {
            SKIP("no " << backend.name << " bytecode is checked in");
        }
        auto pool = device.createQueryPool({.type = QueryType::Occlusion, .count = kQueries, .statistics = {}});
        INFO(failureOf(pool));
        REQUIRE(pool.has_value());
        const auto layout = pool->queryLayout();
        REQUIRE(layout.has_value());
        REQUIRE(layout->value != BFR_NO_OFFSET);
        REQUIRE(layout->stride >= layout->value + sizeof(std::uint64_t));
        const std::uint64_t resultBytes = std::uint64_t{layout->stride} * kQueries;
        Buffer results = created(device,
                                 {.size = resultBytes, .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
        const bool precise = caps.features.occlusionQueryPrecise;
        Buffer positions = uploaded(device, kTopHalf, {.storage = true});
        const auto slot = storageSlot(positions);
        REQUIRE(slot.has_value());
        const auto depthCaps = formatCapsOf(made.instance, kFirstAdapter, Format::D32Float);
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
        Pipeline passes = created(device, depthDesc);
        depthDesc.depthStencil.depthCompare = CompareOp::Less;
        Pipeline rejected = created(device, depthDesc);
        Texture depth = created(device, {.dimension = TextureDimension::D2,
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
        Texture target = created(device, {.dimension = TextureDimension::D2,
                                          .format = Format::Rgba8Unorm,
                                          .width = kSide,
                                          .height = kSide,
                                          .depth = 1,
                                          .layers = 1,
                                          .mipLevels = 1,
                                          .samples = 1,
                                          .usage = {.copySrc = true, .colorAttachment = true}});
        Buffer image = created(device, {.size = bytesPerRow * kSide,
                                        .memory = BufferMemory::Readback,
                                        .usage = {.copyDst = true}});
        constexpr std::uint32_t kPushes = 2;
        auto commands = device.createCommandPool({.queue = 0, .listCount = kPoolLists, .pushConstantUpdates = kPushes});
        REQUIRE(commands.has_value());
        auto list = commands->beginCommands();
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
        for (const auto& [query, pipeline] : {std::pair{0U, &passes}, std::pair{1U, &rejected}}) {
            CHECK(device.bindPipeline(*list, *pipeline).has_value());
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
        const auto mappedImage = image.map();
        REQUIRE(mappedImage.has_value());
        const std::span<const std::byte> texels = mappedBytes(*mappedImage, bytesPerRow * kSide);
        std::uint64_t red = 0;
        for (std::uint32_t y = 0; y < kSide; ++y) {
            for (std::uint32_t x = 0; x < kSide; ++x) {
                std::array<std::uint8_t, kTexelBytes> rgba{};
                std::memcpy(rgba.data(), texels.subspan((y * bytesPerRow) + (std::uint64_t{x} * kTexelBytes)).data(),
                            kTexelBytes);
                red += rgba == kRed ? 1U : 0U;
            }
        }
        REQUIRE(red != 0);
        const auto mappedResults = results.map();
        REQUIRE(mappedResults.has_value());
        const std::span<const std::byte> read = mappedBytes(*mappedResults, resultBytes);
        const std::uint64_t drawn = valueAt(read, layout->value);
        const std::uint64_t none = valueAt(read, layout->stride + std::uint64_t{layout->value});
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
                  .code == Result::ErrorInvalidArgument);
        CHECK(resetResult(device, destroyed(*commands)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(target)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(image)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(results)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(positions)) == Result::ErrorInvalidHandle);
        CHECK(bindResult(device, 0, destroyed(passes)) == Result::ErrorInvalidHandle);
        CHECK(bindResult(device, 0, destroyed(rejected)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(depth)) == Result::ErrorInvalidHandle);
        const bfr_query_pool gone = destroyed(*pool);
        CHECK(layoutResult(device, gone) == Result::ErrorInvalidHandle);
        // d1: stale destroy reported as an Error message (track B)
        bfr_destroy_query_pool(device.get(), gone);
        CHECK(layoutResult(device, gone) == Result::ErrorInvalidHandle);
        // d1: null destroy: no message (track B)
        bfr_destroy_query_pool(device.get(), bfr_query_pool{});
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device counts the pipeline statistics a pool selects, a dispatch's and a draw's, where queryLayout says",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages), [](DeviceDesc& desc, const Caps& caps) {
                                          desc.features.pipelineStatistics = caps.features.pipelineStatistics;
                                      }};
        const Device& device = made.device;
        const Caps caps = capsOf(made.instance, kFirstAdapter);
        if (!caps.features.queries || !caps.features.pipelineStatistics) {
            SKIP("the adapter counts no pipeline statistics");
        }
        if (!takesSpirv(backend)) {
            SKIP("no " << backend.name << " bytecode is checked in");
        }
        constexpr std::uint32_t kQueries = 2;
        auto pool = device.createQueryPool({.type = QueryType::PipelineStatistics,
                                            .count = kQueries,
                                            .statistics = {.inputVertices = true,
                                                           .inputPrimitives = true,
                                                           .vertexInvocations = true,
                                                           .computeInvocations = true}});
        INFO(failureOf(pool));
        REQUIRE(pool.has_value());
        const auto layout = pool->queryLayout();
        REQUIRE(layout.has_value());
        CHECK(layout->value == BFR_NO_OFFSET);
        const auto offsetOf = [&](PipelineStatistic statistic) {
            return layout->statistics.at(static_cast<std::size_t>(statistic));
        };
        for (const PipelineStatistic statistic :
             {PipelineStatistic::InputVertices, PipelineStatistic::VertexInvocations,
              PipelineStatistic::ComputeInvocations}) {
            INFO("statistic " << static_cast<int>(statistic));
            REQUIRE(offsetOf(statistic) != BFR_NO_OFFSET);
            REQUIRE(offsetOf(statistic) + sizeof(std::uint64_t) <= layout->stride);
        }
        const std::uint64_t resultBytes = std::uint64_t{layout->stride} * kQueries;
        Buffer results = created(device,
                                 {.size = resultBytes, .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
        constexpr std::uint64_t kStorageBytes = std::uint64_t{kWorkgroup} * sizeof(std::uint32_t);
        Buffer storage = created(device, {.size = kStorageBytes, .usage = {.storage = true}});
        const auto storageAt = storageSlot(storage);
        REQUIRE(storageAt.has_value());
        Pipeline compute = created(device, ComputePipelineDesc{.compute = shaderCode(bfr::test::spirv::kBindlessComp),
                                                               .workgroupSize = {kWorkgroup, 1, 1}});
        Buffer positions = uploaded(device, kTopHalf, {.storage = true});
        const auto positionsAt = storageSlot(positions);
        REQUIRE(positionsAt.has_value());
        Pipeline draw = created(device, colorPipeline());
        Texture target = created(device, {.dimension = TextureDimension::D2,
                                          .format = Format::Rgba8Unorm,
                                          .width = kSide,
                                          .height = kSide,
                                          .depth = 1,
                                          .layers = 1,
                                          .mipLevels = 1,
                                          .samples = 1,
                                          .usage = {.colorAttachment = true}});
        constexpr std::uint32_t kPushes = 2;
        auto commands = device.createCommandPool({.queue = 0, .listCount = kPoolLists, .pushConstantUpdates = kPushes});
        REQUIRE(commands.has_value());
        auto list = commands->beginCommands();
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

        const auto mapped = results.map();
        REQUIRE(mapped.has_value());
        const std::span<const std::byte> read = mappedBytes(*mapped, resultBytes);
        const auto statistic = [&](std::uint32_t query, PipelineStatistic which) {
            return valueAt(read, (std::uint64_t{layout->stride} * query) + offsetOf(which));
        };
        CHECK(statistic(0, PipelineStatistic::ComputeInvocations) == kWorkgroup);
        CHECK(statistic(0, PipelineStatistic::InputVertices) == 0);
        CHECK(statistic(1, PipelineStatistic::InputVertices) == kVertices);
        CHECK(statistic(1, PipelineStatistic::InputPrimitives) == 1);
        CHECK(statistic(1, PipelineStatistic::VertexInvocations) != 0);
        CHECK(statistic(1, PipelineStatistic::ComputeInvocations) == 0);

        CHECK(resetResult(device, destroyed(*commands)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(target)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(results)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(storage)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(positions)) == Result::ErrorInvalidHandle);
        CHECK(bindResult(device, 0, destroyed(compute)) == Result::ErrorInvalidHandle);
        CHECK(bindResult(device, 0, destroyed(draw)) == Result::ErrorInvalidHandle);
        CHECK(layoutResult(device, destroyed(*pool)) == Result::ErrorInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device counts the primitives and invocations of a mesh draw, where it has mesh statistics",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages), [](DeviceDesc& desc, const Caps& caps) {
                                          desc.features.meshShader = caps.features.meshShader;
                                          desc.features.taskShader = caps.features.taskShader;
                                          desc.features.meshStatistics = caps.features.meshStatistics;
                                          desc.features.pipelineStatistics = caps.features.pipelineStatistics;
                                      }};
        const Device& device = made.device;
        const Caps caps = capsOf(made.instance, kFirstAdapter);
        if (!caps.features.queries || !caps.features.meshShader || !caps.features.meshStatistics ||
            !caps.features.pipelineStatistics) {
            SKIP("the adapter counts no mesh work");
        }
        if (!takesSpirv(backend)) {
            SKIP("no " << backend.name << " bytecode is checked in");
        }
        auto primitives = device.createQueryPool(
            {.type = QueryType::MeshPrimitivesGenerated, .count = 1, .statistics = {}});
        INFO(failureOf(primitives));
        REQUIRE(primitives.has_value());
        auto invocations = device.createQueryPool(
            {.type = QueryType::PipelineStatistics, .count = 1, .statistics = {.meshInvocations = true}});
        INFO(failureOf(invocations));
        REQUIRE(invocations.has_value());
        const auto primitivesLayout = primitives->queryLayout();
        const auto invocationsLayout = invocations->queryLayout();
        REQUIRE(primitivesLayout.has_value());
        REQUIRE(invocationsLayout.has_value());
        const std::uint32_t meshAt = invocationsLayout->statistics.at(
            static_cast<std::size_t>(PipelineStatistic::MeshInvocations));
        REQUIRE(primitivesLayout->value != BFR_NO_OFFSET);
        REQUIRE(meshAt != BFR_NO_OFFSET);
        const std::uint64_t resultBytes = std::uint64_t{primitivesLayout->stride} + invocationsLayout->stride;
        Buffer results = created(device,
                                 {.size = resultBytes, .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
        // Two triangles covering clip space, one a workgroup.
        constexpr std::uint32_t kTriangleCorners = 6;
        constexpr std::array<float, std::size_t{kTriangleCorners} * kPositionFloats> kHalves{-1, -1, 0,  1, 1, -1,
                                                                                             0,  1,  -1, 1, 0, 1,
                                                                                             1,  -1, 0,  1, 1, 1,
                                                                                             0,  1,  -1, 1, 0, 1};
        constexpr std::uint32_t kGroups = 2;
        Buffer positions = uploaded(device, kHalves, {.storage = true});
        const auto slot = storageSlot(positions);
        REQUIRE(slot.has_value());
        GraphicsPipelineDesc meshDesc = colorPipeline();
        meshDesc.vertex = {};
        meshDesc.mesh = shaderCode(bfr::test::spirv::kTriangleMesh);
        meshDesc.meshWorkgroupSize = {1, 1, 1};
        Pipeline mesh = created(device, meshDesc);
        Texture target = created(device, {.dimension = TextureDimension::D2,
                                          .format = Format::Rgba8Unorm,
                                          .width = kSide,
                                          .height = kSide,
                                          .depth = 1,
                                          .layers = 1,
                                          .mipLevels = 1,
                                          .samples = 1,
                                          .usage = {.colorAttachment = true}});
        auto commands = device.createCommandPool({.queue = 0, .listCount = kPoolLists, .pushConstantUpdates = 1});
        REQUIRE(commands.has_value());
        auto list = commands->beginCommands();
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

        const auto mapped = results.map();
        REQUIRE(mapped.has_value());
        const std::span<const std::byte> read = mappedBytes(*mapped, resultBytes);
        CHECK(valueAt(read, primitivesLayout->value) == kGroups);
        CHECK(valueAt(read, std::uint64_t{primitivesLayout->stride} + meshAt) != 0);

        CHECK(resetResult(device, destroyed(*commands)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(target)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(results)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(positions)) == Result::ErrorInvalidHandle);
        CHECK(bindResult(device, 0, destroyed(mesh)) == Result::ErrorInvalidHandle);
        CHECK(layoutResult(device, destroyed(*primitives)) == Result::ErrorInvalidHandle);
        CHECK(layoutResult(device, destroyed(*invocations)) == Result::ErrorInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Vulkan refuses indirect arguments it doesn't take: offset, stride, or draws or a count past the buffer",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    if (backend.backend != Backend::Vulkan) {
        SKIP("the rules are Vulkan's");
    }
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        constexpr std::uint64_t kDrawBytes = 16;
        constexpr std::uint32_t kTwoDraws = 2;
        constexpr std::uint64_t kUnaligned = 2;
        constexpr std::uint32_t kShortStride = 8;
        Buffer arguments = created(device, BufferDesc{.size = kDrawBytes * kTwoDraws, .usage = {.indirect = true}});
        auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        const auto list = pool->beginCommands();
        REQUIRE(list.has_value());

        CHECK(device.drawIndirect(*list, arguments, kUnaligned, 1, 0).error().code == Result::ErrorInvalidArgument);
        CHECK(device.drawIndirect(*list, arguments, 0, kTwoDraws, kShortStride).error().code ==
              Result::ErrorInvalidArgument);
        CHECK(device.drawIndirect(*list, arguments, kDrawBytes, kTwoDraws, kDrawBytes).error().code ==
              Result::ErrorInvalidArgument);
        // 4-aligned offsets so large that adding the arguments' bytes wraps.
        constexpr std::uint64_t kWrappingOffset = ~std::uint64_t{3};
        CHECK(device.drawIndirect(*list, arguments, kWrappingOffset, 1, 0).error().code ==
              Result::ErrorInvalidArgument);
        CHECK(device.drawIndirectCount(*list, arguments, 0, arguments, kWrappingOffset, 1, kDrawBytes).error().code ==
              Result::ErrorInvalidArgument);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}
