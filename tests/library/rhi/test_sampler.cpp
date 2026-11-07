// What every backend does with samplers, through the interface alone: the
// slot each holds, stale handles, leftovers, and a texture sampled through
// one. Each case runs on every backend the build has; the sampling case runs
// the SPIR-V in tests/library/shaders, so it skips on a backend with none
// checked in.

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/instance.h"
#include "bfr/rhi/pipeline.h"
#include "bfr/rhi/sampler.h"
#include "bfr/rhi/texture.h"
#include "rhi_test_support.h"
#include "shaders/spirv.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_container_properties.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

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

// An Instance and a device on its first adapter, presenting nowhere, holding
// what desc says.
struct SamplerDevice {
    SamplerDevice(const InstanceDesc& instanceDesc, const DeviceDesc& desc) {
        auto created = createInstance(instanceDesc, kValidation);
        INFO(failureOf(created));
        REQUIRE(created.has_value());
        instance = std::move(*created);
        auto made = instance->createDevice(0, kNoWindow, desc);
        INFO(failureOf(made));
        REQUIRE(made.has_value());
        device = std::move(*made);
    }

    std::unique_ptr<Instance> instance;
    std::unique_ptr<Device> device;
};

SamplerHandle created(Device& device, const SamplerDesc& desc) {
    const auto sampler = device.createSampler(desc);
    INFO(failureOf(sampler));
    REQUIRE(sampler.has_value());
    return *sampler;
}

BufferHandle created(Device& device, const BufferDesc& desc) {
    const auto buffer = device.createBuffer(desc);
    INFO(failureOf(buffer));
    REQUIRE(buffer.has_value());
    return *buffer;
}

// Whether backend takes the SPIR-V checked in, as test_pipeline.cpp says.
bool takesSpirv(const TestBackend& backend) {
    return backend.backend == Backend::Vulkan;
}

// The push constants sample.comp declares.
struct SamplePush {
    std::uint32_t textureSlot = 0;
    std::uint32_t samplerSlot = 0;
    std::uint32_t target = 0;
};

// sample.comp's invocations: one value each.
constexpr std::uint32_t kSamples = 2;
// The texture it samples: 2 by 1 texels, black then white.
constexpr std::uint32_t kTexels = 2;
constexpr std::array<std::uint8_t, kTexels> kBlackWhite{0, 255};
// One step of an 8-bit unorm, the most a linear filter's weights may round by.
constexpr float kUnormStep = 1.0F / 255.0F;
// What a linear filter reads where sample.comp samples inside: a quarter of
// the way from the black texel's centre to the white one's.
constexpr float kQuarterWhite = 0.25F;

std::uint64_t alignUp(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

} // namespace

TEST_CASE("Device gives each sampler its own slot, and refuses one past the table with RhiLimitReached",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    constexpr std::uint32_t kSlots = 2;
    std::vector<std::string> messages;
    {
        const SamplerDevice made{recordingInto(backend, messages),
                                 testDeviceWith([](DeviceDesc& desc) { desc.maxBindlessSamplers = kSlots; })};
        Device& device = *made.device;
        const SamplerHandle first = created(device, SamplerDesc{});
        const SamplerHandle second = created(device, {.magFilter = Filter::Linear, .minFilter = Filter::Linear});
        const auto firstSlot = device.samplerSlot(first);
        const auto secondSlot = device.samplerSlot(second);
        REQUIRE(firstSlot.has_value());
        REQUIRE(secondSlot.has_value());
        CHECK(*firstSlot != *secondSlot);
        CHECK(*firstSlot < kSlots);
        CHECK(*secondSlot < kSlots);
        CHECK(device.createSampler({}).error().code == ErrorCode::RhiLimitReached);

        // No submit uses it, so its slot is free again once it is destroyed.
        CHECK(device.destroySampler(first).has_value());
        const SamplerHandle third = created(device, SamplerDesc{});
        CHECK(device.samplerSlot(third).has_value());
        CHECK(device.destroySampler(second).has_value());
        CHECK(device.destroySampler(third).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device fails sampler calls with RhiInvalidHandle for a sampler that is gone", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const SamplerDevice made{recordingInto(backend, messages), kTestDevice};
        Device& device = *made.device;
        const SamplerHandle gone = created(device, SamplerDesc{});
        REQUIRE(device.destroySampler(gone).has_value());

        CHECK(device.samplerSlot(gone).error().code == ErrorCode::RhiInvalidHandle);
        CHECK(device.destroySampler(gone).error().code == ErrorCode::RhiInvalidHandle);
        CHECK(device.destroySampler({}).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device destroys the samplers left when it goes, and says so", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const SamplerDevice made{recordingInto(backend, messages), kTestDevice};
        static_cast<void>(created(*made.device, SamplerDesc{}));
    }

    const std::size_t warnings = bfr::core::kLoggingEnabled ? 1 : 0;
    REQUIRE(messages.size() == warnings);
    for (const std::string& message : messages) {
        CHECK_THAT(message, Catch::Matchers::ContainsSubstring("1 samplers were left when the device went"));
    }
}

TEST_CASE("Device samples a texture through a sampler's slot: nearest and linear, border and edge", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    if (!takesSpirv(backend)) {
        SKIP("no " << backend.name << " bytecode is checked in");
    }
    std::vector<std::string> messages;
    {
        const SamplerDevice made{recordingInto(backend, messages), kTestDevice};
        Device& device = *made.device;
        const Limits& limits = made.instance->adapters().front().caps.limits;

        const auto texture = device.createTexture({.dimension = TextureDimension::D2,
                                                   .format = Format::R8Unorm,
                                                   .width = kTexels,
                                                   .height = 1,
                                                   .depth = 1,
                                                   .layers = 1,
                                                   .mipLevels = 1,
                                                   .samples = 1,
                                                   .usage = {.copyDst = true, .sampled = true}});
        INFO(failureOf(texture));
        REQUIRE(texture.has_value());
        const std::uint64_t bytesPerRow = alignUp(kTexels, std::max<std::uint64_t>(limits.copyRowPitchAlignment, 1));
        const BufferHandle upload =
            created(device, {.size = bytesPerRow, .memory = BufferMemory::Upload, .usage = {.copySrc = true}});
        const auto mapped = device.mapBuffer(upload);
        REQUIRE(mapped.has_value());
        std::memcpy(mapped->data(), kBlackWhite.data(), kBlackWhite.size());

        // Nearest, with an opaque white border: black inside, white before it.
        const SamplerHandle bordered = created(device, {.addressU = AddressMode::ClampToBorder,
                                                        .addressV = AddressMode::ClampToBorder,
                                                        .addressW = AddressMode::ClampToBorder,
                                                        .borderColor = {1, 1, 1, 1}});
        // Linear, clamped to the edge: a quarter white inside, black before it.
        const SamplerHandle edged = created(device, {.magFilter = Filter::Linear,
                                                     .minFilter = Filter::Linear,
                                                     .addressU = AddressMode::ClampToEdge,
                                                     .addressV = AddressMode::ClampToEdge,
                                                     .addressW = AddressMode::ClampToEdge});
        constexpr std::uint64_t kBytes = std::uint64_t{kSamples} * sizeof(float);
        const BufferHandle borderedOut = created(device, {.size = kBytes, .usage = {.copySrc = true, .storage = true}});
        const BufferHandle edgedOut = created(device, {.size = kBytes, .usage = {.copySrc = true, .storage = true}});
        const BufferHandle readback =
            created(device, {.size = 2 * kBytes, .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
        const auto pipeline = device.createComputePipeline(
            {.compute = {.code = std::as_bytes(std::span{bfr::test::spirv::kSampleComp}), .entryPoint = "main"},
             .workgroupSize = {kSamples, 1, 1}});
        INFO(failureOf(pipeline));
        REQUIRE(pipeline.has_value());

        const auto textureSlots = device.textureSlots(*texture);
        const auto borderedSlot = device.samplerSlot(bordered);
        const auto edgedSlot = device.samplerSlot(edged);
        const auto borderedTarget = device.bufferSlot(borderedOut);
        const auto edgedTarget = device.bufferSlot(edgedOut);
        REQUIRE(textureSlots.has_value());
        REQUIRE(borderedSlot.has_value());
        REQUIRE(edgedSlot.has_value());
        REQUIRE(borderedTarget.has_value());
        REQUIRE(edgedTarget.has_value());

        const auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists, .pushConstantUpdates = 2});
        REQUIRE(pool.has_value());
        auto list = device.beginCommands(*pool);
        REQUIRE(list.has_value());
        CHECK(device.textureBarrier(*list, *texture, TextureLayout::Undefined, TextureLayout::CopyDst).has_value());
        const BufferTextureCopy region{.bufferOffset = 0,
                                       .bytesPerRow = bytesPerRow,
                                       .rowsPerImage = 1,
                                       .mipLevel = 0,
                                       .layer = 0,
                                       .aspect = TextureAspect::Color,
                                       .x = 0,
                                       .y = 0,
                                       .z = 0,
                                       .width = kTexels,
                                       .height = 1,
                                       .depth = 1};
        CHECK(device.copyBufferToTexture(*list, upload, *texture, std::span{&region, 1}).has_value());
        CHECK(device.textureBarrier(*list, *texture, TextureLayout::CopyDst, TextureLayout::Sampled).has_value());
        CHECK(device.bindPipeline(*list, *pipeline).has_value());
        for (const SamplePush& push :
             {SamplePush{.textureSlot = textureSlots->sampled, .samplerSlot = *borderedSlot, .target = *borderedTarget},
              SamplePush{.textureSlot = textureSlots->sampled, .samplerSlot = *edgedSlot, .target = *edgedTarget}}) {
            const auto bytes = std::bit_cast<std::array<std::byte, sizeof(SamplePush)>>(push);
            CHECK(device.pushConstants(*list, bytes).has_value());
            CHECK(device.dispatch(*list, 1, 1, 1).has_value());
        }
        CHECK(device.barrier(*list).has_value());
        const BufferCopy first{.srcOffset = 0, .dstOffset = 0, .size = kBytes};
        const BufferCopy second{.srcOffset = 0, .dstOffset = kBytes, .size = kBytes};
        CHECK(device.copyBuffer(*list, borderedOut, readback, std::span{&first, 1}).has_value());
        CHECK(device.copyBuffer(*list, edgedOut, readback, std::span{&second, 1}).has_value());
        const auto sent = device.submit({.queue = 0, .commandLists = std::span{&*list, 1}, .waits = {}});
        INFO(failureOf(sent));
        REQUIRE(sent.has_value());
        REQUIRE(device.wait(*sent, kTestWait).has_value());

        const auto read = device.mapBuffer(readback);
        REQUIRE(read.has_value());
        std::array<float, std::size_t{2} * kSamples> values{};
        std::memcpy(values.data(), read->data(), sizeof(values));
        using Catch::Matchers::WithinAbs;
        CHECK_THAT(values[0], WithinAbs(0.0, 0.0));
        CHECK_THAT(values[1], WithinAbs(1.0, 0.0));
        CHECK_THAT(values[2], WithinAbs(kQuarterWhite, kUnormStep));
        CHECK_THAT(values[3], WithinAbs(0.0, 0.0));
        CHECK(device.destroyPipeline(*pipeline).has_value());
        CHECK(device.destroySampler(bordered).has_value());
        CHECK(device.destroySampler(edged).has_value());
        for (const BufferHandle buffer : {upload, borderedOut, edgedOut, readback}) {
            CHECK(device.destroyBuffer(buffer).has_value());
        }
        CHECK(device.destroyTexture(*texture).has_value());
        CHECK(device.destroyCommandPool(*pool).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}
