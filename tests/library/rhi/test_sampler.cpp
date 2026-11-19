// What every backend does with samplers, through the public API alone: the
// slot each holds, stale handles, leftovers, and a texture sampled through
// one. Each case runs on every backend the build has; the sampling case runs
// the SPIR-V in tests/library/shaders, so it skips on a backend with none
// checked in.

#include "bfr.hpp"
#include "bfr/bfr.h"
#include "bfr_test_support.h"
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
#include <expected>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace bfr;
using namespace bfr::test;

namespace {

Sampler created(const Device& device, const SamplerDesc& desc) {
    auto sampler = device.createSampler(desc);
    INFO(failureOf(sampler));
    REQUIRE(sampler.has_value());
    return std::move(*sampler);
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

} // namespace

TEST_CASE("Device gives each sampler its own slot, and refuses one past the table with RhiLimitReached",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    constexpr std::uint32_t kSlots = 2;
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages), [](DeviceDesc& desc) {
                                          desc.maxBindlessSamplers = kSlots;
                                      }};
        const Device& device = made.device;
        Sampler first = created(device, SamplerDesc{});
        Sampler second = created(device, {.magFilter = Filter::Linear, .minFilter = Filter::Linear});
        const auto firstSlot = first.samplerSlot();
        const auto secondSlot = second.samplerSlot();
        REQUIRE(firstSlot.has_value());
        REQUIRE(secondSlot.has_value());
        CHECK(*firstSlot != *secondSlot);
        CHECK(*firstSlot < kSlots);
        CHECK(*secondSlot < kSlots);
        CHECK(device.createSampler({}).error().code == Result::ErrorLimitReached);

        // No submit uses it, so garbage collection frees its slot once it is
        // destroyed; until then the slot is held.
        CHECK(slotResult(device, destroyed(first)) == Result::ErrorInvalidHandle);
        CHECK(device.createSampler({}).error().code == Result::ErrorLimitReached);
        REQUIRE(device.runGarbageCollection().has_value());
        Sampler third = created(device, SamplerDesc{});
        CHECK(third.samplerSlot().has_value());
        CHECK(slotResult(device, destroyed(second)) == Result::ErrorInvalidHandle);
        CHECK(slotResult(device, destroyed(third)) == Result::ErrorInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device fails sampler calls with RhiInvalidHandle for a sampler that is gone", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        Sampler sampler = created(device, SamplerDesc{});
        const bfr_sampler gone = destroyed(sampler);

        CHECK(slotResult(device, gone) == Result::ErrorInvalidHandle);
        // d1: stale destroy reported as an Error message (track B)
        bfr_destroy_sampler(device.get(), gone);
        // d1: null destroy: no message (track B)
        bfr_destroy_sampler(device.get(), bfr_sampler{});
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device destroys the samplers left when it goes, and says so", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        // A Sampler keeps its Device alive, so one is left only through bfr.h:
        // SamplerDesc{} as bfr_sampler_desc, never destroyed.
        const bfr_sampler_desc desc{};
        bfr_sampler left{};
        REQUIRE(resultOf(bfr_create_sampler(made.device.get(), &desc, &left)) == Result::Success);
    }

    const std::size_t warnings = BFR_LOGGING_ENABLED ? 1 : 0;
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
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        const CapsLimits limits = capsOf(made.instance, kFirstAdapter).limits;

        auto texture = device.createTexture({.dimension = TextureDimension::D2,
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
        Buffer upload = created(device,
                                {.size = bytesPerRow, .memory = BufferMemory::Upload, .usage = {.copySrc = true}});
        const auto mapped = upload.map();
        REQUIRE(mapped.has_value());
        std::memcpy(*mapped, kBlackWhite.data(), kBlackWhite.size());

        // Nearest, with an opaque white border: black inside, white before it.
        Sampler bordered = created(device, {.addressU = AddressMode::ClampToBorder,
                                            .addressV = AddressMode::ClampToBorder,
                                            .addressW = AddressMode::ClampToBorder,
                                            .borderColor = {1, 1, 1, 1}});
        // Linear, clamped to the edge: a quarter white inside, black before it.
        Sampler edged = created(device, {.magFilter = Filter::Linear,
                                         .minFilter = Filter::Linear,
                                         .addressU = AddressMode::ClampToEdge,
                                         .addressV = AddressMode::ClampToEdge,
                                         .addressW = AddressMode::ClampToEdge});
        constexpr std::uint64_t kBytes = std::uint64_t{kSamples} * sizeof(float);
        Buffer borderedOut = created(device, {.size = kBytes, .usage = {.copySrc = true, .storage = true}});
        Buffer edgedOut = created(device, {.size = kBytes, .usage = {.copySrc = true, .storage = true}});
        Buffer readback = created(device,
                                  {.size = 2 * kBytes, .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
        auto pipeline = device.createComputePipeline(
            {.compute = shaderCode(bfr::test::spirv::kSampleComp), .workgroupSize = {kSamples, 1, 1}});
        INFO(failureOf(pipeline));
        REQUIRE(pipeline.has_value());

        const auto textureSlots = texture->textureSlots();
        const auto borderedSlot = bordered.samplerSlot();
        const auto edgedSlot = edged.samplerSlot();
        const auto borderedTarget = storageSlot(borderedOut);
        const auto edgedTarget = storageSlot(edgedOut);
        REQUIRE(textureSlots.has_value());
        REQUIRE(borderedSlot.has_value());
        REQUIRE(edgedSlot.has_value());
        REQUIRE(borderedTarget.has_value());
        REQUIRE(edgedTarget.has_value());

        auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists, .pushConstantUpdates = 2});
        REQUIRE(pool.has_value());
        auto list = pool->beginCommands();
        REQUIRE(list.has_value());
        CHECK(device.textureBarrier(*list, *texture, TextureLayout::Undefined, TextureLayout::CopyDst, {}).has_value());
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
        CHECK(device.textureBarrier(*list, *texture, TextureLayout::CopyDst, TextureLayout::Sampled, {}).has_value());
        CHECK(device.bindPipeline(*list, *pipeline).has_value());
        for (const SamplePush& push :
             {SamplePush{.textureSlot = textureSlots->sampled, .samplerSlot = *borderedSlot, .target = *borderedTarget},
              SamplePush{.textureSlot = textureSlots->sampled, .samplerSlot = *edgedSlot, .target = *edgedTarget}}) {
            const auto bytes = std::bit_cast<std::array<std::uint8_t, sizeof(SamplePush)>>(push);
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

        const auto read = readback.map();
        REQUIRE(read.has_value());
        std::array<float, std::size_t{2} * kSamples> values{};
        std::memcpy(values.data(), *read, sizeof(values));
        using Catch::Matchers::WithinAbs;
        CHECK_THAT(values[0], WithinAbs(0.0, 0.0));
        CHECK_THAT(values[1], WithinAbs(1.0, 0.0));
        CHECK_THAT(values[2], WithinAbs(kQuarterWhite, kUnormStep));
        CHECK_THAT(values[3], WithinAbs(0.0, 0.0));
        const bfr_pipeline gonePipeline = destroyed(*pipeline);
        CHECK(bindResult(device, 0, gonePipeline) == Result::ErrorInvalidHandle);
        CHECK(slotResult(device, destroyed(bordered)) == Result::ErrorInvalidHandle);
        CHECK(slotResult(device, destroyed(edged)) == Result::ErrorInvalidHandle);
        for (Buffer* buffer : std::array{&upload, &borderedOut, &edgedOut, &readback}) {
            CHECK(slotsResult(device, destroyed(*buffer)) == Result::ErrorInvalidHandle);
        }
        CHECK(slotsResult(device, destroyed(*texture)) == Result::ErrorInvalidHandle);
        CHECK(resetResult(device, destroyed(*pool)) == Result::ErrorInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device makes a sampler with each setting its features have, and refuses one they lack with RhiUnsupported",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const CapsFeatures features = capsOf(made.instance, kFirstAdapter).features;
        constexpr std::uint32_t kAnisotropy = 4;
        constexpr float kLodBias = 0.5F;
        constexpr float kCustomGray = 0.5F;
        SamplerDesc anisotropic{};
        anisotropic.maxAnisotropy = kAnisotropy;
        SamplerDesc minimum{};
        minimum.reduction = SamplerReduction::Minimum;
        SamplerDesc mirrorClamp{};
        mirrorClamp.addressU = AddressMode::MirrorClampToEdge;
        SamplerDesc customBorder{};
        customBorder.addressU = AddressMode::ClampToBorder;
        customBorder.borderColor = {kCustomGray, kCustomGray, kCustomGray, 1};
        SamplerDesc integerBorder{};
        integerBorder.borderInteger = true;
        SamplerDesc unnormalized{};
        unnormalized.unnormalizedCoordinates = true;
        unnormalized.addressU = AddressMode::ClampToEdge;
        unnormalized.addressV = AddressMode::ClampToEdge;
        SamplerDesc biased{};
        biased.lodBias = kLodBias;
        SamplerDesc averaged{};
        averaged.lodAverage = true;

        for (const auto& [desc, has, name] :
             {std::tuple{anisotropic, features.samplerAnisotropy, "anisotropy"},
              std::tuple{minimum, features.samplerMinmax, "min reduction"},
              std::tuple{mirrorClamp, features.samplerMirrorClampToEdge, "mirror clamp to edge"},
              std::tuple{customBorder, features.samplerCustomBorderColor, "custom border"},
              std::tuple{integerBorder, features.samplerIntegerBorderColor, "integer border"},
              std::tuple{unnormalized, features.samplerUnnormalizedCoordinates, "unnormalized coordinates"},
              std::tuple{biased, features.samplerLodBias, "LOD bias"},
              std::tuple{averaged, features.samplerLodAverage, "LOD average"}}) {
            INFO(name);
            const auto sampler = made.device.createSampler(desc);
            INFO(failureOf(sampler));
            if (has) {
                CHECK(sampler.has_value());
            }
            else {
                REQUIRE_FALSE(sampler.has_value());
                CHECK(sampler.error().code == Result::ErrorUnsupported);
            }
        }
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Vulkan refuses an unnormalized-coordinate sampler it doesn't take", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    if (backend.backend != Backend::Vulkan) {
        SKIP("the rules are Vulkan's");
    }
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        if (!capsOf(made.instance, kFirstAdapter).features.samplerUnnormalizedCoordinates) {
            SKIP("the device has no unnormalized coordinates");
        }
        SamplerDesc repeating{};
        repeating.unnormalizedCoordinates = true;
        CHECK(made.device.createSampler(repeating).error().code == Result::ErrorInvalidArgument);
        SamplerDesc clamped = repeating;
        clamped.addressU = AddressMode::ClampToEdge;
        clamped.addressV = AddressMode::ClampToEdge;
        CHECK(made.device.createSampler(clamped).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}
