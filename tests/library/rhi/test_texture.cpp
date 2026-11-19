// What every backend does with textures, through the public API alone. Each
// case runs on every backend the build has.

#include "bfr.hpp"
#include "bfr/bfr.h"
#include "bfr_test_support.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_container_properties.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <string>
#include <vector>

using namespace bfr;
using namespace bfr::test;

namespace {

constexpr std::uint32_t kWidth = 16;
constexpr std::uint32_t kHeight = 8;
constexpr std::uint32_t kLayers = 2;
constexpr std::uint32_t kMips = 2;
constexpr std::uint32_t kSlices = 2;
// The region copied: 8 by 4 texels, placed 4 texels in on each side, which is
// whole blocks of every format here.
constexpr std::uint32_t kRegionWidth = 8;
constexpr std::uint32_t kRegionHeight = 4;
constexpr std::uint32_t kInset = 4;

constexpr TextureUsage kCopies{.copySrc = true, .copyDst = true};
// A number Format has no value for.
constexpr auto kNoSuchFormat = static_cast<Format>(0xffff);

// A format the copy runs in, with the size of its texel blocks, which the
// caller knows of its own data, and the shape of texture it fills.
struct CopyCase {
    const char* name = "";
    Format format = Format::Rgba8Unorm;
    std::uint32_t blockBytes = 0;
    std::uint32_t blockSize = 0;
    TextureDimension dimension = TextureDimension::D2;
};

constexpr std::uint32_t kRgba8Bytes = 4;
constexpr std::uint32_t kBc1Bytes = 8;
constexpr std::uint32_t kBc1Size = 4;

} // namespace

TEST_CASE("Device creates and destroys a texture of each dimension", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    const TextureDesc desc = GENERATE(TextureDesc{.dimension = TextureDimension::D2,
                                                  .format = Format::Rgba8Unorm,
                                                  .width = kWidth,
                                                  .height = kHeight,
                                                  .depth = 1,
                                                  .layers = kLayers,
                                                  .mipLevels = kMips,
                                                  .samples = 1,
                                                  .usage = {.copyDst = true, .sampled = true}},
                                      TextureDesc{.dimension = TextureDimension::D1,
                                                  .format = Format::Rgba8Unorm,
                                                  .width = kWidth,
                                                  .height = 1,
                                                  .depth = 1,
                                                  .layers = 1,
                                                  .mipLevels = 1,
                                                  .samples = 1,
                                                  .usage = {.copyDst = true}},
                                      TextureDesc{.dimension = TextureDimension::D3,
                                                  .format = Format::Rgba8Unorm,
                                                  .width = kWidth,
                                                  .height = kHeight,
                                                  .depth = kSlices,
                                                  .layers = 1,
                                                  .mipLevels = 1,
                                                  .samples = 1,
                                                  .usage = {.copyDst = true}},
                                      TextureDesc{.dimension = TextureDimension::Cube,
                                                  .format = Format::Rgba8Unorm,
                                                  .width = kWidth,
                                                  .height = kWidth,
                                                  .depth = 1,
                                                  .layers = 6,
                                                  .mipLevels = 1,
                                                  .samples = 1,
                                                  .usage = {.copyDst = true, .sampled = true}},
                                      TextureDesc{.dimension = TextureDimension::D2,
                                                  .format = Format::Rgba8Unorm,
                                                  .width = kWidth,
                                                  .height = kHeight,
                                                  .depth = 1,
                                                  .layers = 1,
                                                  .mipLevels = 1,
                                                  .samples = 1,
                                                  .usage = {.colorAttachment = true}});
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        Texture texture = created(made.device, desc);

        const bfr_texture stale = destroyed(texture);
        CHECK(slotsResult(made.device, stale) == Result::ErrorInvalidHandle);
        // d1: stale destroy reported as an Error message (track B)
        bfr_destroy_texture(made.device.get(), stale);
        // d1: null destroy: no message (track B)
        bfr_destroy_texture(made.device.get(), bfr_texture{});
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device copies texels from a buffer into a texture, to another texture and back to a buffer",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    const CopyCase copy = GENERATE(CopyCase{.name = "rgba8, a layer and mip level of an array",
                                            .format = Format::Rgba8Unorm,
                                            .blockBytes = kRgba8Bytes,
                                            .blockSize = 1,
                                            .dimension = TextureDimension::D2},
                                   CopyCase{.name = "rgba8, two depth slices",
                                            .format = Format::Rgba8Unorm,
                                            .blockBytes = kRgba8Bytes,
                                            .blockSize = 1,
                                            .dimension = TextureDimension::D3},
                                   CopyCase{.name = "bc1, a layer and mip level of an array",
                                            .format = Format::Bc1Unorm,
                                            .blockBytes = kBc1Bytes,
                                            .blockSize = kBc1Size,
                                            .dimension = TextureDimension::D2});
    INFO(copy.name);
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        const auto formatCaps = formatCapsOf(made.instance, kFirstAdapter, copy.format);
        INFO(failureOf(formatCaps));
        REQUIRE(formatCaps.has_value());
        if (!formatCaps->copySrc || !formatCaps->copyDst) {
            SKIP("the adapter does not copy this format");
        }
        const bool slices = copy.dimension == TextureDimension::D3;
        const std::uint32_t depth = slices ? kSlices : 1;
        // The source is an array with mip levels, the region its second layer's
        // second level; or a 3D texture, the region two of its slices.
        Texture source = created(device, {.dimension = copy.dimension,
                                          .format = copy.format,
                                          .width = kWidth,
                                          .height = kHeight,
                                          .depth = depth,
                                          .layers = slices ? 1 : kLayers,
                                          .mipLevels = slices ? 1 : kMips,
                                          .samples = 1,
                                          .usage = kCopies});
        Texture target = created(device, {.dimension = copy.dimension,
                                          .format = copy.format,
                                          .width = kWidth,
                                          .height = kHeight,
                                          .depth = depth,
                                          .layers = 1,
                                          .mipLevels = 1,
                                          .samples = 1,
                                          .usage = kCopies});
        const std::uint32_t mip = slices ? 0 : 1;
        const std::uint32_t layer = slices ? 0 : 1;

        // Rows and offsets keep the alignment the caps state and whole blocks.
        const CapsLimits limits = capsOf(made.instance, kFirstAdapter).limits;
        const std::uint64_t rowBytes = std::uint64_t{kRegionWidth / copy.blockSize} * copy.blockBytes;
        const std::uint64_t bytesPerRow = alignUp(rowBytes,
                                                  std::lcm(std::max<std::uint64_t>(limits.copyRowPitchAlignment, 1),
                                                           copy.blockBytes));
        const std::uint64_t offset = std::lcm(std::max<std::uint64_t>(limits.copyOffsetAlignment, 1), copy.blockBytes);
        const std::uint32_t blockRows = kRegionHeight / copy.blockSize;
        const std::uint64_t size = offset + (bytesPerRow * blockRows * depth);
        Buffer upload = created(device,
                                BufferDesc{.size = size, .memory = BufferMemory::Upload, .usage = {.copySrc = true}});
        Buffer readback =
            created(device, BufferDesc{.size = size, .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
        const auto mappedUpload = upload.map();
        REQUIRE(mappedUpload.has_value());
        const std::span<std::byte> written = mappedBytes(*mappedUpload, size);
        for (std::size_t index = 0; index < written.size(); ++index) {
            written[index] = static_cast<std::byte>(static_cast<unsigned char>(index));
        }

        const BufferTextureCopy in{.bufferOffset = offset,
                                   .bytesPerRow = bytesPerRow,
                                   .rowsPerImage = kRegionHeight,
                                   .mipLevel = mip,
                                   .layer = layer,
                                   .aspect = TextureAspect::Color,
                                   .x = 0,
                                   .y = 0,
                                   .z = 0,
                                   .width = kRegionWidth,
                                   .height = kRegionHeight,
                                   .depth = depth};
        const TextureCopy across{.srcMipLevel = mip,
                                 .srcLayer = layer,
                                 .srcX = 0,
                                 .srcY = 0,
                                 .srcZ = 0,
                                 .dstMipLevel = 0,
                                 .dstLayer = 0,
                                 .dstX = kInset,
                                 .dstY = kInset,
                                 .dstZ = 0,
                                 .width = kRegionWidth,
                                 .height = kRegionHeight,
                                 .depth = depth};
        BufferTextureCopy out = in;
        out.mipLevel = 0;
        out.layer = 0;
        out.x = kInset;
        out.y = kInset;

        auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        const auto list = pool->beginCommands();
        REQUIRE(list.has_value());
        CHECK(device.textureBarrier(*list, source, TextureLayout::Undefined, TextureLayout::CopyDst, {}).has_value());
        CHECK(device.textureBarrier(*list, target, TextureLayout::Undefined, TextureLayout::CopyDst, {}).has_value());
        CHECK(device.copyBufferToTexture(*list, upload, source, std::span{&in, 1}).has_value());
        CHECK(device.textureBarrier(*list, source, TextureLayout::CopyDst, TextureLayout::CopySrc, {}).has_value());
        CHECK(device.copyTexture(*list, source, target, std::span{&across, 1}).has_value());
        CHECK(device.textureBarrier(*list, target, TextureLayout::CopyDst, TextureLayout::CopySrc, {}).has_value());
        CHECK(device.copyTextureToBuffer(*list, target, readback, std::span{&out, 1}).has_value());
        const auto token = device.submit({.queue = 0, .commandLists = std::span{&*list, 1}, .waits = {}});
        INFO(failureOf(token));
        REQUIRE(token.has_value());
        REQUIRE(device.wait(*token, kTestWait).has_value());

        const auto mappedReadback = readback.map();
        REQUIRE(mappedReadback.has_value());
        const std::span<const std::byte> read = mappedBytes(*mappedReadback, size);
        for (std::uint64_t row = 0; row < std::uint64_t{blockRows} * depth; ++row) {
            const std::uint64_t at = offset + (row * bytesPerRow);
            INFO("block row " << row);
            CHECK(std::ranges::equal(read.subspan(at, rowBytes), written.subspan(at, rowBytes)));
        }
        CHECK(resetResult(device, destroyed(*pool)) == Result::ErrorInvalidHandle);
        for (Buffer* buffer : std::array{&upload, &readback}) {
            CHECK(slotsResult(device, destroyed(*buffer)) == Result::ErrorInvalidHandle);
        }
        for (Texture* texture : std::array{&source, &target}) {
            CHECK(slotsResult(device, destroyed(*texture)) == Result::ErrorInvalidHandle);
        }
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device copies the depth and the stencil of a texture, each whole, from a buffer and back",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    const Format format = GENERATE(Format::D32FloatS8Uint, Format::D24UnormS8Uint);
    INFO(static_cast<int>(format));
    // A buffer holds 4 bytes a depth texel and 1 a stencil one. 32-bit float
    // depth outside 0 to 1, and the 8 bits above 24-bit depth, are not kept.
    constexpr std::uint64_t kDepthBytes = 4;
    constexpr std::uint32_t kDepth24Mask = 0x00FFFFFF;
    constexpr std::uint64_t kAspectAlignment = 4;
    const bool depth24 = format == Format::D24UnormS8Uint;
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        const auto formatCaps = formatCapsOf(made.instance, kFirstAdapter, format);
        INFO(failureOf(formatCaps));
        REQUIRE(formatCaps.has_value());
        if (!formatCaps->copySrc || !formatCaps->copyDst || !formatCaps->depthStencil) {
            SKIP("the adapter does not copy this format");
        }
        Texture texture = created(device, {.dimension = TextureDimension::D2,
                                           .format = format,
                                           .width = kWidth,
                                           .height = kHeight,
                                           .depth = 1,
                                           .layers = 1,
                                           .mipLevels = 1,
                                           .samples = 1,
                                           .usage = {.copySrc = true, .copyDst = true, .depthStencil = true}});
        const CapsLimits limits = capsOf(made.instance, kFirstAdapter).limits;
        const std::uint64_t rowAlignment = std::lcm(std::max<std::uint64_t>(limits.copyRowPitchAlignment, 1), 4);
        const std::uint64_t offsetAlignment = std::lcm(std::max<std::uint64_t>(limits.copyOffsetAlignment, 1),
                                                       kAspectAlignment);
        const std::uint64_t depthRow = kWidth * kDepthBytes;
        const std::uint64_t depthPitch = alignUp(depthRow, rowAlignment);
        const std::uint64_t stencilPitch = alignUp(kWidth, rowAlignment);
        const std::uint64_t depthAt = offsetAlignment;
        const std::uint64_t stencilAt = alignUp(depthAt + (depthPitch * kHeight), offsetAlignment);
        const std::uint64_t size = stencilAt + (stencilPitch * kHeight);
        Buffer upload = created(device,
                                BufferDesc{.size = size, .memory = BufferMemory::Upload, .usage = {.copySrc = true}});
        Buffer readback =
            created(device, BufferDesc{.size = size, .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
        const auto mappedUpload = upload.map();
        REQUIRE(mappedUpload.has_value());
        const std::span<std::byte> written = mappedBytes(*mappedUpload, size);
        std::ranges::fill(written, std::byte{0});
        for (std::uint32_t y = 0; y < kHeight; ++y) {
            for (std::uint32_t x = 0; x < kWidth; ++x) {
                const std::uint32_t texel = (y * kWidth) + x;
                const std::uint32_t value = depth24 ? (texel * kWidth) & kDepth24Mask
                                                    : std::bit_cast<std::uint32_t>(static_cast<float>(texel) /
                                                                                   (kWidth * kHeight));
                const auto bytes = std::bit_cast<std::array<std::byte, sizeof(value)>>(value);
                std::ranges::copy(bytes, written.subspan(depthAt + (y * depthPitch) + (x * kDepthBytes)).begin());
                written[stencilAt + (y * stencilPitch) + x] = static_cast<std::byte>(texel);
            }
        }
        const auto whole = [](std::uint64_t at, std::uint64_t pitch, TextureAspect aspect) {
            return BufferTextureCopy{.bufferOffset = at,
                                     .bytesPerRow = pitch,
                                     .rowsPerImage = kHeight,
                                     .mipLevel = 0,
                                     .layer = 0,
                                     .aspect = aspect,
                                     .x = 0,
                                     .y = 0,
                                     .z = 0,
                                     .width = kWidth,
                                     .height = kHeight,
                                     .depth = 1};
        };
        const std::array regions{whole(depthAt, depthPitch, TextureAspect::Depth),
                                 whole(stencilAt, stencilPitch, TextureAspect::Stencil)};

        auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        const auto list = pool->beginCommands();
        REQUIRE(list.has_value());
        CHECK(device.textureBarrier(*list, texture, TextureLayout::Undefined, TextureLayout::CopyDst, {}).has_value());
        CHECK(device.copyBufferToTexture(*list, upload, texture, regions).has_value());
        CHECK(device.textureBarrier(*list, texture, TextureLayout::CopyDst, TextureLayout::CopySrc, {}).has_value());
        CHECK(device.copyTextureToBuffer(*list, texture, readback, regions).has_value());
        const auto token = device.submit({.queue = 0, .commandLists = std::span{&*list, 1}, .waits = {}});
        INFO(failureOf(token));
        REQUIRE(token.has_value());
        REQUIRE(device.wait(*token, kTestWait).has_value());

        const auto mappedReadback = readback.map();
        REQUIRE(mappedReadback.has_value());
        const std::span<const std::byte> read = mappedBytes(*mappedReadback, size);
        const auto depthOf = [depth24](std::span<const std::byte> bytes, std::uint64_t at) {
            std::array<std::byte, sizeof(std::uint32_t)> texel{};
            std::ranges::copy(bytes.subspan(at, texel.size()), texel.begin());
            const auto value = std::bit_cast<std::uint32_t>(texel);
            return depth24 ? value & kDepth24Mask : value;
        };
        for (std::uint32_t y = 0; y < kHeight; ++y) {
            INFO("row " << y);
            for (std::uint32_t x = 0; x < kWidth; ++x) {
                const std::uint64_t depthTexel = depthAt + (y * depthPitch) + (x * kDepthBytes);
                const std::uint64_t stencilTexel = stencilAt + (y * stencilPitch) + x;
                CHECK(depthOf(read, depthTexel) == depthOf(written, depthTexel));
                CHECK(read[stencilTexel] == written[stencilTexel]);
            }
        }
        CHECK(resetResult(device, destroyed(*pool)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(upload)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(readback)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(texture)) == Result::ErrorInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device fails texture calls with RhiInvalidHandle for a texture or list that is gone", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        Texture texture = created(device, {.dimension = TextureDimension::D2,
                                           .format = Format::Rgba8Unorm,
                                           .width = kWidth,
                                           .height = kHeight,
                                           .depth = 1,
                                           .layers = 1,
                                           .mipLevels = 1,
                                           .samples = 1,
                                           .usage = kCopies});
        Texture goneTexture = created(device, {.dimension = TextureDimension::D2,
                                               .format = Format::Rgba8Unorm,
                                               .width = kWidth,
                                               .height = kHeight,
                                               .depth = 1,
                                               .layers = 1,
                                               .mipLevels = 1,
                                               .samples = 1,
                                               .usage = kCopies});
        const bfr_texture gone = destroyed(goneTexture);
        REQUIRE(slotsResult(device, gone) == Result::ErrorInvalidHandle);
        Buffer buffer = created(device, BufferDesc{.size = std::uint64_t{kWidth} * kHeight * kRgba8Bytes,
                                                   .memory = BufferMemory::Upload,
                                                   .usage = {.copySrc = true}});
        auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        const auto list = pool->beginCommands();
        REQUIRE(list.has_value());
        // bfr.hpp names a texture only by a live Texture, so the stale one goes
        // through bfr.h, with TextureRange{}, BufferTextureCopy{} and
        // TextureCopy{} as their C structs.
        const bfr_texture_range range{.mip_level = 0,
                                      .mip_count = BFR_REMAINING,
                                      .layer = 0,
                                      .layer_count = BFR_REMAINING};
        const bfr_buffer_texture_copy region{};
        const bfr_texture_copy between{};

        CHECK(resultOf(bfr_texture_barrier(device.get(), list->get(), gone, BFR_TEXTURE_LAYOUT_UNDEFINED,
                                           BFR_TEXTURE_LAYOUT_COPY_DST, &range)) == Result::ErrorInvalidHandle);
        CHECK(resultOf(bfr_copy_buffer_to_texture(device.get(), list->get(), buffer.get(), gone, &region, 1)) ==
              Result::ErrorInvalidHandle);
        CHECK(resultOf(bfr_copy_texture_to_buffer(device.get(), list->get(), gone, buffer.get(), &region, 1)) ==
              Result::ErrorInvalidHandle);
        CHECK(resultOf(bfr_copy_texture(device.get(), list->get(), texture.get(), gone, &between, 1)) ==
              Result::ErrorInvalidHandle);
        device.destroyCommandList(*list);
        CHECK(device.barrier(*list).error().code == Result::ErrorInvalidHandle);
        CHECK(
            device.textureBarrier(*list, texture, TextureLayout::Undefined, TextureLayout::CopyDst, {}).error().code ==
            Result::ErrorInvalidHandle);
        CHECK(resetResult(device, destroyed(*pool)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(buffer)) == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(texture)) == Result::ErrorInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device refuses a texture or a format query of a format BFR does not have", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const Format format = GENERATE(Format::Undefined, kNoSuchFormat);
        const TextureDesc desc{.dimension = TextureDimension::D2,
                               .format = format,
                               .width = kWidth,
                               .height = kHeight,
                               .depth = 1,
                               .layers = 1,
                               .mipLevels = 1,
                               .samples = 1,
                               .usage = kCopies};

        CHECK(made.device.createTexture(desc).error().code == Result::ErrorInvalidArgument);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device destroys the textures left when it goes, and says so", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        // A Texture keeps its Device alive, so one is left only through bfr.h,
        // never destroyed.
        constexpr TextureDesc kLeft{.dimension = TextureDimension::D2,
                                    .format = Format::Rgba8Unorm,
                                    .width = kWidth,
                                    .height = kHeight,
                                    .depth = 1,
                                    .layers = 1,
                                    .mipLevels = 1,
                                    .samples = 1,
                                    .usage = kCopies};
        const bfr_texture_desc desc = detail::toC(kLeft, nullptr);
        bfr_texture left{};
        REQUIRE(resultOf(bfr_create_texture(made.device.get(), &desc, &left)) == Result::Success);
    }

    const std::size_t warnings = BFR_LOGGING_ENABLED ? 1 : 0;
    REQUIRE(messages.size() == warnings);
    for (const std::string& message : messages) {
        CHECK_THAT(message, Catch::Matchers::ContainsSubstring("1 textures were left when the device went"));
    }
}

TEST_CASE("Device refuses a buffer copy of an aspect the texture's format lacks", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        constexpr std::uint64_t kBufferSize = 4096;
        Buffer buffer = created(device, BufferDesc{.size = kBufferSize,
                                                   .memory = BufferMemory::Upload,
                                                   .usage = {.copySrc = true, .copyDst = true}});
        const auto textureOf = [&device](Format format, TextureUsage usage) {
            return created(device, {.dimension = TextureDimension::D2,
                                    .format = format,
                                    .width = kWidth,
                                    .height = kHeight,
                                    .depth = 1,
                                    .layers = 1,
                                    .mipLevels = 1,
                                    .samples = 1,
                                    .usage = usage});
        };
        Texture color = textureOf(Format::Rgba8Unorm, kCopies);
        Texture depth = textureOf(Format::D32Float, {.copySrc = true, .copyDst = true, .depthStencil = true});
        auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        const auto list = pool->beginCommands();
        REQUIRE(list.has_value());
        const auto region = [](TextureAspect aspect) {
            return BufferTextureCopy{.bufferOffset = 0,
                                     .bytesPerRow = 0,
                                     .rowsPerImage = 0,
                                     .mipLevel = 0,
                                     .layer = 0,
                                     .aspect = aspect,
                                     .x = 0,
                                     .y = 0,
                                     .z = 0,
                                     .width = kWidth,
                                     .height = kHeight,
                                     .depth = 1};
        };

        for (const auto& [texture, aspect] :
             {std::pair{&color, TextureAspect::Depth}, std::pair{&depth, TextureAspect::Color},
              std::pair{&depth, TextureAspect::Stencil}}) {
            const BufferTextureCopy wrong = region(aspect);
            INFO(static_cast<int>(aspect));
            CHECK(device.copyBufferToTexture(*list, buffer, *texture, std::span{&wrong, 1}).error().code ==
                  Result::ErrorInvalidArgument);
            CHECK(device.copyTextureToBuffer(*list, *texture, buffer, std::span{&wrong, 1}).error().code ==
                  Result::ErrorInvalidArgument);
        }
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device refuses a read-only depth-stencil attachment that would clear or discard it", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        Texture depth = created(device, {.dimension = TextureDimension::D2,
                                         .format = Format::D32Float,
                                         .width = kWidth,
                                         .height = kHeight,
                                         .depth = 1,
                                         .layers = 1,
                                         .mipLevels = 1,
                                         .samples = 1,
                                         .usage = {.sampled = true, .depthStencil = true}});
        auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        const auto list = pool->beginCommands();
        REQUIRE(list.has_value());
        CHECK(device.textureBarrier(*list, depth, TextureLayout::Undefined, TextureLayout::DepthStencilRead, {})
                  .has_value());
        const auto readOnly = [&depth](LoadOp load, StoreOp store) {
            RenderingDesc rendering{};
            rendering.depthStencil.texture = depth;
            rendering.depthStencil.depthLoadOp = load;
            rendering.depthStencil.depthStoreOp = store;
            rendering.depthStencil.readOnly = true;
            return rendering;
        };

        for (const auto& [load, store] :
             {std::pair{LoadOp::Clear, StoreOp::Store}, std::pair{LoadOp::DontCare, StoreOp::Store},
              std::pair{LoadOp::Load, StoreOp::DontCare}}) {
            INFO(static_cast<int>(load) << ' ' << static_cast<int>(store));
            CHECK(device.beginRendering(*list, {}, readOnly(load, store)).error().code == Result::ErrorInvalidArgument);
        }
        CHECK(device.beginRendering(*list, {}, readOnly(LoadOp::Load, StoreOp::Store)).has_value());
        CHECK(device.endRendering(*list).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device refuses resolveTexture with RhiUnsupported where the caps say it resolves only as rendering ends",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        if (capsOf(made.instance, kFirstAdapter).resolve.texture) {
            SKIP("the device resolves outside a rendering");
        }
        const Device& device = made.device;
        const auto texture = [&device](std::uint32_t samples) {
            return created(device, {.dimension = TextureDimension::D2,
                                    .format = Format::Rgba8Unorm,
                                    .width = kWidth,
                                    .height = kHeight,
                                    .depth = 1,
                                    .layers = 1,
                                    .mipLevels = 1,
                                    .samples = samples,
                                    .usage = {.copySrc = true, .copyDst = true, .colorAttachment = true}});
        };
        constexpr std::uint32_t kFourSamples = 4;
        Texture multisampled = texture(kFourSamples);
        Texture single = texture(1);
        auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        const auto list = pool->beginCommands();
        REQUIRE(list.has_value());
        CHECK(device.resolveTexture(*list, multisampled, 0, 0, single, 0, 0).error().code == Result::ErrorUnsupported);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device refuses an attachment drawing to layers past the texture's, or to none", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        Texture color = created(device, {.dimension = TextureDimension::D2,
                                         .format = Format::Rgba8Unorm,
                                         .width = kWidth,
                                         .height = kHeight,
                                         .depth = 1,
                                         .layers = kLayers,
                                         .mipLevels = 1,
                                         .samples = 1,
                                         .usage = {.colorAttachment = true}});
        auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        const auto list = pool->beginCommands();
        REQUIRE(list.has_value());
        CHECK(
            device.textureBarrier(*list, color, TextureLayout::Undefined, TextureLayout::ColorTarget, {}).has_value());
        const auto drawing = [&color](std::uint32_t layer, std::uint32_t layerCount) {
            RenderingDesc rendering{};
            rendering.colorAttachmentCount = 1;
            rendering.colorAttachments.at(0).texture = color;
            rendering.colorAttachments.at(0).layer = layer;
            rendering.colorAttachments.at(0).layerCount = layerCount;
            return rendering;
        };

        for (const auto& [layer, layerCount] :
             {std::pair{0U, kLayers + 1}, std::pair{1U, kLayers}, std::pair{kLayers, 1U}, std::pair{0U, 0U}}) {
            INFO(layer << ' ' << layerCount);
            CHECK(device.beginRendering(*list, {}, drawing(layer, layerCount)).error().code ==
                  Result::ErrorInvalidArgument);
        }
        CHECK(device.beginRendering(*list, {}, drawing(1, kLayers - 1)).has_value());
        CHECK(device.endRendering(*list).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}
