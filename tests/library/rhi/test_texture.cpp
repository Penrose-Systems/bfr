// What every backend does with textures, through the interface alone. Each
// case runs on every backend the build has.

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/instance.h"
#include "bfr/rhi/texture.h"
#include "rhi_test_support.h"

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
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace bfr::rhi;
using namespace bfr::rhi::test;
using bfr::core::ErrorCode;

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

// An Instance and a device on its first adapter, presenting nowhere.
struct TextureDevice {
    explicit TextureDevice(const TestInstanceDesc& desc) {
        auto created = createInstance(desc, kValidation);
        INFO(failureOf(created));
        REQUIRE(created.has_value());
        instance = std::move(*created);
        auto made = instance->createDevice(0, kNoWindow, {});
        INFO(failureOf(made));
        REQUIRE(made.has_value());
        device = std::move(*made);
    }

    std::unique_ptr<Instance> instance;
    std::unique_ptr<Device> device;
};

TextureHandle created(Device& device, const TextureDesc& desc) {
    const auto texture = device.createTexture(desc);
    INFO(failureOf(texture));
    REQUIRE(texture.has_value());
    return *texture;
}

BufferHandle created(Device& device, const BufferDesc& desc) {
    const auto buffer = device.createBuffer(desc);
    INFO(failureOf(buffer));
    REQUIRE(buffer.has_value());
    return *buffer;
}

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

std::uint64_t alignUp(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

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
        const TextureDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        const TextureHandle texture = created(device, desc);

        CHECK(device.destroyTexture(texture).has_value());
        CHECK(device.destroyTexture(texture).error().code == ErrorCode::RhiInvalidHandle);
        CHECK(device.destroyTexture({}).has_value());
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
        const TextureDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        const auto formatCaps = made.instance->formatCaps(0, copy.format);
        REQUIRE(formatCaps.has_value());
        if (!formatCaps->copySrc || !formatCaps->copyDst) {
            SKIP("the adapter does not copy this format");
        }
        const bool slices = copy.dimension == TextureDimension::D3;
        const std::uint32_t depth = slices ? kSlices : 1;
        // The source is an array with mip levels, the region its second layer's
        // second level; or a 3D texture, the region two of its slices.
        const TextureHandle source = created(device, {.dimension = copy.dimension,
                                                      .format = copy.format,
                                                      .width = kWidth,
                                                      .height = kHeight,
                                                      .depth = depth,
                                                      .layers = slices ? 1 : kLayers,
                                                      .mipLevels = slices ? 1 : kMips,
                                                      .samples = 1,
                                                      .usage = kCopies});
        const TextureHandle target = created(device, {.dimension = copy.dimension,
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
        const Limits& limits = made.instance->adapters()[0].caps.limits;
        const std::uint64_t rowBytes = std::uint64_t{kRegionWidth / copy.blockSize} * copy.blockBytes;
        const std::uint64_t bytesPerRow = alignUp(rowBytes,
                                                  std::lcm(std::max<std::uint64_t>(limits.copyRowPitchAlignment, 1),
                                                           copy.blockBytes));
        const std::uint64_t offset = std::lcm(std::max<std::uint64_t>(limits.copyOffsetAlignment, 1), copy.blockBytes);
        const std::uint32_t blockRows = kRegionHeight / copy.blockSize;
        const std::uint64_t size = offset + (bytesPerRow * blockRows * depth);
        const BufferHandle upload =
            created(device, BufferDesc{.size = size, .memory = BufferMemory::Upload, .usage = {.copySrc = true}});
        const BufferHandle readback =
            created(device, BufferDesc{.size = size, .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
        const auto written = device.mapBuffer(upload);
        REQUIRE(written.has_value());
        for (std::size_t index = 0; index < written->size(); ++index) {
            (*written)[index] = static_cast<std::byte>(static_cast<unsigned char>(index));
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

        const auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        const auto list = device.beginCommands(*pool);
        REQUIRE(list.has_value());
        CHECK(device.textureBarrier(*list, source, TextureLayout::Undefined, TextureLayout::CopyDst).has_value());
        CHECK(device.textureBarrier(*list, target, TextureLayout::Undefined, TextureLayout::CopyDst).has_value());
        CHECK(device.copyBufferToTexture(*list, upload, source, std::span{&in, 1}).has_value());
        CHECK(device.textureBarrier(*list, source, TextureLayout::CopyDst, TextureLayout::CopySrc).has_value());
        CHECK(device.copyTexture(*list, source, target, std::span{&across, 1}).has_value());
        CHECK(device.textureBarrier(*list, target, TextureLayout::CopyDst, TextureLayout::CopySrc).has_value());
        CHECK(device.copyTextureToBuffer(*list, target, readback, std::span{&out, 1}).has_value());
        const auto token = device.submit({.queue = 0, .commandLists = std::span{&*list, 1}, .waits = {}});
        INFO(failureOf(token));
        REQUIRE(token.has_value());
        REQUIRE(device.wait(*token).has_value());

        const auto read = device.mapBuffer(readback);
        REQUIRE(read.has_value());
        for (std::uint64_t row = 0; row < std::uint64_t{blockRows} * depth; ++row) {
            const std::uint64_t at = offset + (row * bytesPerRow);
            INFO("block row " << row);
            CHECK(std::ranges::equal(read->subspan(at, rowBytes), written->subspan(at, rowBytes)));
        }
        CHECK(device.destroyCommandPool(*pool).has_value());
        for (const BufferHandle buffer : {upload, readback}) {
            CHECK(device.destroyBuffer(buffer).has_value());
        }
        for (const TextureHandle texture : {source, target}) {
            CHECK(device.destroyTexture(texture).has_value());
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
        const TextureDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        const auto formatCaps = made.instance->formatCaps(0, format);
        REQUIRE(formatCaps.has_value());
        if (!formatCaps->copySrc || !formatCaps->copyDst || !formatCaps->depthStencil) {
            SKIP("the adapter does not copy this format");
        }
        const TextureHandle texture = created(device,
                                              {.dimension = TextureDimension::D2,
                                               .format = format,
                                               .width = kWidth,
                                               .height = kHeight,
                                               .depth = 1,
                                               .layers = 1,
                                               .mipLevels = 1,
                                               .samples = 1,
                                               .usage = {.copySrc = true, .copyDst = true, .depthStencil = true}});
        const Limits& limits = made.instance->adapters()[0].caps.limits;
        const std::uint64_t rowAlignment = std::lcm(std::max<std::uint64_t>(limits.copyRowPitchAlignment, 1), 4);
        const std::uint64_t offsetAlignment = std::lcm(std::max<std::uint64_t>(limits.copyOffsetAlignment, 1),
                                                       kAspectAlignment);
        const std::uint64_t depthRow = kWidth * kDepthBytes;
        const std::uint64_t depthPitch = alignUp(depthRow, rowAlignment);
        const std::uint64_t stencilPitch = alignUp(kWidth, rowAlignment);
        const std::uint64_t depthAt = offsetAlignment;
        const std::uint64_t stencilAt = alignUp(depthAt + (depthPitch * kHeight), offsetAlignment);
        const std::uint64_t size = stencilAt + (stencilPitch * kHeight);
        const BufferHandle upload =
            created(device, BufferDesc{.size = size, .memory = BufferMemory::Upload, .usage = {.copySrc = true}});
        const BufferHandle readback =
            created(device, BufferDesc{.size = size, .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
        const auto written = device.mapBuffer(upload);
        REQUIRE(written.has_value());
        std::ranges::fill(*written, std::byte{0});
        for (std::uint32_t y = 0; y < kHeight; ++y) {
            for (std::uint32_t x = 0; x < kWidth; ++x) {
                const std::uint32_t texel = (y * kWidth) + x;
                const std::uint32_t value = depth24 ? (texel * kWidth) & kDepth24Mask
                                                    : std::bit_cast<std::uint32_t>(static_cast<float>(texel) /
                                                                                   (kWidth * kHeight));
                const auto bytes = std::bit_cast<std::array<std::byte, sizeof(value)>>(value);
                std::ranges::copy(bytes, written->subspan(depthAt + (y * depthPitch) + (x * kDepthBytes)).begin());
                (*written)[stencilAt + (y * stencilPitch) + x] = static_cast<std::byte>(texel);
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

        const auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        const auto list = device.beginCommands(*pool);
        REQUIRE(list.has_value());
        CHECK(device.textureBarrier(*list, texture, TextureLayout::Undefined, TextureLayout::CopyDst).has_value());
        CHECK(device.copyBufferToTexture(*list, upload, texture, regions).has_value());
        CHECK(device.textureBarrier(*list, texture, TextureLayout::CopyDst, TextureLayout::CopySrc).has_value());
        CHECK(device.copyTextureToBuffer(*list, texture, readback, regions).has_value());
        const auto token = device.submit({.queue = 0, .commandLists = std::span{&*list, 1}, .waits = {}});
        INFO(failureOf(token));
        REQUIRE(token.has_value());
        REQUIRE(device.wait(*token).has_value());

        const auto read = device.mapBuffer(readback);
        REQUIRE(read.has_value());
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
                CHECK(depthOf(*read, depthTexel) == depthOf(*written, depthTexel));
                CHECK((*read)[stencilTexel] == (*written)[stencilTexel]);
            }
        }
        CHECK(device.destroyCommandPool(*pool).has_value());
        CHECK(device.destroyBuffer(upload).has_value());
        CHECK(device.destroyBuffer(readback).has_value());
        CHECK(device.destroyTexture(texture).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device fails texture calls with RhiInvalidHandle for a texture or list that is gone", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const TextureDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        const TextureHandle texture = created(device, {.dimension = TextureDimension::D2,
                                                       .format = Format::Rgba8Unorm,
                                                       .width = kWidth,
                                                       .height = kHeight,
                                                       .depth = 1,
                                                       .layers = 1,
                                                       .mipLevels = 1,
                                                       .samples = 1,
                                                       .usage = kCopies});
        const TextureHandle gone = created(device, {.dimension = TextureDimension::D2,
                                                    .format = Format::Rgba8Unorm,
                                                    .width = kWidth,
                                                    .height = kHeight,
                                                    .depth = 1,
                                                    .layers = 1,
                                                    .mipLevels = 1,
                                                    .samples = 1,
                                                    .usage = kCopies});
        REQUIRE(device.destroyTexture(gone).has_value());
        const BufferHandle buffer = created(device, BufferDesc{.size = std::uint64_t{kWidth} * kHeight * kRgba8Bytes,
                                                               .memory = BufferMemory::Upload,
                                                               .usage = {.copySrc = true}});
        const auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        const auto list = device.beginCommands(*pool);
        REQUIRE(list.has_value());
        const BufferTextureCopy region{};
        const TextureCopy between{};

        CHECK(device.textureBarrier(*list, gone, TextureLayout::Undefined, TextureLayout::CopyDst).error().code ==
              ErrorCode::RhiInvalidHandle);
        CHECK(device.copyBufferToTexture(*list, buffer, gone, std::span{&region, 1}).error().code ==
              ErrorCode::RhiInvalidHandle);
        CHECK(device.copyTextureToBuffer(*list, gone, buffer, std::span{&region, 1}).error().code ==
              ErrorCode::RhiInvalidHandle);
        CHECK(device.copyTexture(*list, texture, gone, std::span{&between, 1}).error().code ==
              ErrorCode::RhiInvalidHandle);
        CHECK(device.destroyCommandList(*list).has_value());
        CHECK(device.textureBarrier(*list, texture, TextureLayout::Undefined, TextureLayout::CopyDst).error().code ==
              ErrorCode::RhiInvalidHandle);
        CHECK(device.destroyCommandPool(*pool).has_value());
        CHECK(device.destroyBuffer(buffer).has_value());
        CHECK(device.destroyTexture(texture).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device destroys the textures left when it goes, and says so", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const TextureDevice made{recordingInto(backend, messages)};
        static_cast<void>(created(*made.device, {.dimension = TextureDimension::D2,
                                                 .format = Format::Rgba8Unorm,
                                                 .width = kWidth,
                                                 .height = kHeight,
                                                 .depth = 1,
                                                 .layers = 1,
                                                 .mipLevels = 1,
                                                 .samples = 1,
                                                 .usage = kCopies}));
    }

    const std::size_t warnings = bfr::core::kLoggingEnabled ? 1 : 0;
    REQUIRE(messages.size() == warnings);
    for (const std::string& message : messages) {
        CHECK_THAT(message, Catch::Matchers::ContainsSubstring("1 textures were left when the device went"));
    }
}
