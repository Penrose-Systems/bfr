// What the Vulkan backend refuses of a copy: a value Vulkan's copy structs
// have no room for, which it would otherwise cut short or wrap.

#include "bfr/core/error.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/instance.h"
#include "bfr/rhi/texture.h"
#include "rhi/rhi_test_support.h"
#include "vulkan_test_support.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_container_properties.hpp>

#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace bfr::rhi;
using namespace bfr::rhi::vulkan::test;
using bfr::core::ErrorCode;

namespace {

constexpr std::uint32_t kSize = 16;
constexpr std::uint32_t kRgba8Bytes = 4;
// Past the largest int32_t, which Vulkan's offsets are.
constexpr std::uint32_t kPastInt32 = std::uint32_t{std::numeric_limits<std::int32_t>::max()} + 1;

} // namespace

TEST_CASE("Vulkan device refuses a copy whose rows are not whole texels or whose offset is past an int32_t",
          "[vulkan][gpu]") {
    std::vector<std::string> messages;
    {
        auto created = Instance::create(recordingInto(messages), kValidation);
        INFO(failureOf(created));
        REQUIRE(created.has_value());
        const std::unique_ptr<Instance> instance = std::move(*created);
        auto made = instance->createDevice(0, kNoWindow, kTestDevice);
        INFO(failureOf(made));
        REQUIRE(made.has_value());
        Device& device = **made;
        const TextureDesc desc{.dimension = TextureDimension::D2,
                               .format = Format::Rgba8Unorm,
                               .width = kSize,
                               .height = kSize,
                               .depth = 1,
                               .layers = 1,
                               .mipLevels = 1,
                               .samples = 1,
                               .usage = {.copySrc = true, .copyDst = true}};
        const auto texture = device.createTexture(desc);
        const auto other = device.createTexture(desc);
        const auto buffer = device.createBuffer({.size = std::uint64_t{kSize} * kSize * kRgba8Bytes,
                                                 .memory = BufferMemory::Upload,
                                                 .usage = {.copySrc = true, .copyDst = true}});
        REQUIRE((texture.has_value() && other.has_value() && buffer.has_value()));
        const auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        const auto list = device.beginCommands(*pool);
        REQUIRE(list.has_value());
        const BufferTextureCopy partRow{.bytesPerRow = std::uint64_t{kSize} * kRgba8Bytes + 1,
                                        .rowsPerImage = kSize,
                                        .width = kSize,
                                        .height = kSize,
                                        .depth = 1};
        const BufferTextureCopy farOff{.bytesPerRow = std::uint64_t{kSize} * kRgba8Bytes,
                                       .rowsPerImage = kSize,
                                       .x = kPastInt32,
                                       .width = 1,
                                       .height = 1,
                                       .depth = 1};
        const TextureCopy farBetween{.dstX = kPastInt32, .width = 1, .height = 1, .depth = 1};

        CHECK(device.copyBufferToTexture(*list, *buffer, *texture, std::span{&partRow, 1}).error().code ==
              ErrorCode::RhiInvalidCall);
        CHECK(device.copyTextureToBuffer(*list, *texture, *buffer, std::span{&partRow, 1}).error().code ==
              ErrorCode::RhiInvalidCall);
        CHECK(device.copyBufferToTexture(*list, *buffer, *texture, std::span{&farOff, 1}).error().code ==
              ErrorCode::RhiInvalidCall);
        CHECK(device.copyTexture(*list, *texture, *other, std::span{&farBetween, 1}).error().code ==
              ErrorCode::RhiInvalidCall);
        CHECK(device.destroyCommandList(*list).has_value());
        CHECK(device.destroyCommandPool(*pool).has_value());
        CHECK(device.destroyBuffer(*buffer).has_value());
        CHECK(device.destroyTexture(*texture).has_value());
        CHECK(device.destroyTexture(*other).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}
