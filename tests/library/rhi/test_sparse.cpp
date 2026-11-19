// What every backend does with sparse buffers and textures, through the
// interface alone. Each case runs on every backend the build has, and makes
// only what the adapter's caps list.

#include "bfr.hpp"
#include "bfr/bfr.h"
#include "bfr_test_support.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_container_properties.hpp>

#include <algorithm>
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

// Room for the few memories and regions a case maps.
constexpr std::uint32_t kSparseMemories = 2;
constexpr std::uint32_t kTileMappings = 4;
// Bytes of each test buffer: enough for a few tiles of any size a backend has.
constexpr std::uint64_t kSparseBufferBytes = std::uint64_t{1} << 20U;
// A sparse texture big enough to have tiles below its tail.
constexpr std::uint32_t kSparseSide = 512;
constexpr std::uint32_t kSparseMips = 10;
constexpr std::uint32_t kRgba8Bytes = 4;
constexpr std::uint8_t kPattern = 0x5A;

// An Instance and a device on its first adapter, presenting nowhere, with
// room for sparse memories and mappings.
struct SparseDevice {
    explicit SparseDevice(const InstanceDesc& desc) {
        auto created = createInstance(desc, kValidation);
        INFO(failureOf(created));
        REQUIRE(created.has_value());
        instance = std::move(*created);
        adapterCaps = capsOf(instance, kFirstAdapter);
        queueFamily = testQueueFamily(instance, kFirstAdapter);
        auto made = instance.createDevice(
            testDeviceWith(instance, kFirstAdapter, kNoWindow, [](DeviceDesc& deviceDesc) {
                deviceDesc.maxSparseMemories = kSparseMemories;
                deviceDesc.maxTileMappings = kTileMappings;
            }));
        INFO(failureOf(made));
        REQUIRE(made.has_value());
        device = std::move(*made);
    }

    [[nodiscard]] const Caps& caps() const { return adapterCaps; }

    // The family of the device's queue 0.
    [[nodiscard]] const QueueFamily& family() const { return queueFamily; }

    // The tile size a resource is made with: the backend's first, or 0 where
    // it sets them itself.
    [[nodiscard]] std::uint32_t tileBytes() const {
        return caps().sparse.tileByteCount == 0 ? 0 : caps().sparse.tileBytes.front();
    }

    Instance instance;
    Caps adapterCaps;
    QueueFamily queueFamily;
    Device device;
};

// Submits list on queue 0 after waits and blocks until it has run.
void run(const Device& device, CommandList list, std::span<const Token> waits) {
    const auto sent = device.submit({.queue = 0, .commandLists = std::span{&list, 1}, .waits = waits});
    INFO(failureOf(sent));
    REQUIRE(sent.has_value());
    REQUIRE(device.wait(*sent, kTestWait).has_value());
}

} // namespace

TEST_CASE("Device maps a sparse buffer's tile to memory, copies through it, and unmaps it", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const SparseDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        if (!made.caps().sparse.buffers || !made.family().sparse) {
            SKIP("the adapter maps no sparse buffers on its device's queue");
        }
        auto sparse = device.createBuffer({.size = kSparseBufferBytes,
                                           .memory = BufferMemory::Device,
                                           .usage = {.copySrc = true, .copyDst = true},
                                           .sparse = true,
                                           .sparseAliased = false,
                                           .tileBytes = made.tileBytes()});
        INFO(failureOf(sparse));
        REQUIRE(sparse.has_value());
        const auto tiling = sparse->bufferTiling();
        INFO(failureOf(tiling));
        REQUIRE(tiling.has_value());
        REQUIRE(tiling->tileBytes != 0);
        REQUIRE(tiling->tileBytes <= kSparseBufferBytes);
        // A sparse buffer has its address before any memory is mapped to it.
        if (made.caps().features.bufferAddress) {
            const auto address = sparse->bufferAddress();
            INFO(failureOf(address));
            REQUIRE(address.has_value());
            CHECK(*address != 0);
        }
        const std::uint64_t tile = tiling->tileBytes;
        constexpr std::uint64_t kMemoryTiles = 2;
        auto memory = device.createSparseMemory(
            {.size = kMemoryTiles * tile, .tileBytes = made.tileBytes(), .texture = {}, .buffer = *sparse});
        INFO(failureOf(memory));
        REQUIRE(memory.has_value());

        // The buffer's first tile on the memory's second.
        const std::array mapped{BufferTileRegion{.mapping = TileMapping::Map, .tile = 0, .tiles = 1, .memoryTile = 1}};
        const auto bound = device.updateTileMappings({.queue = 0,
                                                      .waits = {},
                                                      .memory = *memory,
                                                      .texture = {},
                                                      .textureRegions = {},
                                                      .buffer = *sparse,
                                                      .bufferRegions = mapped});
        INFO(failureOf(bound));
        REQUIRE(bound.has_value());

        auto upload = device.createBuffer({.size = tile, .memory = BufferMemory::Upload, .usage = {.copySrc = true}});
        REQUIRE(upload.has_value());
        auto readback = device.createBuffer(
            {.size = tile, .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
        REQUIRE(readback.has_value());
        const auto written = upload->map();
        REQUIRE(written.has_value());
        std::ranges::fill(mappedBytes(*written, tile), std::byte{kPattern});

        auto commands = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(commands.has_value());
        auto list = commands->beginCommands();
        REQUIRE(list.has_value());
        const std::array copy{BufferCopy{.srcOffset = 0, .dstOffset = 0, .size = tile}};
        CHECK(device.copyBuffer(*list, *upload, *sparse, copy).has_value());
        CHECK(device.barrier(*list).has_value());
        CHECK(device.copyBuffer(*list, *sparse, *readback, copy).has_value());
        run(device, *list, std::span{&*bound, 1});

        const auto read = readback->map();
        REQUIRE(read.has_value());
        CHECK(std::ranges::all_of(mappedBytes(*read, tile),
                                  [](std::byte value) { return value == std::byte{kPattern}; }));

        const std::array unmapped{BufferTileRegion{.mapping = TileMapping::Unmap, .tile = 0, .tiles = 1}};
        const auto unbound = device.updateTileMappings({.queue = 0,
                                                        .waits = {},
                                                        .memory = {},
                                                        .texture = {},
                                                        .textureRegions = {},
                                                        .buffer = *sparse,
                                                        .bufferRegions = unmapped});
        INFO(failureOf(unbound));
        REQUIRE(unbound.has_value());
        REQUIRE(device.wait(*unbound, kTestWait).has_value());

        // bfr.h's destroy calls return nothing, so each destroy is checked by
        // the destroyed object's next use being refused. The memory is then
        // destroyed again, stale, and a null one, through bfr.h.
        const CommandPoolRef destroyedCommands = *commands;
        *commands = CommandPool{};
        CHECK(resetResult(device, destroyedCommands.get()) == Result::ErrorInvalidHandle);
        const BufferRef destroyedReadback = *readback;
        *readback = Buffer{};
        CHECK(slotsResult(device, destroyedReadback.get()) == Result::ErrorInvalidHandle);
        const BufferRef destroyedUpload = *upload;
        *upload = Buffer{};
        CHECK(slotsResult(device, destroyedUpload.get()) == Result::ErrorInvalidHandle);
        const SparseMemoryRef destroyedMemory = *memory;
        *memory = SparseMemory{};
        CHECK(device
                  .updateTileMappings({.queue = 0,
                                       .waits = {},
                                       .memory = destroyedMemory,
                                       .texture = {},
                                       .textureRegions = {},
                                       .buffer = {},
                                       .bufferRegions = {}})
                  .error()
                  .code == Result::ErrorInvalidHandle);
        // d1: stale destroy reported as an Error message (track B)
        bfr_destroy_sparse_memory(device.get(), destroyedMemory.get());
        // d1: null destroy: no message (track B)
        bfr_destroy_sparse_memory(device.get(), SparseMemoryRef{}.get());
        const BufferRef destroyedSparse = *sparse;
        *sparse = Buffer{};
        CHECK(slotsResult(device, destroyedSparse.get()) == Result::ErrorInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device gives a sparse texture's tiling, maps a tile and its tail, and copies through the tile",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const SparseDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        if (!made.caps().sparse.textures2d || !made.family().sparse) {
            SKIP("the adapter maps no sparse 2D textures on its device's queue");
        }
        const TextureUsage usage{.copySrc = true, .copyDst = true, .sampled = true};
        const auto format = device.sparseFormat({.format = Format::Rgba8Unorm,
                                                 .dimension = TextureDimension::D2,
                                                 .samples = 1,
                                                 .usage = usage,
                                                 .aspect = TextureAspect::Color,
                                                 .tileBytes = made.tileBytes()});
        INFO(failureOf(format));
        REQUIRE(format.has_value());
        if (!format->supported) {
            SKIP("the adapter makes no sparse Rgba8Unorm textures");
        }
        auto texture = device.createTexture({.dimension = TextureDimension::D2,
                                             .format = Format::Rgba8Unorm,
                                             .width = kSparseSide,
                                             .height = kSparseSide,
                                             .depth = 1,
                                             .layers = 1,
                                             .mipLevels = kSparseMips,
                                             .samples = 1,
                                             .usage = usage,
                                             .sparse = true,
                                             .sparseAliased = false,
                                             .tileBytes = made.tileBytes()});
        INFO(failureOf(texture));
        REQUIRE(texture.has_value());
        const auto tiling = texture->textureTiling(TextureAspect::Color);
        INFO(failureOf(tiling));
        REQUIRE(tiling.has_value());
        CHECK(tiling->shape.width == format->shape.width);
        CHECK(tiling->shape.height == format->shape.height);
        REQUIRE(tiling->tileBytes != 0);
        REQUIRE(tiling->shape.width != 0);
        REQUIRE(tiling->shape.height != 0);
        REQUIRE(tiling->firstTailMip > 0);
        const std::uint64_t tailTiles = (tiling->tailBytes + tiling->tileBytes - 1) / tiling->tileBytes;
        const std::uint64_t memoryTiles = 1 + tailTiles;
        auto memory = device.createSparseMemory({.size = memoryTiles * tiling->tileBytes,
                                                 .tileBytes = made.tileBytes(),
                                                 .texture = *texture,
                                                 .buffer = {}});
        INFO(failureOf(memory));
        REQUIRE(memory.has_value());

        // Mip 0's first tile on the memory's first, and the tail after it.
        std::vector<TextureTileRegion> regions{
            {.mapping = TileMapping::Map, .width = 1, .height = 1, .depth = 1, .memoryTile = 0}};
        if (tailTiles != 0) {
            regions.push_back({.mapping = TileMapping::Map,
                               .mipLevel = tiling->firstTailMip,
                               .tailTile = 0,
                               .tailTiles = tailTiles,
                               .memoryTile = 1});
        }
        const auto bound = device.updateTileMappings({.queue = 0,
                                                      .waits = {},
                                                      .memory = *memory,
                                                      .texture = *texture,
                                                      .textureRegions = regions,
                                                      .buffer = {},
                                                      .bufferRegions = {}});
        INFO(failureOf(bound));
        REQUIRE(bound.has_value());

        const std::uint32_t width = tiling->shape.width;
        const std::uint32_t height = tiling->shape.height;
        const std::uint64_t bytes = std::uint64_t{width} * height * kRgba8Bytes;
        auto upload = device.createBuffer({.size = bytes, .memory = BufferMemory::Upload, .usage = {.copySrc = true}});
        REQUIRE(upload.has_value());
        auto readback = device.createBuffer(
            {.size = bytes, .memory = BufferMemory::Readback, .usage = {.copyDst = true}});
        REQUIRE(readback.has_value());
        const auto written = upload->map();
        REQUIRE(written.has_value());
        std::ranges::fill(mappedBytes(*written, bytes), std::byte{kPattern});

        auto commands = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(commands.has_value());
        auto list = commands->beginCommands();
        REQUIRE(list.has_value());
        const std::array copy{BufferTextureCopy{.bytesPerRow = std::uint64_t{width} * kRgba8Bytes,
                                                .rowsPerImage = height,
                                                .width = width,
                                                .height = height,
                                                .depth = 1}};
        CHECK(device.textureBarrier(*list, *texture, TextureLayout::Undefined, TextureLayout::CopyDst, {}).has_value());
        CHECK(device.copyBufferToTexture(*list, *upload, *texture, copy).has_value());
        CHECK(device.textureBarrier(*list, *texture, TextureLayout::CopyDst, TextureLayout::CopySrc, {}).has_value());
        CHECK(device.copyTextureToBuffer(*list, *texture, *readback, copy).has_value());
        run(device, *list, std::span{&*bound, 1});

        const auto read = readback->map();
        REQUIRE(read.has_value());
        CHECK(std::ranges::all_of(mappedBytes(*read, bytes),
                                  [](std::byte value) { return value == std::byte{kPattern}; }));

        // bfr.h's destroy calls return nothing, so each destroy is checked by
        // the destroyed object's next use being refused.
        const CommandPoolRef destroyedCommands = *commands;
        *commands = CommandPool{};
        CHECK(resetResult(device, destroyedCommands.get()) == Result::ErrorInvalidHandle);
        const BufferRef destroyedReadback = *readback;
        *readback = Buffer{};
        CHECK(slotsResult(device, destroyedReadback.get()) == Result::ErrorInvalidHandle);
        const BufferRef destroyedUpload = *upload;
        *upload = Buffer{};
        CHECK(slotsResult(device, destroyedUpload.get()) == Result::ErrorInvalidHandle);
        const TextureRef destroyedTexture = *texture;
        *texture = Texture{};
        CHECK(slotsResult(device, destroyedTexture.get()) == Result::ErrorInvalidHandle);
        const SparseMemoryRef destroyedMemory = *memory;
        *memory = SparseMemory{};
        CHECK(device
                  .updateTileMappings({.queue = 0,
                                       .waits = {},
                                       .memory = destroyedMemory,
                                       .texture = {},
                                       .textureRegions = {},
                                       .buffer = {},
                                       .bufferRegions = {}})
                  .error()
                  .code == Result::ErrorInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device refuses sparse calls on resources that are not sparse, and mappings it can't make",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const SparseDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        auto plain = device.createBuffer(
            {.size = kSparseBufferBytes, .memory = BufferMemory::Device, .usage = {.copyDst = true}});
        REQUIRE(plain.has_value());
        CHECK(plain->bufferTiling().error().code == Result::ErrorInvalidArgument);
        CHECK(device
                  .updateTileMappings({.queue = 1,
                                       .waits = {},
                                       .memory = {},
                                       .texture = {},
                                       .textureRegions = {},
                                       .buffer = {},
                                       .bufferRegions = {}})
                  .error()
                  .code == Result::ErrorInvalidArgument);
        // bfr.h's destroy calls return nothing, so each destroy is checked by
        // the destroyed buffer's next use being refused.
        const BufferRef destroyedPlain = *plain;
        *plain = Buffer{};
        CHECK(slotsResult(device, destroyedPlain.get()) == Result::ErrorInvalidHandle);

        if (!made.caps().sparse.buffers || !made.family().sparse) {
            SKIP("the adapter maps no sparse buffers on its device's queue");
        }
        auto sparse = device.createBuffer({.size = kSparseBufferBytes,
                                           .memory = BufferMemory::Device,
                                           .usage = {.copyDst = true},
                                           .sparse = true,
                                           .sparseAliased = false,
                                           .tileBytes = made.tileBytes()});
        REQUIRE(sparse.has_value());
        // A region that maps needs memory to map to.
        const std::array mapsNowhere{BufferTileRegion{.mapping = TileMapping::Map, .tile = 0, .tiles = 1}};
        CHECK(device
                  .updateTileMappings({.queue = 0,
                                       .waits = {},
                                       .memory = {},
                                       .texture = {},
                                       .textureRegions = {},
                                       .buffer = *sparse,
                                       .bufferRegions = mapsNowhere})
                  .error()
                  .code == Result::ErrorInvalidArgument);
        // More regions than the device was opened with.
        const std::vector<BufferTileRegion> tooMany(kTileMappings + 1,
                                                    BufferTileRegion{.mapping = TileMapping::Unmap, .tiles = 1});
        CHECK(device
                  .updateTileMappings({.queue = 0,
                                       .waits = {},
                                       .memory = {},
                                       .texture = {},
                                       .textureRegions = {},
                                       .buffer = *sparse,
                                       .bufferRegions = tooMany})
                  .error()
                  .code == Result::ErrorInvalidArgument);
        const BufferRef destroyedSparse = *sparse;
        *sparse = Buffer{};
        CHECK(slotsResult(device, destroyedSparse.get()) == Result::ErrorInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}
