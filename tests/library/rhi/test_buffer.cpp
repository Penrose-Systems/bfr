// What every backend does with buffers, through the public API alone. Each case
// runs on every backend the build has.

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
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

using namespace bfr;
using namespace bfr::test;

namespace {

constexpr std::uint64_t kSize = 4096;
constexpr std::byte kPattern{0x5A};

constexpr BufferDesc kDeviceBuffer{.size = kSize,
                                   .memory = BufferMemory::Device,
                                   .usage = {.copyDst = true, .storage = true}};
constexpr BufferDesc kUploadBuffer{.size = kSize, .memory = BufferMemory::Upload, .usage = {.copySrc = true}};
constexpr BufferDesc kReadbackBuffer{.size = kSize, .memory = BufferMemory::Readback, .usage = {.copyDst = true}};

// What bfr_map_buffer says of a buffer handle, which may be stale: bfr.hpp
// names a buffer only by a live Buffer.
Result mapResult(const Device& device, bfr_buffer buffer) {
    void* data = nullptr;
    return resultOf(bfr_map_buffer(device.get(), buffer, &data));
}

} // namespace

TEST_CASE("Device maps upload and readback buffers, whose bytes the CPU writes and reads", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    const BufferDesc desc = GENERATE(kUploadBuffer, kReadbackBuffer);
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        Buffer buffer = created(made.device, desc);
        auto mapped = buffer.map();
        INFO(failureOf(mapped));
        REQUIRE(mapped.has_value());
        const std::span<std::byte> bytes = mappedBytes(*mapped, kSize);
        std::ranges::fill(bytes, kPattern);
        CHECK(std::ranges::all_of(bytes, [](std::byte value) { return value == kPattern; }));
        CHECK(slotsResult(made.device, destroyed(buffer)) == Result::ErrorInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device refuses to map a buffer in device memory", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        Buffer buffer = created(made.device, kDeviceBuffer);

        CHECK(buffer.map().error().code == Result::ErrorInvalidArgument);
        CHECK(slotsResult(made.device, destroyed(buffer)) == Result::ErrorInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device fails with RhiInvalidHandle on a destroyed buffer, and destroying a null one does nothing",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        Buffer buffer = created(made.device, kUploadBuffer);
        const bfr_buffer stale = destroyed(buffer);
        REQUIRE(slotsResult(made.device, stale) == Result::ErrorInvalidHandle);

        CHECK(mapResult(made.device, stale) == Result::ErrorInvalidHandle);
        // d1: stale destroy reported as an Error message (track B)
        bfr_destroy_buffer(made.device.get(), stale);
        // d1: null destroy: no message (track B)
        bfr_destroy_buffer(made.device.get(), bfr_buffer{});
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device ends a buffer handle destroyed after a submit at once, and the wait still passes",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        Buffer buffer = created(device, kDeviceBuffer);
        auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        const auto commands = pool->beginCommands();
        REQUIRE(commands.has_value());
        const CommandList list = *commands;
        const auto token = device.submit({.queue = 0, .commandLists = std::span{&list, 1}, .waits = {}});
        INFO(failureOf(token));
        REQUIRE(token.has_value());

        const bfr_buffer stale = destroyed(buffer);
        CHECK(slotsResult(device, stale) == Result::ErrorInvalidHandle);
        CHECK(mapResult(device, stale) == Result::ErrorInvalidHandle);
        CHECK(device.wait(*token, kTestWait).has_value());
        CHECK(resetResult(device, destroyed(*pool)) == Result::ErrorInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device copies bytes through device memory to readback, in one list or across a token wait",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    const bool acrossSubmits = GENERATE(false, true);
    constexpr std::uint64_t kHalf = kSize / 2;
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        Buffer upload = created(device, kUploadBuffer);
        Buffer middle = created(device, {.size = kSize,
                                         .memory = BufferMemory::Device,
                                         .usage = {.copySrc = true, .copyDst = true}});
        Buffer readback = created(device, kReadbackBuffer);
        auto mappedUpload = upload.map();
        REQUIRE(mappedUpload.has_value());
        const std::span<std::byte> written = mappedBytes(*mappedUpload, kSize);
        for (std::size_t index = 0; index < written.size(); ++index) {
            written[index] = static_cast<std::byte>(static_cast<unsigned char>(index));
        }
        auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());

        // Whole into device memory, then its halves swapped into readback.
        const BufferCopy whole{.srcOffset = 0, .dstOffset = 0, .size = kSize};
        const std::array swapped{BufferCopy{.srcOffset = 0, .dstOffset = kHalf, .size = kHalf},
                                 BufferCopy{.srcOffset = kHalf, .dstOffset = 0, .size = kHalf}};
        auto first = pool->beginCommands();
        REQUIRE(first.has_value());
        CHECK(device.copyBuffer(*first, upload, middle, std::span{&whole, 1}).has_value());
        Token token{};
        if (acrossSubmits) {
            const auto sent = device.submit({.queue = 0, .commandLists = std::span{&*first, 1}, .waits = {}});
            INFO(failureOf(sent));
            REQUIRE(sent.has_value());
            auto second = pool->beginCommands();
            REQUIRE(second.has_value());
            CHECK(device.copyBuffer(*second, middle, readback, swapped).has_value());
            const auto last = device.submit(
                {.queue = 0, .commandLists = std::span{&*second, 1}, .waits = std::span{&*sent, 1}});
            INFO(failureOf(last));
            REQUIRE(last.has_value());
            token = *last;
        }
        else {
            CHECK(globalBarrier(device, *first, kCopyWrites, kCopyReads).has_value());
            CHECK(device.copyBuffer(*first, middle, readback, swapped).has_value());
            const auto sent = device.submit({.queue = 0, .commandLists = std::span{&*first, 1}, .waits = {}});
            INFO(failureOf(sent));
            REQUIRE(sent.has_value());
            token = *sent;
        }
        REQUIRE(device.wait(token, kTestWait).has_value());

        auto mappedReadback = readback.map();
        REQUIRE(mappedReadback.has_value());
        const std::span<const std::byte> read = mappedBytes(*mappedReadback, kSize);
        CHECK(std::ranges::equal(read.first(kHalf), written.subspan(kHalf)));
        CHECK(std::ranges::equal(read.subspan(kHalf), written.first(kHalf)));
        for (Buffer* buffer : std::array{&upload, &middle, &readback}) {
            CHECK(slotsResult(device, destroyed(*buffer)) == Result::ErrorInvalidHandle);
        }
        CHECK(resetResult(device, destroyed(*pool)) == Result::ErrorInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device fails a copy and a barrier with RhiInvalidHandle for a list or buffer that is gone",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        Buffer upload = created(device, kUploadBuffer);
        Buffer goneBuffer = created(device, kReadbackBuffer);
        const bfr_buffer gone = destroyed(goneBuffer);
        REQUIRE(slotsResult(device, gone) == Result::ErrorInvalidHandle);
        auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        const auto list = pool->beginCommands();
        REQUIRE(list.has_value());
        const BufferCopy region{.srcOffset = 0, .dstOffset = 0, .size = kSize};
        // bfr.hpp names a buffer only by a live Buffer, so the stale one goes
        // through bfr.h, with region as bfr_buffer_copy.
        const bfr_buffer_copy cRegion = detail::toC(region);

        CHECK(resultOf(bfr_copy_buffer(device.get(), list->get(), upload.get(), gone, &cRegion, 1)) ==
              Result::ErrorInvalidHandle);
        device.destroyCommandList(*list);
        CHECK(device.copyBuffer(*list, upload, upload, std::span{&region, 1}).error().code ==
              Result::ErrorInvalidHandle);
        CHECK(device.barrier(*list, {}).error().code == Result::ErrorInvalidHandle);
        CHECK(slotsResult(device, destroyed(upload)) == Result::ErrorInvalidHandle);
        CHECK(resetResult(device, destroyed(*pool)) == Result::ErrorInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device passes on what the backend says of storage in upload memory", "[rhi][agnostic]") {
    // Not every API lets shaders write memory the CPU maps, so the backend
    // decides: a buffer that is made is a buffer, and a refusal is its own.
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        BufferDesc desc = kUploadBuffer;
        desc.usage.storage = true;
        auto buffer = made.device.createBuffer(desc);
        if (buffer.has_value()) {
            CHECK(buffer->map().has_value());
            CHECK(slotsResult(made.device, destroyed(*buffer)) == Result::ErrorInvalidHandle);
        }
        else {
            CHECK(buffer.error().code == Result::ErrorDeviceFailed);
        }
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device destroys the buffers left when it goes, and says so", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        // A Buffer keeps its Device alive, so one is left only through bfr.h:
        // kDeviceBuffer as bfr_buffer_desc, never destroyed.
        const bfr_buffer_desc desc = detail::toC(kDeviceBuffer, nullptr);
        bfr_buffer left{};
        REQUIRE(resultOf(bfr_create_buffer(made.device.get(), &desc, &left)) == Result::Success);
    }

    const std::size_t warnings = BFR_LOGGING_ENABLED ? 1 : 0;
    REQUIRE(messages.size() == warnings);
    for (const std::string& message : messages) {
        CHECK_THAT(message, Catch::Matchers::ContainsSubstring("1 buffers were left when the device went"));
    }
}

TEST_CASE("Device collects garbage of buffers destroyed after a submit the GPU has passed", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        const Device& device = made.device;
        Buffer buffer = created(device, kDeviceBuffer);
        auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        const auto commands = pool->beginCommands();
        REQUIRE(commands.has_value());
        const CommandList list = *commands;
        const auto token = device.submit({.queue = 0, .commandLists = std::span{&list, 1}, .waits = {}});
        REQUIRE(token.has_value());
        REQUIRE(slotsResult(device, destroyed(buffer)) == Result::ErrorInvalidHandle);

        const auto pending = device.runGarbageCollection();
        INFO(failureOf(pending));
        CHECK(pending.has_value());
        REQUIRE(device.wait(*token, kTestWait).has_value());
        const auto passed = device.runGarbageCollection();
        INFO(failureOf(passed));
        CHECK(passed.has_value());
        CHECK(resetResult(device, destroyed(*pool)) == Result::ErrorInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device holds as many buffers as its desc's maxBuffers, and fails with RhiLimitReached past it",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        auto instance = createInstance(recordingInto(backend, messages), kValidation);
        INFO(failureOf(instance));
        REQUIRE(instance.has_value());
        auto made = instance->createDevice(
            testDeviceWith(*instance, kFirstAdapter, kNoWindow, [](DeviceDesc& desc) { desc.maxBuffers = 1; }));
        INFO(failureOf(made));
        REQUIRE(made.has_value());
        const Device& device = *made;
        Buffer buffer = created(device, kUploadBuffer);
        CHECK(device.createBuffer(kUploadBuffer).error().code == Result::ErrorLimitReached);
        CHECK(slotsResult(device, destroyed(buffer)) == Result::ErrorInvalidHandle);
        // A destroyed buffer holds its place until garbage collection frees it.
        CHECK(device.createBuffer(kUploadBuffer).error().code == Result::ErrorLimitReached);
        REQUIRE(device.runGarbageCollection().has_value());
        auto again = device.createBuffer(kUploadBuffer);
        INFO(failureOf(again));
        REQUIRE(again.has_value());
        CHECK(slotsResult(device, destroyed(*again)) == Result::ErrorInvalidHandle);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device fails a uniform buffer with RhiLimitReached when it holds no uniform buffer slots",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const FirstAdapterDevice made{recordingInto(backend, messages)};
        REQUIRE(kTestDevice.maxBindlessUniformBuffers == 0);
        const auto uniform = made.device.createBuffer(
            {.size = kSize, .memory = BufferMemory::Upload, .usage = {.uniform = true}});
        REQUIRE_FALSE(uniform.has_value());
        CHECK(uniform.error().code == Result::ErrorLimitReached);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}
