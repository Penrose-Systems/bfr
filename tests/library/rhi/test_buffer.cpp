// What every backend does with buffers, through the interface alone. Each case
// runs on every backend the build has.

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/instance.h"
#include "rhi_test_support.h"

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
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace bfr::rhi;
using namespace bfr::rhi::test;
using bfr::core::ErrorCode;

namespace {

constexpr std::uint64_t kSize = 4096;
constexpr std::byte kPattern{0x5A};

constexpr BufferDesc kDeviceBuffer{.size = kSize,
                                   .memory = BufferMemory::Device,
                                   .usage = {.copyDst = true, .storage = true}};
constexpr BufferDesc kUploadBuffer{.size = kSize, .memory = BufferMemory::Upload, .usage = {.copySrc = true}};
constexpr BufferDesc kReadbackBuffer{.size = kSize, .memory = BufferMemory::Readback, .usage = {.copyDst = true}};

// An Instance and a device on its first adapter, presenting nowhere.
struct BufferDevice {
    explicit BufferDevice(const TestInstanceDesc& desc) {
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

BufferHandle created(Device& device, const BufferDesc& desc) {
    const auto buffer = device.createBuffer(desc);
    INFO(failureOf(buffer));
    REQUIRE(buffer.has_value());
    return *buffer;
}

} // namespace

TEST_CASE("Device maps upload and readback buffers, whose bytes the CPU writes and reads", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    const BufferDesc desc = GENERATE(kUploadBuffer, kReadbackBuffer);
    std::vector<std::string> messages;
    {
        const BufferDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        const BufferHandle buffer = created(device, desc);
        const auto mapped = device.mapBuffer(buffer);
        INFO(failureOf(mapped));
        REQUIRE(mapped.has_value());
        REQUIRE(mapped->size() == kSize);
        std::ranges::fill(*mapped, kPattern);
        CHECK(std::ranges::all_of(*mapped, [](std::byte value) { return value == kPattern; }));
        CHECK(device.destroyBuffer(buffer).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device refuses to map a buffer in device memory", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const BufferDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        const BufferHandle buffer = created(device, kDeviceBuffer);

        CHECK(device.mapBuffer(buffer).error().code == ErrorCode::RhiInvalidCall);
        CHECK(device.destroyBuffer(buffer).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device fails with RhiInvalidHandle on a destroyed buffer, and destroying a null one does nothing",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const BufferDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        const BufferHandle buffer = created(device, kUploadBuffer);
        REQUIRE(device.destroyBuffer(buffer).has_value());

        CHECK(device.mapBuffer(buffer).error().code == ErrorCode::RhiInvalidHandle);
        CHECK(device.destroyBuffer(buffer).error().code == ErrorCode::RhiInvalidHandle);
        CHECK(device.destroyBuffer({}).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device ends a buffer handle destroyed after a submit at once, and the wait still passes",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const BufferDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        const BufferHandle buffer = created(device, kDeviceBuffer);
        const auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        const auto commands = device.beginCommands(*pool);
        REQUIRE(commands.has_value());
        const CommandListHandle list = *commands;
        const auto token = device.submit({.queue = 0, .commandLists = std::span{&list, 1}, .waits = {}});
        INFO(failureOf(token));
        REQUIRE(token.has_value());

        CHECK(device.destroyBuffer(buffer).has_value());
        CHECK(device.mapBuffer(buffer).error().code == ErrorCode::RhiInvalidHandle);
        CHECK(device.wait(*token).has_value());
        CHECK(device.destroyCommandPool(*pool).has_value());
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
        const BufferDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        const BufferHandle upload = created(device, kUploadBuffer);
        const BufferHandle middle = created(device, {.size = kSize,
                                                     .memory = BufferMemory::Device,
                                                     .usage = {.copySrc = true, .copyDst = true}});
        const BufferHandle readback = created(device, kReadbackBuffer);
        const auto written = device.mapBuffer(upload);
        REQUIRE(written.has_value());
        for (std::size_t index = 0; index < written->size(); ++index) {
            (*written)[index] = static_cast<std::byte>(static_cast<unsigned char>(index));
        }
        const auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());

        // Whole into device memory, then its halves swapped into readback.
        const BufferCopy whole{.srcOffset = 0, .dstOffset = 0, .size = kSize};
        const std::array swapped{BufferCopy{.srcOffset = 0, .dstOffset = kHalf, .size = kHalf},
                                 BufferCopy{.srcOffset = kHalf, .dstOffset = 0, .size = kHalf}};
        auto first = device.beginCommands(*pool);
        REQUIRE(first.has_value());
        CHECK(device.copyBuffer(*first, upload, middle, std::span{&whole, 1}).has_value());
        Token token{};
        if (acrossSubmits) {
            const auto sent = device.submit({.queue = 0, .commandLists = std::span{&*first, 1}, .waits = {}});
            INFO(failureOf(sent));
            REQUIRE(sent.has_value());
            auto second = device.beginCommands(*pool);
            REQUIRE(second.has_value());
            CHECK(device.copyBuffer(*second, middle, readback, swapped).has_value());
            const auto last = device.submit(
                {.queue = 0, .commandLists = std::span{&*second, 1}, .waits = std::span{&*sent, 1}});
            INFO(failureOf(last));
            REQUIRE(last.has_value());
            token = *last;
        }
        else {
            CHECK(device.barrier(*first).has_value());
            CHECK(device.copyBuffer(*first, middle, readback, swapped).has_value());
            const auto sent = device.submit({.queue = 0, .commandLists = std::span{&*first, 1}, .waits = {}});
            INFO(failureOf(sent));
            REQUIRE(sent.has_value());
            token = *sent;
        }
        REQUIRE(device.wait(token).has_value());

        const auto read = device.mapBuffer(readback);
        REQUIRE(read.has_value());
        CHECK(std::ranges::equal(read->first(kHalf), written->subspan(kHalf)));
        CHECK(std::ranges::equal(read->subspan(kHalf), written->first(kHalf)));
        for (const BufferHandle buffer : {upload, middle, readback}) {
            CHECK(device.destroyBuffer(buffer).has_value());
        }
        CHECK(device.destroyCommandPool(*pool).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device fails a copy and a barrier with RhiInvalidHandle for a list or buffer that is gone",
          "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const BufferDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        const BufferHandle upload = created(device, kUploadBuffer);
        const BufferHandle gone = created(device, kReadbackBuffer);
        REQUIRE(device.destroyBuffer(gone).has_value());
        const auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        const auto list = device.beginCommands(*pool);
        REQUIRE(list.has_value());
        const BufferCopy region{.srcOffset = 0, .dstOffset = 0, .size = kSize};

        CHECK(device.copyBuffer(*list, upload, gone, std::span{&region, 1}).error().code ==
              ErrorCode::RhiInvalidHandle);
        CHECK(device.destroyCommandList(*list).has_value());
        CHECK(device.copyBuffer(*list, upload, upload, std::span{&region, 1}).error().code ==
              ErrorCode::RhiInvalidHandle);
        CHECK(device.barrier(*list).error().code == ErrorCode::RhiInvalidHandle);
        CHECK(device.destroyBuffer(upload).has_value());
        CHECK(device.destroyCommandPool(*pool).has_value());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device refuses a buffer of no bytes or no usage with RhiInvalidCall", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const BufferDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        BufferDesc empty = kUploadBuffer;
        empty.size = 0;
        CHECK(device.createBuffer(empty).error().code == ErrorCode::RhiInvalidCall);
        BufferDesc unused = kUploadBuffer;
        unused.usage = {};
        CHECK(device.createBuffer(unused).error().code == ErrorCode::RhiInvalidCall);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device passes on what the backend says of storage in upload memory", "[rhi][agnostic]") {
    // Not every API lets shaders write memory the CPU maps, so the backend
    // decides: a buffer that is made is a buffer, and a refusal is its own.
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const BufferDevice made{recordingInto(backend, messages)};
        Device& device = *made.device;
        BufferDesc desc = kUploadBuffer;
        desc.usage.storage = true;
        const auto buffer = device.createBuffer(desc);
        if (buffer.has_value()) {
            CHECK(device.mapBuffer(*buffer).has_value());
            CHECK(device.destroyBuffer(*buffer).has_value());
        }
        else {
            CHECK(buffer.error().code == ErrorCode::RhiDeviceFailed);
        }
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Device destroys the buffers left when it goes, and says so", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const BufferDevice made{recordingInto(backend, messages)};
        static_cast<void>(created(*made.device, kDeviceBuffer));
    }

    const std::size_t warnings = bfr::core::kLoggingEnabled ? 1 : 0;
    REQUIRE(messages.size() == warnings);
    for (const std::string& message : messages) {
        CHECK_THAT(message, Catch::Matchers::ContainsSubstring("1 buffers were left when the device went"));
    }
}
