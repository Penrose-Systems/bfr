#include "bfr/core/error.h"

#include <catch2/catch_test_macros.hpp>

#include <expected>
#include <string>
#include <string_view>
#include <vector>

namespace {

using bfr::core::Error;
using bfr::core::ErrorCode;
using bfr::core::Result;

Result<int> requirePositive(int value) {
    if (value <= 0) {
        return std::unexpected(Error{.code = ErrorCode::RhiDeviceFailed, .message = "value must be positive"});
    }
    return value;
}

Result<void> requireOpen(bool open) {
    if (!open) {
        return std::unexpected(Error{.code = ErrorCode::RhiInvalidHandle, .message = "closed"});
    }
    return {};
}

} // namespace

TEST_CASE("Result carries the value on success", "[core][unit]") {
    const Result<int> result = requirePositive(3);
    REQUIRE(result.has_value());
    CHECK(*result == 3);
}

TEST_CASE("Result carries the code and message on failure", "[core][unit]") {
    const Result<int> result = requirePositive(0);
    REQUIRE_FALSE(result.has_value());
    CHECK(result.error().code == ErrorCode::RhiDeviceFailed);
    CHECK(result.error().message.text() == "value must be positive");
}

TEST_CASE("Result<void> distinguishes success from failure", "[core][unit]") {
    CHECK(requireOpen(true).has_value());
    const Result<void> failed = requireOpen(false);
    REQUIRE_FALSE(failed.has_value());
    CHECK(failed.error().code == ErrorCode::RhiInvalidHandle);
}

TEST_CASE("Every error code names itself", "[core][unit]") {
    CHECK(bfr::core::name(ErrorCode::RhiInvalidHandle) == "RhiInvalidHandle");
    CHECK(bfr::core::name(ErrorCode::RhiDeviceFailed) == "RhiDeviceFailed");
}

TEST_CASE("failure makes a failed Result of any type", "[core][unit]") {
    const Result<int> result = bfr::core::failure(ErrorCode::RhiDeviceFailed, "no window");
    REQUIRE_FALSE(result.has_value());
    CHECK(result.error().code == ErrorCode::RhiDeviceFailed);
    CHECK(result.error().message.text() == "no window");
}

TEST_CASE("withReasons lists each reason after the message", "[core][unit]") {
    const std::vector<std::string> reasons{"first", "second"};
    CHECK(bfr::core::error::withReasons("no loader", reasons) == "no loader: first; second");
}

TEST_CASE("withReasons without reasons is the message alone", "[core][unit]") {
    CHECK(bfr::core::error::withReasons("no loader", {}) == "no loader");
}

TEST_CASE("failure writes its format and arguments into the error", "[core][unit]") {
    constexpr int kQueue = 7;
    const Result<void> result = bfr::core::failure(ErrorCode::RhiInvalidCall, "no queue {}; it has {}", kQueue, 1);
    REQUIRE_FALSE(result.has_value());
    CHECK(result.error().message == "no queue 7; it has 1");
}

TEST_CASE("Message cuts text longer than it holds and marks the cut", "[core][unit]") {
    using bfr::core::Message;
    const std::string longer(Message::kCapacity + 1, 'x');
    const Message assigned{longer};
    CHECK(assigned.view().size() == Message::kCapacity);
    CHECK(assigned.view().ends_with("..."));
    Message formatted;
    formatted.format("{}{}", longer, longer);
    CHECK(formatted.view().size() == Message::kCapacity);
    CHECK(formatted.view().ends_with("..."));
    const Message copied = formatted;
    CHECK(copied.view() == formatted.view());
    const Message exact{std::string(Message::kCapacity, 'y')};
    CHECK(exact.view() == std::string(Message::kCapacity, 'y'));
}

TEST_CASE("Message cuts text between UTF-8 characters, never inside one", "[core][unit]") {
    using bfr::core::Message;
    // Each character is two bytes, so a cut inside one leaves an odd prefix.
    constexpr std::string_view kTwoBytes = "\u00e9";
    std::string longer;
    while (longer.size() <= Message::kCapacity) {
        longer += kTwoBytes;
    }
    const Message cut{longer};
    REQUIRE(cut.view().ends_with("..."));
    const std::string_view kept = cut.view().substr(0, cut.view().size() - 3);
    CHECK(kept.size() % kTwoBytes.size() == 0);
    CHECK(cut.view().size() <= Message::kCapacity);
    CHECK(longer.starts_with(kept));
}
