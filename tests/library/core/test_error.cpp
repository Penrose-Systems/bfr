#include "error.h"

#include <catch2/catch_test_macros.hpp>

#include <expected>
#include <string>
#include <vector>

namespace {

using bfr::core::Error;
using bfr::core::ErrorCode;
using bfr::core::Result;

Result<int> requirePositive(int value) {
    if (value <= 0) {
        return std::unexpected(Error{.code = ErrorCode::DisplayWindowFailed, .message = "value must be positive"});
    }
    return value;
}

Result<void> requireOpen(bool open) {
    if (!open) {
        return std::unexpected(Error{.code = ErrorCode::DisplayNotInitialised, .message = "closed"});
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
    CHECK(result.error().code == ErrorCode::DisplayWindowFailed);
    CHECK(result.error().message == "value must be positive");
}

TEST_CASE("Result<void> distinguishes success from failure", "[core][unit]") {
    CHECK(requireOpen(true).has_value());
    const Result<void> failed = requireOpen(false);
    REQUIRE_FALSE(failed.has_value());
    CHECK(failed.error().code == ErrorCode::DisplayNotInitialised);
}

TEST_CASE("Every error code names itself", "[core][unit]") {
    CHECK(bfr::core::name(ErrorCode::DisplayNotInitialised) == "DisplayNotInitialised");
    CHECK(bfr::core::name(ErrorCode::DisplayWindowFailed) == "DisplayWindowFailed");
}

TEST_CASE("failure makes a failed Result of any type", "[core][unit]") {
    const Result<int> result = bfr::core::failure(ErrorCode::DisplayWindowFailed, "no window");
    REQUIRE_FALSE(result.has_value());
    CHECK(result.error().code == ErrorCode::DisplayWindowFailed);
    CHECK(result.error().message == "no window");
}

TEST_CASE("withReasons lists each reason after the message", "[core][unit]") {
    const std::vector<std::string> reasons{"first", "second"};
    CHECK(bfr::core::error::withReasons("no loader", reasons) == "no loader: first; second");
}

TEST_CASE("withReasons without reasons is the message alone", "[core][unit]") {
    CHECK(bfr::core::error::withReasons("no loader", {}) == "no loader");
}
