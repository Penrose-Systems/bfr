#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>

namespace bfr::core {

/// What went wrong inside BFR. Internal: the C API turns each into a
/// bfr_result.
enum class ErrorCode : std::uint8_t {
    RhiInstanceFailed,
    RhiLoaderFailed,
    RhiValidationUnavailable,
    RhiAdapterOutOfRange,
    RhiDeviceFailed,
    RhiNoAdapter,
    RhiWindowFailed,
    RhiInvalidHandle,
    RhiPresentUnsupported,
    RhiSwapchainFailed,
    RhiLimitReached,
    RhiOutOfMemory,
    RhiDeviceLost,
    RhiWindowLost,
    RhiUnsupported,
    RhiQueryFailed,
    RhiBackendUnavailable,
    RhiOutOfDate,
    RhiInvalidCall,
};

struct Error {
    ErrorCode code;
    std::string message;
};

template <class T>
using Result = std::expected<T, Error>;

/// A failed Result of any type: `return core::failure(code, message);`.
[[nodiscard]] std::unexpected<Error> failure(ErrorCode code, std::string message);

/// The code's identifier, for diagnostics.
[[nodiscard]] std::string_view name(ErrorCode code) noexcept;

} // namespace bfr::core

namespace bfr::core::error {

/// The message followed by each reason for it, "message: first; second", or
/// the message alone without reasons. Builds only the text, so any error type
/// can carry it: `core::failure(code, core::error::withReasons(message, reasons))`.
[[nodiscard]] std::string withReasons(std::string_view message, std::span<const std::string> reasons);

} // namespace bfr::core::error
