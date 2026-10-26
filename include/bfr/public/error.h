#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>

namespace bfr::core {

/// The engine-wide failure list, grouped by the module that raises each code.
/// Codes are appended, never reordered or reused. The base type is the
/// smallest that holds the list and widens with it.
enum class ErrorCode : std::uint8_t {
    // Platform
    DisplayAlreadyInitialised,
    DisplayNotInitialised,
    DisplayBackendFailed,
    DisplayWindowFailed,
    DisplayFullscreenFailed,
    // RHI
    RhiInstanceFailed,
    RhiLoaderFailed,
    RhiValidationUnavailable,
    RhiAdapterOutOfRange,
    RhiDeviceFailed,
    RhiNoAdapter,
    // Config
    ConfigLoadFailed,
    // RHI
    RhiSurfaceFailed,
    RhiInvalidHandle,
    RhiPresentUnsupported,
    RhiSwapchainFailed,
    RhiLimitReached,
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
