#include "error.h"

#include "text.h"

#include <expected>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace bfr::core {

std::unexpected<Error> failure(ErrorCode code, std::string message) {
    return std::unexpected(Error{.code = code, .message = std::move(message)});
}

std::string_view name(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::RhiInstanceFailed:
            return "RhiInstanceFailed";
        case ErrorCode::RhiLoaderFailed:
            return "RhiLoaderFailed";
        case ErrorCode::RhiValidationUnavailable:
            return "RhiValidationUnavailable";
        case ErrorCode::RhiAdapterOutOfRange:
            return "RhiAdapterOutOfRange";
        case ErrorCode::RhiDeviceFailed:
            return "RhiDeviceFailed";
        case ErrorCode::RhiNoAdapter:
            return "RhiNoAdapter";
        case ErrorCode::RhiSurfaceFailed:
            return "RhiSurfaceFailed";
        case ErrorCode::RhiInvalidHandle:
            return "RhiInvalidHandle";
        case ErrorCode::RhiPresentUnsupported:
            return "RhiPresentUnsupported";
        case ErrorCode::RhiSwapchainFailed:
            return "RhiSwapchainFailed";
        case ErrorCode::RhiLimitReached:
            return "RhiLimitReached";
    }
    return "Unknown";
}

} // namespace bfr::core

namespace bfr::core::error {

std::string withReasons(std::string_view message, std::span<const std::string> reasons) {
    if (reasons.empty()) {
        return std::string{message};
    }
    return std::format("{}: {}", message, text::join(reasons, "; "));
}

} // namespace bfr::core::error
