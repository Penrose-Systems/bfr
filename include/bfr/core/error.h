#pragma once

#include "bfr/bfr.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <utility>

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
    RhiTimeout,
};

/// Text held inside an error, so making, copying or recording one never
/// allocates: a call on the frame path may fail every frame, such as acquire
/// on a minimised window, and must not touch the heap when it does. Text past
/// kCapacity, bfr.h's BFR_MAX_MESSAGE_LENGTH, is cut short and ends "...".
class Message {
public:
    static constexpr std::size_t kCapacity = BFR_MAX_MESSAGE_LENGTH;

    Message() = default;
    // Implicit, so a message reads as text wherever one is given or asked for.
    Message(std::string_view text) { assign(text); }
    Message(const char* text) : Message(std::string_view{text}) {}
    Message(const Message& other) { assign(other.view()); }
    Message& operator=(const Message& other) {
        if (this != &other) {
            assign(other.view());
        }
        return *this;
    }
    // A move copies too: the text is held in place, so there is nothing to take.
    Message(Message&& other) noexcept { assign(other.view()); }
    Message& operator=(Message&& other) noexcept {
        if (this != &other) {
            assign(other.view());
        }
        return *this;
    }
    ~Message() = default;

    /// Replaces the text with text.
    void assign(std::string_view text) noexcept {
        size_ = std::min(text.size(), kCapacity);
        std::ranges::copy(text.substr(0, size_), std::begin(chars_));
        if (text.size() > kCapacity) {
            markCut();
        }
    }

    /// Replaces the text with format filled with args, written straight into
    /// the message: nothing is allocated, unlike std::format. A format the
    /// standard library refuses leaves a message saying so, never a throw, so
    /// a destructor may log.
    template <class... Args>
    void format(std::format_string<Args...> format, Args&&... args) noexcept {
        try {
            const auto written = std::format_to_n(std::begin(chars_), kCapacity, format, std::forward<Args>(args)...);
            size_ = std::min(static_cast<std::size_t>(written.size), kCapacity);
            if (static_cast<std::size_t>(written.size) > kCapacity) {
                markCut();
            }
        } catch (...) {
            // The unfilled format is what is known for certain, so it is the
            // message, never text BFR makes up.
            assign(format.get());
        }
    }

    void clear() { size_ = 0; }
    [[nodiscard]] std::string_view view() const { return {std::data(chars_), size_}; }
    // Implicit, so a message reads as text wherever one is given or asked for.
    operator std::string_view() const { return view(); }
    /// A std::string of the text, for cold code and tests; it allocates.
    [[nodiscard]] std::string text() const { return std::string{view()}; }
    bool operator==(std::string_view other) const { return view() == other; }

private:
    // Ends the text with kCut at a character boundary: a UTF-8 character
    // that straddles the cut goes whole, so the text stays valid UTF-8.
    void markCut() noexcept {
        char* at = std::data(chars_) + (kCapacity - kCut.size());
        while (at != std::data(chars_) && isContinuation(*at)) {
            --at;
        }
        std::ranges::copy(kCut, at);
        size_ = static_cast<std::size_t>(at - std::data(chars_)) + kCut.size();
    }

    // Whether byte continues a multi-byte UTF-8 character: 10xxxxxx.
    [[nodiscard]] static bool isContinuation(char byte) noexcept {
        constexpr unsigned kMask = 0xC0U;
        constexpr unsigned kContinuation = 0x80U;
        return (static_cast<unsigned char>(byte) & kMask) == kContinuation;
    }

    static constexpr std::string_view kCut = "...";
    static_assert(kCapacity > kCut.size());

    // A plain array, not std::array, so it is left as it is until written: a
    // std::array member is zero-filled on every construction, 1 KiB each
    // time, and a message is made on the frame path whenever a call there
    // fails or logs. Only the size_ bytes written are ever read.
    char chars_[kCapacity];
    std::size_t size_ = 0;
};

struct Error {
    ErrorCode code;
    Message message;
};

template <class T>
using Result = std::expected<T, Error>;

/// A failed Result of any type: `return core::failure(code, message);`.
[[nodiscard]] std::unexpected<Error> failure(ErrorCode code, std::string_view message);

/// A failed Result whose message is format filled with args, written straight
/// into the error, so failing allocates nothing:
/// `return core::failure(code, "queue {} of {}", queue, count);`.
template <class... Args>
    requires(sizeof...(Args) > 0)
[[nodiscard]] std::unexpected<Error> failure(ErrorCode code, std::format_string<Args...> format, Args&&... args) {
    Error error{.code = code, .message = {}};
    error.message.format(format, std::forward<Args>(args)...);
    return std::unexpected(std::move(error));
}

/// The code's identifier, for diagnostics.
[[nodiscard]] std::string_view name(ErrorCode code) noexcept;

} // namespace bfr::core

namespace bfr::core::error {

/// The message followed by each reason for it, "message: first; second", or
/// the message alone without reasons. Builds only the text, so any error type
/// can carry it: `core::failure(code, core::error::withReasons(message, reasons))`.
[[nodiscard]] std::string withReasons(std::string_view message, std::span<const std::string> reasons);

} // namespace bfr::core::error
