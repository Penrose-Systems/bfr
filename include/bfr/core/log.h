#pragma once

#include <cstdint>
#include <format>
#include <functional>
#include <string_view>
#include <utility>

namespace bfr::core {

/// False when the build sets BFR_LOGGING off.
inline constexpr bool kLoggingEnabled = BFR_LOGGING_ENABLED;

/// How serious a message is, least first. The values are the C API's.
enum class Severity : std::uint8_t {
    Trace = 1,
    Debug,
    Info,
    Warning,
    Error,
};

/// Where BFR's messages go: a callback, and the least severe message it
/// wants. Without a callback nothing is sent. It may be called on any thread.
class Logger {
public:
    using Callback = std::function<void(Severity severity, std::string_view message)>;

    /// Sends nothing.
    Logger() = default;
    Logger(Callback callback, Severity least) : callback_(std::move(callback)), least_(least) {}

    /// One that sends nothing, for code that has no logger to hand.
    [[nodiscard]] static const Logger& none() {
        static const Logger logger;
        return logger;
    }

    [[nodiscard]] bool wants(Severity severity) const { return callback_ && severity >= least_; }

    /// Sends message if severity is wanted.
    void send(Severity severity, std::string_view message) const {
        if (wants(severity)) {
            callback_(severity, message);
        }
    }

private:
    Callback callback_;
    Severity least_ = Severity::Warning;
};

} // namespace bfr::core

// A macro, not a function: a message the logger doesn't want is never
// formatted and its arguments are never evaluated, and a build with logging
// off drops the call whole. The discarded branch is still compiled, so format
// strings stay checked and a variable only a log call uses is not reported
// unused. The empty else makes the call one whole statement, so an else after
// it binds to the caller's if.
#define BFR_LOG_AT(logger, severity, ...)                                                                              \
    if constexpr (::bfr::core::kLoggingEnabled) {                                                                      \
        if (const ::bfr::core::Logger& bfrLogger = (logger); bfrLogger.wants(severity)) {                              \
            bfrLogger.send(severity, std::format(__VA_ARGS__));                                                        \
        }                                                                                                              \
    }                                                                                                                  \
    else                                                                                                               \
        static_cast<void>(0)

#define BFR_LOG_TRACE(logger, ...) BFR_LOG_AT(logger, ::bfr::core::Severity::Trace, __VA_ARGS__)
#define BFR_LOG_DEBUG(logger, ...) BFR_LOG_AT(logger, ::bfr::core::Severity::Debug, __VA_ARGS__)
#define BFR_LOG_INFO(logger, ...) BFR_LOG_AT(logger, ::bfr::core::Severity::Info, __VA_ARGS__)
#define BFR_LOG_WARN(logger, ...) BFR_LOG_AT(logger, ::bfr::core::Severity::Warning, __VA_ARGS__)
#define BFR_LOG_ERROR(logger, ...) BFR_LOG_AT(logger, ::bfr::core::Severity::Error, __VA_ARGS__)
