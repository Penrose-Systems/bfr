#pragma once

#include <spdlog/spdlog.h>

#include <memory>
#include <utility>

#define BFR_LOGLEVEL_TYPE spdlog::level::level_enum
#define BFR_LOGLEVEL_TRACE spdlog::level::trace
#define BFR_LOGLEVEL_DEBUG spdlog::level::debug
#define BFR_LOGLEVEL_CRITICAL spdlog::level::critical
#define BFR_LOGLEVEL_WARN spdlog::level::warn
#define BFR_LOGLEVEL_INFO spdlog::level::info
#define BFR_LOGLEVEL_ERROR spdlog::level::err
#define BFR_LOGLEVEL_OFF spdlog::level::off

namespace bfr::core {

/// False when the build sets BFR_LOGGING off.
inline constexpr bool kLoggingEnabled = BFR_LOGGING_ENABLED;

struct LogConfig {
    LogConfig(BFR_LOGLEVEL_TYPE level = BFR_LOGLEVEL_TRACE) : level(level) {};
    BFR_LOGLEVEL_TYPE level;
};

class Log {
public:
    /// Replaces the logger. Call before other threads log.
    static void init(const LogConfig& conf);
    /// A stdout logger at the default config until init runs.
    static std::shared_ptr<spdlog::logger>& getLogger();

    /// Called through the BFR_LOG_* macros, which drop it when logging is off.
    template <typename... Args>
    static void write(spdlog::source_loc where, BFR_LOGLEVEL_TYPE level, spdlog::format_string_t<Args...> format,
                      Args&&... args) {
        getLogger()->log(where, level, format, std::forward<Args>(args)...);
    }
};

} // namespace bfr::core

// A macro, not a function, so a build with logging off drops each call whole:
// no code, no format string, and the arguments are never evaluated. The
// discarded branch is still compiled, so format strings stay checked and a
// variable only a log call uses is not reported unused. The empty else makes
// the call one whole statement, so an else after it binds to the caller's if.
#define BFR_LOG_AT(level, ...)                                                                                         \
    if constexpr (::bfr::core::kLoggingEnabled) {                                                                      \
        ::bfr::core::Log::write(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, level, __VA_ARGS__);          \
    }                                                                                                                  \
    else                                                                                                               \
        static_cast<void>(0)

#define BFR_LOG_TRACE(...) BFR_LOG_AT(BFR_LOGLEVEL_TRACE, __VA_ARGS__)
#define BFR_LOG_DEBUG(...) BFR_LOG_AT(BFR_LOGLEVEL_DEBUG, __VA_ARGS__)
#define BFR_LOG_INFO(...) BFR_LOG_AT(BFR_LOGLEVEL_INFO, __VA_ARGS__)
#define BFR_LOG_WARN(...) BFR_LOG_AT(BFR_LOGLEVEL_WARN, __VA_ARGS__)
#define BFR_LOG_CRITICAL(...) BFR_LOG_AT(BFR_LOGLEVEL_CRITICAL, __VA_ARGS__)
#define BFR_LOG_ERROR(...) BFR_LOG_AT(BFR_LOGLEVEL_ERROR, __VA_ARGS__)
