#include "log.h"

#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <memory>

namespace bfr::core {

namespace {

std::shared_ptr<spdlog::logger> createLogger(const LogConfig& conf) {
    auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    sink->set_pattern("%^[%T] %n: %v%$");

    auto logger = std::make_shared<spdlog::logger>("BFR", sink);
    // On the logger, not the sink, so a dropped message is never formatted.
    logger->set_level(conf.level);
    logger->flush_on(BFR_LOGLEVEL_TRACE);
    return logger;
}

} // namespace

void Log::init(const LogConfig& conf) {
    getLogger() = createLogger(conf);
}

std::shared_ptr<spdlog::logger>& Log::getLogger() {
    static std::shared_ptr<spdlog::logger> logger = createLogger(LogConfig{});
    return logger;
}

} // namespace bfr::core
