#pragma once

#include "bfr/core/log.h"

#include <mutex>
#include <sstream>
#include <string_view>

// A logger that writes each message from least up to output, one a line.
// output outlives whatever logs to it.
inline bfr::core::Logger loggingInto(std::ostringstream& output,
                                     bfr::core::Severity least = bfr::core::Severity::Trace) {
    return {[](bfr::core::Severity /*severity*/, std::string_view message, void* userData) {
                static std::mutex mutex;
                const std::scoped_lock lock{mutex};
                *static_cast<std::ostringstream*>(userData) << message << '\n';
            },
            &output, least};
}
