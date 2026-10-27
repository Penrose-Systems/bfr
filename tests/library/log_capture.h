#pragma once

#include "log.h"

#include <mutex>
#include <sstream>
#include <string_view>

// A logger that writes each message from least up to output, one a line.
// output outlives whatever logs to it.
inline bfr::core::Logger loggingInto(std::ostringstream& output,
                                     bfr::core::Severity least = bfr::core::Severity::Trace) {
    return {[&output](bfr::core::Severity /*severity*/, std::string_view message) {
                static std::mutex mutex;
                const std::scoped_lock lock{mutex};
                output << message << '\n';
            },
            least};
}
