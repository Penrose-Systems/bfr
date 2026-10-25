#pragma once

#include "log.h"

#include <spdlog/sinks/ostream_sink.h>

#include <memory>
#include <sstream>

// Replaces the logger's sinks with one that writes the bare message to output.
inline void captureLogInto(std::ostringstream& output) {
    auto sink = std::make_shared<spdlog::sinks::ostream_sink_st>(output);
    sink->set_pattern("%v");
    bfr::core::Log::getLogger()->sinks() = {sink};
}
