#pragma once

#include "config.h"

#include <cstdint>

namespace bfr::rhi {

/// The RHI's settings. Each field's initialiser is its default; set overwrites
/// the fields the config's rhi section sets.
struct RhiConfig {
    /// The layout of the rhi section this code reads.
    static constexpr std::uint32_t kVersion = 1;

    /// Runs the API's validation layer and reports its messages.
    bool validation = false;

    void set(const config::Config& config);
};

} // namespace bfr::rhi
