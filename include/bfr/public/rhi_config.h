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
    /// The images a swapchain asks for, within what its surface allows and
    /// never fewer than two.
    std::uint32_t swapchainImages = 3;

    void set(const config::Config& config);
};

} // namespace bfr::rhi
