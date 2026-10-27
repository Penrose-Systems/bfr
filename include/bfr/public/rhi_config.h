#pragma once

namespace bfr::rhi {

/// The RHI's settings. Each field's initialiser is its default.
struct RhiConfig {
    /// Runs the API's validation layer and reports its messages.
    bool validation = false;
};

} // namespace bfr::rhi
