#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/format.h"

#include <algorithm>
#include <cstdint>

namespace bfr::rhi {

/// Fails with RhiInvalidCall when format is not one of kFormats: Undefined,
/// which names no format, or a number Format does not have. Each backend's
/// format tables hold kFormats only, so every call that looks a caller's
/// format up in them calls this first.
[[nodiscard]] inline core::Result<void> checkFormat(Format format) {
    if (std::ranges::find(kFormats, format) == kFormats.end()) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "format {} is not one BFR has",
                             static_cast<std::uint32_t>(format));
    }
    return {};
}

} // namespace bfr::rhi
