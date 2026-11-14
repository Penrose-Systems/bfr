#pragma once

#include "bfr/bfr_enums.hpp"
#include "bfr/core/error.h"

#include <utility>

namespace bfr::rhi {

/// Fails with RhiInvalidCall when format is Undefined, which names no format,
/// or a number Format does not have. Each backend's format tables hold every
/// other format, so every call that looks a caller's format up in them calls
/// this first.
[[nodiscard]] inline core::Result<void> checkFormat(Format format) {
    if (format == Format::Undefined || name(format).empty()) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "format {} is not one BFR has",
                             std::to_underlying(format));
    }
    return {};
}

} // namespace bfr::rhi
