#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/format.h"

#include <algorithm>
#include <cstdint>
#include <optional>

namespace bfr::rhi {

/// number as the Format of kFormats it is; nothing for Undefined, which names
/// no format, or a number Format does not have. Takes the number wider than
/// Format, so a caller holding one checks it before a cast that would cut it
/// short.
[[nodiscard]] constexpr std::optional<Format> knownFormat(std::int64_t number) {
    const auto* found = std::ranges::find_if(kFormats, [number](Format format) {
        return static_cast<std::int64_t>(format) == number;
    });
    if (found == kFormats.end()) {
        return std::nullopt;
    }
    return *found;
}

/// Fails with RhiInvalidCall when format is not one of kFormats: Undefined,
/// which names no format, or a number Format does not have. Each backend's
/// format tables hold kFormats only, so every call that looks a caller's
/// format up in them calls this first.
[[nodiscard]] inline core::Result<void> checkFormat(Format format) {
    if (!knownFormat(static_cast<std::int64_t>(format))) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "format {} is not one BFR has",
                             static_cast<std::uint32_t>(format));
    }
    return {};
}

} // namespace bfr::rhi
