#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/caps.h"

#include <string_view>

namespace bfr::rhi {

/// What a backend reads as the end of a name it is given.
inline constexpr char kNameEnd = '\0';

/// Fails with RhiInvalidCall when a label's or an object's name holds a
/// '\0', which the backend would read as its end.
[[nodiscard]] inline core::Result<void> checkNameEnd(std::string_view name) {
    if (name.contains(kNameEnd)) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "the name holds a '\\0'");
    }
    return {};
}

/// Fails with RhiUnsupported for an object name where features.objectNames
/// is false, and as checkNameEnd. Empty is no name, which passes.
[[nodiscard]] inline core::Result<void> checkObjectName(const Features& features, std::string_view name) {
    if (name.empty()) {
        return {};
    }
    if (!features.objectNames) {
        return core::failure(core::ErrorCode::RhiUnsupported, "the device names no objects (Features::objectNames)");
    }
    return checkNameEnd(name);
}

} // namespace bfr::rhi
