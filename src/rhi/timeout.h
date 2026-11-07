#pragma once

#include <cstdint>

namespace bfr::rhi {

inline constexpr std::uint64_t kNanosecondsPerMillisecond = 1'000'000;

/// timeoutNs in the milliseconds a backend's wait counts in, rounded up so the
/// wait is never shorter than the caller asked.
[[nodiscard]] constexpr std::uint64_t millisecondsRoundedUp(std::uint64_t timeoutNs) {
    return (timeoutNs / kNanosecondsPerMillisecond) + (timeoutNs % kNanosecondsPerMillisecond != 0 ? 1 : 0);
}

} // namespace bfr::rhi
