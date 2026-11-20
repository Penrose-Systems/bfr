#pragma once

#include "bfr/bfr_enums.hpp" // IWYU pragma: export

#include <cstddef>
#include <cstdint>

namespace bfr::rhi {

/// One more than Call's largest value: the entries call timings hold, one for
/// each value, indexed by it. A value Call no longer has keeps its entry, at 0.
inline constexpr std::size_t kCallCount = 124;

/// The CPU time one kind of call has taken, read from std::chrono::steady_clock
/// around each call: how many there were, their nanoseconds in all, and the
/// longest one's.
struct CallTiming {
    std::uint64_t count = 0;
    std::uint64_t totalNs = 0;
    std::uint64_t maxNs = 0;
};

} // namespace bfr::rhi
