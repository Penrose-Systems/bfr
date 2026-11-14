#pragma once

#include "bfr/bfr_enums.hpp" // IWYU pragma: export
#include "bfr/core/handle.h"

#include <cstdint>
#include <string_view>

namespace bfr::rhi {

/// The GPU timestamps a TimestampPoolHandle names.
struct TimestampPool;

using TimestampPoolHandle = core::Handle<TimestampPool>;

/// How to create a timestamp pool: the device's queue whose lists write it,
/// and how many timestamps it holds.
struct TimestampPoolDesc {
    std::uint32_t queue = 0;
    std::uint32_t count = 0;
    /// What a debugger or profiler shows it as, where Features::objectNames;
    /// read during the call only. Empty is none. The create call fails with
    /// RhiUnsupported for a name where Features::objectNames is false,
    /// RhiInvalidCall for one holding a '\0', which the backend reads as its
    /// end, and RhiDeviceFailed when the backend refuses it.
    std::string_view name{};
};

/// How long a timestamp tick lasts, in the unit the backend gives: a period
/// in nanoseconds or a frequency in ticks per second. The other is 0.
struct TimestampRate {
    float periodNs = 0;
    std::uint64_t frequency = 0;
};

/// A GPU timestamp and a host clock's reading taken together, so GPU ticks can
/// be placed on the host's timeline. maxDeviationNs is how far apart the two
/// readings may be, where deviationGiven says the backend gave it.
struct CalibratedTimestamps {
    std::uint64_t gpu = 0;
    std::uint64_t host = 0;
    std::uint64_t maxDeviationNs = 0;
    bool deviationGiven = false;
};

} // namespace bfr::rhi
