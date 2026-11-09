#pragma once

#include "bfr/core/handle.h"

#include <cstdint>

namespace bfr::rhi {

/// The GPU timestamps a TimestampPoolHandle names.
struct TimestampPool;

using TimestampPoolHandle = core::Handle<TimestampPool>;

/// How to create a timestamp pool: the device's queue whose lists write it,
/// and how many timestamps it holds.
struct TimestampPoolDesc {
    std::uint32_t queue = 0;
    std::uint32_t count = 0;
};

/// The point in a list's work a timestamp is written at. Which ones a backend
/// takes, inside and outside rendering, is in Caps::timestamps.
enum class TimestampStage : std::uint32_t {
    /// Once all the work recorded before it is done.
    End = 0,
    /// Once the work recorded before it has started.
    Start = 1,
    /// Once the vertex shading recorded before it is done.
    Vertex = 2,
    /// Once the fragment shading recorded before it is done.
    Fragment = 3,
    /// Once the compute shading recorded before it is done.
    Compute = 4,
    /// Once the copies recorded before it are done.
    Copy = 5,
    /// Once the task shading recorded before it is done.
    Task = 6,
    /// Once the mesh shading recorded before it is done.
    Mesh = 7,
    /// Once the ray tracing shading recorded before it is done.
    RayTracing = 8,
    /// Once the acceleration structure builds and copies recorded before it
    /// are done.
    AccelerationStructureBuild = 9,
};

/// How exactly a timestamp marks its stage, where the backend takes it, as
/// Caps::timestamps says.
enum class TimestampGranularity : std::uint32_t {
    /// The backend may write it later than the stage ends, costing less.
    Relaxed = 0,
    /// As close to the stage's end as the backend can.
    Precise = 1,
};

/// How long a timestamp tick lasts, in the unit the backend gives: a period
/// in nanoseconds or a frequency in ticks per second. The other is 0.
struct TimestampRate {
    float periodNs = 0;
    std::uint64_t frequency = 0;
};

/// A host clock a GPU timestamp can be read beside.
enum class HostClock : std::uint32_t {
    /// CLOCK_MONOTONIC, which std::chrono::steady_clock reads on Linux.
    Monotonic = 0,
    /// CLOCK_MONOTONIC_RAW.
    MonotonicRaw = 1,
    /// QueryPerformanceCounter, which std::chrono::steady_clock reads on Windows.
    PerformanceCounter = 2,
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
