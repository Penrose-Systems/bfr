#pragma once

#include "bfr/bfr.h"
#include "bfr/bfr_enums.hpp" // IWYU pragma: export
#include "bfr/core/handle.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace bfr::rhi {

/// The GPU queries a QueryPoolHandle names.
struct QueryPool;

using QueryPoolHandle = core::Handle<QueryPool>;

/// How many PipelineStatistic values there are.
inline constexpr std::size_t kPipelineStatistics = BFR_PIPELINE_STATISTIC_COUNT;

/// No offset: a statistic a pool's results don't hold.
inline constexpr std::uint32_t kNoOffset = BFR_NO_OFFSET;

/// The statistics a PipelineStatistics pool counts. Task and mesh
/// invocations where Features::meshStatistics.
struct PipelineStatisticsMask {
    bool inputVertices = false;
    bool inputPrimitives = false;
    bool vertexInvocations = false;
    bool geometryInvocations = false;
    bool geometryPrimitives = false;
    bool clipInvocations = false;
    bool clipPrimitives = false;
    bool fragmentInvocations = false;
    bool tessellationControl = false;
    bool tessellationEvaluationInvocations = false;
    bool computeInvocations = false;
    bool taskInvocations = false;
    bool meshInvocations = false;
};

/// How to create a query pool: what its queries count, and how many it
/// holds. statistics is read only for PipelineStatistics.
struct QueryPoolDesc {
    QueryType type = QueryType::Occlusion;
    std::uint32_t count = 0;
    PipelineStatisticsMask statistics;
    /// What a debugger or profiler shows it as, where Features::objectNames;
    /// read during the call only. Empty is none. The create call fails with
    /// RhiUnsupported for a name where Features::objectNames is false,
    /// RhiInvalidCall for one holding a '\0', which the backend reads as its
    /// end, and RhiDeviceFailed when the backend refuses it.
    std::string_view name{};
};

/// Where a pool's results land in the buffer resolveQueries writes, as the
/// backend lays them out: stride bytes for each query, and in each, the
/// 64-bit value of an Occlusion or MeshPrimitivesGenerated query at value,
/// and each statistic of a PipelineStatistics query at its offset, indexed
/// by PipelineStatistic. kNoOffset where the results hold no such value.
struct QueryLayout {
    std::uint32_t stride = 0;
    std::uint32_t value = kNoOffset;
    std::array<std::uint32_t, kPipelineStatistics> statistics{};
};

} // namespace bfr::rhi
