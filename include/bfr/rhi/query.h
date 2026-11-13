#pragma once

#include "bfr/bfr.h"
#include "bfr/core/handle.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace bfr::rhi {

/// The GPU queries a QueryPoolHandle names.
struct QueryPool;

using QueryPoolHandle = core::Handle<QueryPool>;

/// What a pool's queries count between beginQuery and endQuery. The values
/// are the C API's.
enum class QueryType : std::uint8_t {
    /// Samples that pass the depth and stencil tests: an exact count where
    /// beginQuery asks for one, else 0 for none and another value for some.
    Occlusion = 0,
    /// The statistics QueryPoolDesc::statistics selects.
    PipelineStatistics = 1,
    /// Primitives mesh shaders emit.
    MeshPrimitivesGenerated = 2,
};

/// One statistic a PipelineStatistics query counts, in Vulkan's bit order,
/// which is D3D12's field order. Each counts what the backend counts:
/// TessellationControl is patches on Vulkan and hull shader invocations on
/// D3D12. The values are the C API's.
enum class PipelineStatistic : std::uint8_t {
    InputVertices = 0,
    InputPrimitives = 1,
    VertexInvocations = 2,
    GeometryInvocations = 3,
    GeometryPrimitives = 4,
    ClipInvocations = 5,
    ClipPrimitives = 6,
    FragmentInvocations = 7,
    TessellationControl = 8,
    TessellationEvaluationInvocations = 9,
    ComputeInvocations = 10,
    TaskInvocations = 11,
    MeshInvocations = 12,
};

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
