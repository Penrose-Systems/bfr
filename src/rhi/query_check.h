#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/query.h"

namespace bfr::rhi {

/// Fails with RhiUnsupported for a query pool the device's features lack:
/// pipeline statistics (Features::pipelineStatistics), and mesh statistics,
/// as a pool of MeshPrimitivesGenerated or one counting task or mesh
/// invocations (Features::meshStatistics). Vulkan's rules
/// (VUID-VkQueryPoolCreateInfo-queryType-00791, -meshShaderQueries-07068,
/// -07069), which those features name on every backend. A backend's
/// createQueryPool calls it.
[[nodiscard]] inline core::Result<void> checkQueryPoolFeatures(const QueryPoolDesc& desc, const Features& features) {
    if (desc.type == QueryType::PipelineStatistics && !features.pipelineStatistics) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device counts no pipeline statistics (Features::pipelineStatistics)");
    }
    const bool meshStatistics = desc.type == QueryType::MeshPrimitivesGenerated ||
                                (desc.type == QueryType::PipelineStatistics &&
                                 (desc.statistics.taskInvocations || desc.statistics.meshInvocations));
    if (meshStatistics && !features.meshStatistics) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device counts nothing of mesh or task shaders (Features::meshStatistics)");
    }
    return {};
}

/// Fails with RhiUnsupported for a precise query where the device's
/// Features::occlusionQueryPrecise is false (VUID-vkCmdBeginQuery-queryType-00800).
/// A backend's beginQuery calls it.
[[nodiscard]] inline core::Result<void> checkPreciseQuery(bool precise, const Features& features) {
    if (precise && !features.occlusionQueryPrecise) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device counts no exact occlusion (Features::occlusionQueryPrecise)");
    }
    return {};
}

} // namespace bfr::rhi
