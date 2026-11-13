#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/pipeline.h"
#include "rhi/acceleration_structure_check.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace bfr::rhi {

/// Fails with RhiUnsupported where the device has no ray tracing pipelines,
/// as Features::rayTracingPipeline says.
[[nodiscard]] inline core::Result<void> checkRayTracingPipelines(bool rayTracingPipeline) {
    if (!rayTracingPipeline) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device has no ray tracing pipelines (Features::rayTracingPipeline)");
    }
    return {};
}

/// Fails with RhiUnsupported where the device has no indirect traces, as
/// Features::traceRaysIndirect says.
[[nodiscard]] inline core::Result<void> checkTraceRaysIndirect(bool traceRaysIndirect) {
    if (!traceRaysIndirect) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device has no indirect traces (Features::traceRaysIndirect)");
    }
    return {};
}

/// Fails with RhiUnsupported for a library, or libraries linked, where
/// Features::rayTracingPipelineLibrary is false, allowOpacityMicromaps where
/// Features::opacityMicromap is, allowMotion where Features::rayTracingMotion
/// is, or allowClusters where Features::clusterAccelerationStructure is, and
/// RhiInvalidCall for
/// replayIdentifiers neither empty nor at least replaySize bytes for each
/// group, which BFR reads.
[[nodiscard]] inline core::Result<void> checkPipelineExtras(const RayTracingPipelineDesc& desc,
                                                            const Features& features, std::uint32_t replaySize) {
    if ((desc.library || !desc.libraries.empty()) && !features.rayTracingPipelineLibrary) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device has no ray tracing pipeline libraries (Features::rayTracingPipelineLibrary)");
    }
    if (desc.allowMotion && !features.rayTracingMotion) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device has no structures with motion (Features::rayTracingMotion)");
    }
    if (desc.allowClusters) {
        if (auto checked = checkClusters(features.clusterAccelerationStructure); !checked) {
            return checked;
        }
    }
    if (desc.allowOpacityMicromaps && !features.opacityMicromap) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device has no opacity micromaps (Features::opacityMicromap)");
    }
    if (!desc.replayIdentifiers.empty() && desc.replayIdentifiers.size() < desc.groups.size() * replaySize) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "{} bytes of replay identifiers hold fewer than {} groups' of {} bytes",
                             desc.replayIdentifiers.size(), desc.groups.size(), replaySize);
    }
    return {};
}

/// Fails with RhiInvalidCall when out holds fewer than count identifiers of
/// identifierSize bytes, which BFR writes.
[[nodiscard]] inline core::Result<void> checkIdentifiers(std::uint32_t count, std::span<const std::byte> out,
                                                         std::uint32_t identifierSize) {
    if (out.size() < std::size_t{count} * identifierSize) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "{} bytes hold fewer than {} identifiers of {}",
                             out.size(), count, identifierSize);
    }
    return {};
}

/// Fails with RhiUnsupported where the device gives no buffer addresses, as
/// Features::bufferAddress says.
[[nodiscard]] inline core::Result<void> checkBufferAddress(bool bufferAddress) {
    if (!bufferAddress) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device gives no buffer addresses (Features::bufferAddress)");
    }
    return {};
}

} // namespace bfr::rhi
