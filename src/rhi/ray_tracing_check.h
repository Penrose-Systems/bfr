#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/pipeline.h"

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

/// Fails with RhiUnsupported where the device has no capture-replay
/// identifiers, as Features::rayTracingCaptureReplay says.
[[nodiscard]] inline core::Result<void> checkCaptureReplay(bool captureReplay) {
    if (!captureReplay) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device captures no ray tracing identifiers (Features::rayTracingCaptureReplay)");
    }
    return {};
}

/// Fails with RhiUnsupported for a library, or libraries linked, where
/// Features::rayTracingPipelineLibrary is false, or captureReplay where
/// Features::rayTracingCaptureReplay is, and RhiInvalidCall for
/// replayIdentifiers neither empty nor replaySize bytes for each group.
[[nodiscard]] inline core::Result<void> checkPipelineExtras(const RayTracingPipelineDesc& desc,
                                                            const Features& features, std::uint32_t replaySize) {
    if ((desc.library || !desc.libraries.empty()) && !features.rayTracingPipelineLibrary) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device has no ray tracing pipeline libraries (Features::rayTracingPipelineLibrary)");
    }
    if (desc.captureReplay) {
        if (auto checked = checkCaptureReplay(features.rayTracingCaptureReplay); !checked) {
            return checked;
        }
    }
    if (!desc.replayIdentifiers.empty() && desc.replayIdentifiers.size() != desc.groups.size() * replaySize) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "{} bytes of replay identifiers are not {} groups' of {} bytes",
                             desc.replayIdentifiers.size(), desc.groups.size(), replaySize);
    }
    return {};
}

/// Fails with RhiInvalidCall when the pipeline is not a ray tracing one, or
/// out holds fewer than count identifiers of identifierSize bytes.
[[nodiscard]] inline core::Result<void> checkIdentifiers(bool rayTracing, std::uint32_t count,
                                                         std::span<const std::byte> out, std::uint32_t identifierSize) {
    if (!rayTracing) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "the pipeline is not a ray tracing one");
    }
    if (out.size() / identifierSize < count) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "{} bytes hold fewer than {} identifiers of {}",
                             out.size(), count, identifierSize);
    }
    return {};
}

/// Fails with RhiUnsupported where the device gives no buffer addresses, as
/// Features::bufferAddress says, and RhiInvalidCall for a buffer of no usage
/// that has one: indirect, shaderBindingTable, accelerationStructureInput or
/// accelerationStructureData.
[[nodiscard]] inline core::Result<void> checkBufferAddress(bool bufferAddress, bool addressed) {
    if (!bufferAddress) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device gives no buffer addresses (Features::bufferAddress)");
    }
    if (!addressed) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "the buffer has no indirect, shaderBindingTable or accelerationStructureInput usage, "
                             "so no address");
    }
    return {};
}

} // namespace bfr::rhi
