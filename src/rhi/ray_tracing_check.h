#pragma once

#include "bfr/core/error.h"

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
