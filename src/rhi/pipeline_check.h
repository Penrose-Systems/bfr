#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/pipeline.h"

#include <cstdint>
#include <span>

namespace bfr::rhi {

/// Fails with RhiInvalidCall when desc asks for a compute pipeline no backend
/// can make: one with no code, or a workgroup with a dimension of 0. Each
/// backend's createComputePipeline calls it first, so every backend answers
/// such a desc the same way.
[[nodiscard]] inline core::Result<void> checkComputePipelineDesc(const ComputePipelineDesc& desc) {
    if (desc.compute.code.empty()) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "the compute shader has no code");
    }
    for (const std::uint32_t size : desc.workgroupSize) {
        if (size == 0) {
            return core::failure(core::ErrorCode::RhiInvalidCall,
                                 "a workgroup dimension is 0; set the size the shader declares");
        }
    }
    return {};
}

/// Fails with RhiInvalidCall when desc asks for a graphics pipeline no backend
/// can make: one with no vertex code, more than kMaxColorTargets targets, a
/// target of format Undefined, a depth or stencil test without a
/// depth-stencil format, or samples of 0. Each backend's
/// createGraphicsPipeline calls it first.
[[nodiscard]] inline core::Result<void> checkGraphicsPipelineDesc(const GraphicsPipelineDesc& desc) {
    if (desc.vertex.code.empty()) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "the vertex shader has no code");
    }
    if (desc.colorTargetCount > kMaxColorTargets) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "{} color targets; a pipeline renders to at most {}",
                             desc.colorTargetCount, kMaxColorTargets);
    }
    const std::span targets{desc.colorTargets.data(), desc.colorTargetCount};
    for (const ColorTarget& target : targets) {
        if (target.format == Format::Undefined) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "color target {} has format Undefined",
                                 &target - targets.data());
        }
    }
    if ((desc.depthStencil.depthTest || desc.depthStencil.stencilTest) &&
        desc.depthStencilFormat == Format::Undefined) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "a depth or stencil test is on, but there is no depth-stencil format");
    }
    if (desc.samples == 0) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "samples is 0; 1 is no multisampling");
    }
    return {};
}

} // namespace bfr::rhi
