#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/pipeline.h"
#include "rhi/enum_check.h"

#include <cstdint>
#include <limits>

namespace bfr::rhi {

/// Fails with RhiInvalidCall when desc holds what BFR can't pass on: an enum
/// value or format BFR doesn't have, more than kMaxColorTargets targets, a
/// minSampleShading outside 0 to 1, both a vertex and a mesh shader
/// (VUID-VkGraphicsPipelineCreateInfo-pStages-02095), or a task shader
/// without a mesh shader.
/// A color target of format Undefined is an unused one. Each backend's
/// createGraphicsPipeline calls it first.
[[nodiscard]] inline core::Result<void> checkGraphicsPipelineDesc(const GraphicsPipelineDesc& desc) {
    if (auto checked = checkEnums(desc); !checked) {
        return checked;
    }
    if (desc.colorTargetCount > kMaxColorTargets) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "{} color targets; a pipeline renders to at most {}",
                             desc.colorTargetCount, kMaxColorTargets);
    }
    if (!(desc.minSampleShading >= 0 && desc.minSampleShading <= 1)) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "minSampleShading {} is not 0 to 1",
                             desc.minSampleShading);
    }
    // A pipeline is fed by vertices or by mesh shaders, not both.
    if (!desc.mesh.code.empty() && !desc.vertex.code.empty()) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "a pipeline takes a vertex shader or a mesh shader");
    }
    if (desc.mesh.code.empty() && !desc.task.code.empty()) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "a task shader needs a mesh shader");
    }
    return {};
}

/// Fails with RhiUnsupported for a mesh or task shader the device's features
/// lack. Each backend's createGraphicsPipeline calls it after
/// checkGraphicsPipelineDesc.
[[nodiscard]] inline core::Result<void> checkMeshStages(const GraphicsPipelineDesc& desc, const Features& features) {
    if (!desc.mesh.code.empty() && !features.meshShader) {
        return core::failure(core::ErrorCode::RhiUnsupported, "the device has no mesh shaders (Features::meshShader)");
    }
    if (!desc.task.code.empty() && !features.taskShader) {
        return core::failure(core::ErrorCode::RhiUnsupported, "the device has no task shaders (Features::taskShader)");
    }
    return {};
}

/// The samples a mask of samples bits holds: bit i for sample i.
[[nodiscard]] constexpr std::uint64_t everySample(std::uint32_t samples) {
    return samples >= std::numeric_limits<std::uint64_t>::digits ? ~std::uint64_t{0}
                                                                 : (std::uint64_t{1} << samples) - 1U;
}

/// Fails with RhiUnsupported when desc asks for multisampling the device's
/// features lack: alphaToOne, sample shading, or a sampleMask leaving one of
/// the pipeline's samples out. Each backend's createGraphicsPipeline calls it
/// after checkGraphicsPipelineDesc.
[[nodiscard]] inline core::Result<void> checkMultisample(const GraphicsPipelineDesc& desc, const Features& features) {
    if (desc.alphaToOne && !features.alphaToOne) {
        return core::failure(core::ErrorCode::RhiUnsupported, "the device has no alpha to one (Features::alphaToOne)");
    }
    if (desc.minSampleShading > 0 && !features.sampleRateShading) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device shades no sample apart (Features::sampleRateShading)");
    }
    const std::uint64_t every = everySample(desc.samples);
    if ((desc.sampleMask & every) != every && !features.sampleMask) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device writes every sample; sampleMask {:#x} leaves some out (Features::sampleMask)",
                             desc.sampleMask);
    }
    return {};
}

} // namespace bfr::rhi
