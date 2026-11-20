#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/pipeline.h"
#include "rhi/enum_check.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace bfr::rhi {

/// Fails with RhiInvalidCall when desc holds what BFR can't pass on: an enum
/// value or format BFR doesn't have, or more than kMaxColorTargets targets,
/// which its descs hold.
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

/// Fails with RhiUnsupported for a triangle fan the device's features lack.
/// Each backend's createGraphicsPipeline and setTopology call it.
[[nodiscard]] inline core::Result<void> checkTopology(Topology topology, const Features& features) {
    if (topology == Topology::TriangleFan && !features.triangleFans) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device has no triangle fans (Features::triangleFans)");
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

/// Whether topology is a list, which restarts only with Features::listRestart.
[[nodiscard]] constexpr bool isList(Topology topology) {
    return topology == Topology::TriangleList || topology == Topology::LineList || topology == Topology::PointList;
}

/// Fails with RhiUnsupported when desc asks for a fixed-function setting the
/// device's features lack: depth clamping, wireframe fill, a depth bias
/// clamped, color targets that blend differently (Features::independentBlend),
/// or primitive restart on a list. Vulkan's rules (VUID-VkPipelineRasterizationStateCreateInfo-depthClampEnable-00782,
/// -polygonMode-01507, VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-00754,
/// VUID-VkPipelineColorBlendStateCreateInfo-pAttachments-00605,
/// VUID-VkPipelineInputAssemblyStateCreateInfo-topology-06252) hold on every
/// backend, as their features do. Each backend's createGraphicsPipeline calls
/// it after checkGraphicsPipelineDesc; a mesh pipeline assembles no input, so
/// it restarts nothing.
[[nodiscard]] inline core::Result<void> checkPipelineFeatures(const GraphicsPipelineDesc& desc,
                                                              const Features& features) {
    if (desc.raster.depthClamp && !features.depthClamp) {
        return core::failure(core::ErrorCode::RhiUnsupported, "the device clamps no depth (Features::depthClamp)");
    }
    if (desc.raster.fill == FillMode::Wireframe && !features.wireframe) {
        return core::failure(core::ErrorCode::RhiUnsupported, "the device fills no wireframe (Features::wireframe)");
    }
    const bool biased = desc.raster.depthBias != 0 || desc.raster.depthBiasSlope != 0;
    if (biased && desc.raster.depthBiasClamp != 0 && !features.depthBiasClamp) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device clamps no depth bias; depthBiasClamp is {} (Features::depthBiasClamp)",
                             desc.raster.depthBiasClamp);
    }
    const auto targets = std::span{desc.colorTargets}.first(
        std::min<std::size_t>(desc.colorTargetCount, kMaxColorTargets));
    if (!features.independentBlend && std::ranges::any_of(targets, [&targets](const ColorTarget& target) {
            return target.blend != targets.front().blend;
        })) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device blends every color target alike (Features::independentBlend)");
    }
    if (desc.mesh.code.empty() && desc.primitiveRestart != PrimitiveRestart::Off && isList(desc.topology) &&
        !features.listRestart) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device restarts only strips and fans, not lists (Features::listRestart)");
    }
    return {};
}

} // namespace bfr::rhi
