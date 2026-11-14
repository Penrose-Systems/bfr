#pragma once

#include "bfr/bfr_enums.hpp"
#include "bfr/core/error.h"
#include "bfr/rhi/acceleration_structure.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/pipeline.h"
#include "bfr/rhi/query.h"
#include "bfr/rhi/sampler.h"
#include "bfr/rhi/sparse.h"
#include "bfr/rhi/swapchain.h"
#include "bfr/rhi/texture.h"
#include "bfr/rhi/timestamp.h"
#include "rhi/format_check.h"

#include <algorithm>
#include <cstddef>
#include <span>
#include <string_view>
#include <utility>

namespace bfr::rhi {

/// Fails with RhiInvalidCall when value is not one its enum has, such as a
/// number cast to it; what names it in the message. Every backend's switch
/// over an enum takes only its values, so each call that reads one checks it
/// first, through the checks below.
template <class Enum>
[[nodiscard]] core::Result<void> checkKnown(Enum value, std::string_view what) {
    // bfr_enums.hpp names every value its enum has, and no other.
    if (name(value).empty()) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "{} {} is not one BFR has", what,
                             std::to_underlying(value));
    }
    return {};
}

/// Fails with RhiInvalidCall when a barrier's layout is not one BFR has.
[[nodiscard]] inline core::Result<void> checkLayouts(TextureLayout before, TextureLayout after) {
    if (auto checked = checkKnown(before, "texture layout"); !checked) {
        return checked;
    }
    return checkKnown(after, "texture layout");
}

/// Fails with RhiInvalidCall when desc's memory is not one BufferMemory has.
[[nodiscard]] inline core::Result<void> checkEnums(const BufferDesc& desc) {
    return checkKnown(desc.memory, "buffer memory");
}

/// Fails with RhiInvalidCall when desc's dimension, aspect or format is not
/// one BFR has.
[[nodiscard]] inline core::Result<void> checkEnums(const SparseFormatDesc& desc) {
    if (auto checked = checkKnown(desc.dimension, "texture dimension"); !checked) {
        return checked;
    }
    if (auto checked = checkKnown(desc.aspect, "texture aspect"); !checked) {
        return checked;
    }
    return checkFormat(desc.format);
}

/// Fails with RhiInvalidCall when one of inputs' enums, or a vertex format
/// its triangles name, is not one BFR has.
[[nodiscard]] inline core::Result<void> checkEnums(const AccelerationStructureInputs& inputs) {
    if (auto checked = checkKnown(inputs.type, "acceleration structure type"); !checked) {
        return checked;
    }
    if (auto checked = checkKnown(inputs.flags.preference, "build preference"); !checked) {
        return checked;
    }
    for (const TriangleGeometry& triangles : inputs.triangles) {
        if (auto checked = checkFormat(triangles.vertexFormat); !checked) {
            return checked;
        }
        if (auto checked = checkKnown(triangles.indexType, "index type"); !checked) {
            return checked;
        }
        if (auto checked = checkKnown(triangles.opacityMicromap.indexType, "opacity micromap index type"); !checked) {
            return checked;
        }
    }
    for (const OpacityMicromapUsage& usage : inputs.micromapUsages) {
        if (auto checked = checkKnown(usage.format, "opacity micromap format"); !checked) {
            return checked;
        }
    }
    return {};
}

/// Fails with RhiInvalidCall when a cluster operation's inputs hold an enum
/// value BFR doesn't have.
[[nodiscard]] inline core::Result<void> checkEnums(const ClusterInputs& inputs) {
    if (auto checked = checkKnown(inputs.operation, "cluster operation"); !checked) {
        return checked;
    }
    if (auto checked = checkKnown(inputs.mode, "cluster mode"); !checked) {
        return checked;
    }
    if (auto checked = checkKnown(inputs.flags.preference, "build preference"); !checked) {
        return checked;
    }
    if (auto checked = checkKnown(inputs.move.type, "cluster structure type"); !checked) {
        return checked;
    }
    if (inputs.triangles.vertexFormat == Format::Undefined) {
        return {};
    }
    return checkFormat(inputs.triangles.vertexFormat);
}

/// Fails with RhiInvalidCall when a shader's stage or a group's type in desc
/// is not one BFR has.
[[nodiscard]] inline core::Result<void> checkEnums(const RayTracingPipelineDesc& desc) {
    for (const RayTracingShader& shader : desc.shaders) {
        if (auto checked = checkKnown(shader.stage, "ray tracing stage"); !checked) {
            return checked;
        }
    }
    for (const RayTracingGroup& group : desc.groups) {
        if (auto checked = checkKnown(group.type, "ray tracing group type"); !checked) {
            return checked;
        }
    }
    return {};
}

/// Fails with RhiInvalidCall when desc's dimension or format is not one BFR has.
[[nodiscard]] inline core::Result<void> checkEnums(const TextureDesc& desc) {
    if (auto checked = checkKnown(desc.dimension, "texture dimension"); !checked) {
        return checked;
    }
    return checkFormat(desc.format);
}

/// Fails with RhiInvalidCall when one of desc's enums is not one BFR has.
[[nodiscard]] inline core::Result<void> checkEnums(const SamplerDesc& desc) {
    for (const auto& [filter, what] : {std::pair{desc.magFilter, "mag filter"}, std::pair{desc.minFilter, "min filter"},
                                       std::pair{desc.mipFilter, "mip filter"}}) {
        if (auto checked = checkKnown(filter, what); !checked) {
            return checked;
        }
    }
    for (const auto& [mode, what] :
         {std::pair{desc.addressU, "address mode u"}, std::pair{desc.addressV, "address mode v"},
          std::pair{desc.addressW, "address mode w"}}) {
        if (auto checked = checkKnown(mode, what); !checked) {
            return checked;
        }
    }
    if (auto checked = checkKnown(desc.reduction, "sampler reduction"); !checked) {
        return checked;
    }
    return checkKnown(desc.compareOp, "compare op");
}

/// Fails with RhiInvalidCall when one of desc's enums, or its format, is not
/// one BFR has.
[[nodiscard]] inline core::Result<void> checkEnums(const SwapchainDesc& desc) {
    if (auto checked = checkKnown(desc.presentMode, "present mode"); !checked) {
        return checked;
    }
    if (auto checked = checkFormat(desc.format); !checked) {
        return checked;
    }
    if (auto checked = checkKnown(desc.colorSpace, "color space"); !checked) {
        return checked;
    }
    return checkKnown(desc.alphaMode, "alpha mode");
}

/// Fails with RhiInvalidCall when a load or store op or a resolve mode of
/// desc's attachments is not one BFR has.
[[nodiscard]] inline core::Result<void> checkEnums(const RenderingDesc& desc) {
    const std::span attachments{desc.colorAttachments.data(),
                                std::min<std::size_t>(desc.colorAttachmentCount, desc.colorAttachments.size())};
    for (const ColorAttachment& attachment : attachments) {
        if (auto checked = checkKnown(attachment.loadOp, "load op"); !checked) {
            return checked;
        }
        if (auto checked = checkKnown(attachment.storeOp, "store op"); !checked) {
            return checked;
        }
    }
    const DepthStencilAttachment& depth = desc.depthStencil;
    for (const LoadOp op : {depth.depthLoadOp, depth.stencilLoadOp}) {
        if (auto checked = checkKnown(op, "load op"); !checked) {
            return checked;
        }
    }
    for (const StoreOp op : {depth.depthStoreOp, depth.stencilStoreOp}) {
        if (auto checked = checkKnown(op, "store op"); !checked) {
            return checked;
        }
    }
    for (const ResolveMode mode : {depth.depthResolve, depth.stencilResolve}) {
        if (auto checked = checkKnown(mode, "resolve mode"); !checked) {
            return checked;
        }
    }
    return {};
}

/// Fails with RhiInvalidCall when one of desc's enums, or a format it names,
/// is not one BFR has. A depth-stencil format of Undefined is none.
[[nodiscard]] inline core::Result<void> checkEnums(const GraphicsPipelineDesc& desc) {
    if (auto checked = checkKnown(desc.topology, "topology"); !checked) {
        return checked;
    }
    if (auto checked = checkKnown(desc.primitiveRestart, "primitive restart"); !checked) {
        return checked;
    }
    if (auto checked = checkKnown(desc.raster.cull, "cull mode"); !checked) {
        return checked;
    }
    if (auto checked = checkKnown(desc.raster.frontFace, "front face"); !checked) {
        return checked;
    }
    if (auto checked = checkKnown(desc.raster.fill, "fill mode"); !checked) {
        return checked;
    }
    if (auto checked = checkKnown(desc.depthStencil.depthCompare, "compare op"); !checked) {
        return checked;
    }
    for (const StencilFace* face : {&desc.depthStencil.front, &desc.depthStencil.back}) {
        for (const StencilOp op : {face->fail, face->pass, face->depthFail}) {
            if (auto checked = checkKnown(op, "stencil op"); !checked) {
                return checked;
            }
        }
        if (auto checked = checkKnown(face->compare, "compare op"); !checked) {
            return checked;
        }
    }
    const std::span targets{desc.colorTargets.data(),
                            std::min<std::size_t>(desc.colorTargetCount, desc.colorTargets.size())};
    for (const ColorTarget& target : targets) {
        if (target.format != Format::Undefined) {
            if (auto checked = checkFormat(target.format); !checked) {
                return checked;
            }
        }
        const BlendState& blend = target.blend;
        for (const BlendFactor factor : {blend.srcColor, blend.dstColor, blend.srcAlpha, blend.dstAlpha}) {
            if (auto checked = checkKnown(factor, "blend factor"); !checked) {
                return checked;
            }
        }
        for (const BlendOp op : {blend.colorOp, blend.alphaOp}) {
            if (auto checked = checkKnown(op, "blend op"); !checked) {
                return checked;
            }
        }
    }
    if (desc.depthStencilFormat != Format::Undefined) {
        return checkFormat(desc.depthStencilFormat);
    }
    return {};
}

} // namespace bfr::rhi
