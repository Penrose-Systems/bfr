#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/acceleration_structure.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/pipeline.h"
#include "bfr/rhi/sampler.h"
#include "bfr/rhi/sparse.h"
#include "bfr/rhi/swapchain.h"
#include "bfr/rhi/texture.h"
#include "bfr/rhi/timestamp.h"
#include "rhi/format_check.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <utility>

namespace bfr::rhi {

// The first and last value of each enum a caller passes in. Each enum's
// values run from one to the other with none missing.
constexpr std::pair<LoadOp, LoadOp> rangeOf(LoadOp /*unused*/) {
    return {LoadOp::Load, LoadOp::DontCare};
}
constexpr std::pair<StoreOp, StoreOp> rangeOf(StoreOp /*unused*/) {
    return {StoreOp::Store, StoreOp::DontCare};
}
constexpr std::pair<IndexType, IndexType> rangeOf(IndexType /*unused*/) {
    return {IndexType::Uint32, IndexType::Uint16};
}
constexpr std::pair<BufferMemory, BufferMemory> rangeOf(BufferMemory /*unused*/) {
    return {BufferMemory::Device, BufferMemory::Readback};
}
constexpr std::pair<ColorSpace, ColorSpace> rangeOf(ColorSpace /*unused*/) {
    return {ColorSpace::Srgb, ColorSpace::DisplayNative};
}
constexpr std::pair<ResolveMode, ResolveMode> rangeOf(ResolveMode /*unused*/) {
    return {ResolveMode::None, ResolveMode::Max};
}
constexpr std::pair<Topology, Topology> rangeOf(Topology /*unused*/) {
    return {Topology::TriangleList, Topology::PointList};
}
constexpr std::pair<PrimitiveRestart, PrimitiveRestart> rangeOf(PrimitiveRestart /*unused*/) {
    return {PrimitiveRestart::Off, PrimitiveRestart::Index32};
}
constexpr std::pair<CullMode, CullMode> rangeOf(CullMode /*unused*/) {
    return {CullMode::None, CullMode::Back};
}
constexpr std::pair<FrontFace, FrontFace> rangeOf(FrontFace /*unused*/) {
    return {FrontFace::CounterClockwise, FrontFace::Clockwise};
}
constexpr std::pair<FillMode, FillMode> rangeOf(FillMode /*unused*/) {
    return {FillMode::Solid, FillMode::Wireframe};
}
constexpr std::pair<CompareOp, CompareOp> rangeOf(CompareOp /*unused*/) {
    return {CompareOp::Always, CompareOp::GreaterEqual};
}
constexpr std::pair<StencilOp, StencilOp> rangeOf(StencilOp /*unused*/) {
    return {StencilOp::Keep, StencilOp::DecrementWrap};
}
constexpr std::pair<BlendFactor, BlendFactor> rangeOf(BlendFactor /*unused*/) {
    return {BlendFactor::Zero, BlendFactor::SrcAlphaSaturate};
}
constexpr std::pair<BlendOp, BlendOp> rangeOf(BlendOp /*unused*/) {
    return {BlendOp::Add, BlendOp::Max};
}
constexpr std::pair<Filter, Filter> rangeOf(Filter /*unused*/) {
    return {Filter::Nearest, Filter::Linear};
}
constexpr std::pair<AddressMode, AddressMode> rangeOf(AddressMode /*unused*/) {
    return {AddressMode::Repeat, AddressMode::MirrorClampToEdge};
}
constexpr std::pair<SamplerReduction, SamplerReduction> rangeOf(SamplerReduction /*unused*/) {
    return {SamplerReduction::WeightedAverage, SamplerReduction::Maximum};
}
constexpr std::pair<PresentMode, PresentMode> rangeOf(PresentMode /*unused*/) {
    return {PresentMode::Fifo, PresentMode::Immediate};
}
constexpr std::pair<AlphaMode, AlphaMode> rangeOf(AlphaMode /*unused*/) {
    return {AlphaMode::Opaque, AlphaMode::Inherit};
}
constexpr std::pair<TextureDimension, TextureDimension> rangeOf(TextureDimension /*unused*/) {
    return {TextureDimension::D2, TextureDimension::Cube};
}
constexpr std::pair<TextureLayout, TextureLayout> rangeOf(TextureLayout /*unused*/) {
    return {TextureLayout::Undefined, TextureLayout::ResolveDst};
}
constexpr std::pair<TextureAspect, TextureAspect> rangeOf(TextureAspect /*unused*/) {
    return {TextureAspect::Color, TextureAspect::Stencil};
}
constexpr std::pair<QueueLevel, QueueLevel> rangeOf(QueueLevel /*unused*/) {
    return {QueueLevel::Normal, QueueLevel::Realtime};
}
constexpr std::pair<TimestampStage, TimestampStage> rangeOf(TimestampStage /*unused*/) {
    return {TimestampStage::End, TimestampStage::AccelerationStructureBuild};
}
constexpr std::pair<TimestampGranularity, TimestampGranularity> rangeOf(TimestampGranularity /*unused*/) {
    return {TimestampGranularity::Relaxed, TimestampGranularity::Precise};
}
constexpr std::pair<HostClock, HostClock> rangeOf(HostClock /*unused*/) {
    return {HostClock::Monotonic, HostClock::PerformanceCounter};
}
constexpr std::pair<TileMapping, TileMapping> rangeOf(TileMapping /*unused*/) {
    return {TileMapping::Map, TileMapping::Unmap};
}
constexpr std::pair<AccelerationStructureType, AccelerationStructureType> rangeOf(
    AccelerationStructureType /*unused*/) {
    return {AccelerationStructureType::Bottom, AccelerationStructureType::OpacityMicromap};
}
constexpr std::pair<OpacityMicromapFormat, OpacityMicromapFormat> rangeOf(OpacityMicromapFormat /*unused*/) {
    return {OpacityMicromapFormat::TwoState, OpacityMicromapFormat::FourState};
}
constexpr std::pair<BuildPreference, BuildPreference> rangeOf(BuildPreference /*unused*/) {
    return {BuildPreference::None, BuildPreference::FastBuild};
}
constexpr std::pair<InstanceOpacity, InstanceOpacity> rangeOf(InstanceOpacity /*unused*/) {
    return {InstanceOpacity::AsGeometry, InstanceOpacity::ForceNonOpaque};
}
constexpr std::pair<RayTracingStage, RayTracingStage> rangeOf(RayTracingStage /*unused*/) {
    return {RayTracingStage::RayGeneration, RayTracingStage::Callable};
}
constexpr std::pair<RayTracingGroupType, RayTracingGroupType> rangeOf(RayTracingGroupType /*unused*/) {
    return {RayTracingGroupType::General, RayTracingGroupType::BoxHit};
}
constexpr std::pair<RayTracingGroupShader, RayTracingGroupShader> rangeOf(RayTracingGroupShader /*unused*/) {
    return {RayTracingGroupShader::General, RayTracingGroupShader::Intersection};
}
constexpr std::pair<AccelerationStructureProperty, AccelerationStructureProperty> rangeOf(
    AccelerationStructureProperty /*unused*/) {
    return {AccelerationStructureProperty::CompactedSize, AccelerationStructureProperty::CurrentSize};
}

/// Fails with RhiInvalidCall when value is not one its enum has, such as a
/// number cast to it; what names it in the message. Every backend's switch
/// over an enum takes only its values, so each call that reads one checks it
/// first, through the checks below.
template <class Enum>
[[nodiscard]] core::Result<void> checkKnown(Enum value, std::string_view what) {
    const auto [first, last] = rangeOf(value);
    const auto number = std::to_underlying(value);
    if (number < std::to_underlying(first) || number > std::to_underlying(last)) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "{} {} is not one BFR has", what,
                             static_cast<std::int64_t>(number));
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
    for (const Filter filter : {desc.magFilter, desc.minFilter, desc.mipFilter}) {
        if (auto checked = checkKnown(filter, "filter"); !checked) {
            return checked;
        }
    }
    for (const AddressMode mode : {desc.addressU, desc.addressV, desc.addressW}) {
        if (auto checked = checkKnown(mode, "address mode"); !checked) {
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
