#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/pipeline.h"
#include "bfr/rhi/timestamp.h"
#include "rhi/enum_check.h"
#include "rhi/format_blocks.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

namespace bfr::rhi {

/// Fails with RhiInvalidCall when desc asks for a rendering no backend can
/// begin: a load or store op or resolve mode BFR doesn't have, more color
/// attachments than there are, none at all, a color attachment naming no
/// texture, a resolve texture named with no texture, a read-only
/// depth-stencil attachment that clears or drops an aspect, or, with a
/// swapchain, no color attachment or attachment 0 naming a resolve texture
/// when the swapchain's image is its resolve target. Attachment 0 names no
/// texture when the swapchain's image is it. Each backend's beginRendering
/// calls it first, then looks its handles up.
[[nodiscard]] inline core::Result<void> checkRenderingDesc(bool hasSwapchain, const RenderingDesc& desc) {
    if (auto checked = checkEnums(desc); !checked) {
        return checked;
    }
    if (desc.colorAttachmentCount > kMaxColorTargets) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "{} color attachments is more than the {} there are",
                             desc.colorAttachmentCount, kMaxColorTargets);
    }
    // The render area is the attachments' size; rendering with none needs a
    // size in RenderingDesc, which it does not have yet.
    if (desc.colorAttachmentCount == 0 && desc.depthStencil.texture.isNull()) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "no attachment to render to");
    }
    if (hasSwapchain && desc.colorAttachmentCount == 0) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "a swapchain's image is color attachment 0, so there is at least one");
    }
    for (std::uint32_t i = 0; i < desc.colorAttachmentCount; ++i) {
        const ColorAttachment& attachment = desc.colorAttachments.at(i);
        const bool swapchains = i == 0 && hasSwapchain;
        const bool named = !attachment.texture.isNull();
        if (!swapchains && !named) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "color attachment {} names no texture", i);
        }
        if (!attachment.resolveTexture.isNull() && (swapchains || !named)) {
            return core::failure(core::ErrorCode::RhiInvalidCall,
                                 "color attachment {} resolves into the swapchain's image, or names no texture to "
                                 "resolve, so its resolve texture is null",
                                 i);
        }
    }
    const DepthStencilAttachment& depth = desc.depthStencil;
    if (!depth.resolveTexture.isNull() && depth.texture.isNull()) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "the depth-stencil resolve texture has no texture to resolve");
    }
    if (depth.readOnly && (depth.depthLoadOp != LoadOp::Load || depth.stencilLoadOp != LoadOp::Load ||
                           depth.depthStoreOp != StoreOp::Store || depth.stencilStoreOp != StoreOp::Store)) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "a read-only depth-stencil attachment loads and stores both aspects; it writes neither");
    }
    return {};
}

/// Fails with RhiUnsupported when a depth-stencil attachment's resolve modes,
/// for the aspects format has, are not ones resolve has, alone or together.
/// Each backend's beginRendering calls it once it knows the texture's format,
/// when the attachment names a resolve texture.
[[nodiscard]] inline core::Result<void> checkDepthResolve(const DepthStencilAttachment& attachment, Format format,
                                                          const ResolveCaps& resolve) {
    const FormatBlock& block = blockOf(format);
    const ResolveMode depth = block.depthBytes != 0 ? attachment.depthResolve : ResolveMode::None;
    const ResolveMode stencil = block.stencilBytes != 0 ? attachment.stencilResolve : ResolveMode::None;
    const auto has = [](std::uint32_t modes, ResolveMode mode) {
        return mode == ResolveMode::None || (modes & (1U << std::to_underlying(mode))) != 0;
    };
    if (!has(resolve.depthModes, depth)) {
        return core::failure(core::ErrorCode::RhiUnsupported, "the device resolves no depth by mode {} (Caps::resolve)",
                             std::to_underlying(depth));
    }
    if (!has(resolve.stencilModes, stencil)) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device resolves no stencil by mode {} (Caps::resolve)", std::to_underlying(stencil));
    }
    // Only a format with both aspects pairs its modes.
    if (block.depthBytes == 0 || block.stencilBytes == 0 || depth == stencil) {
        return {};
    }
    const bool oneNone = depth == ResolveMode::None || stencil == ResolveMode::None;
    if (oneNone ? !resolve.independentNone : !resolve.independent) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device resolves depth and stencil by one mode (Caps::resolve); {} and {} were asked",
                             std::to_underlying(depth), std::to_underlying(stencil));
    }
    return {};
}

/// Fails with RhiInvalidCall when data is not whole 32-bit values or more
/// than the size bytes the push constants hold, the size the device's caps
/// report. Each backend's pushConstants calls it first.
[[nodiscard]] inline core::Result<void> checkPushConstants(std::span<const std::byte> data, std::uint32_t size) {
    constexpr std::size_t kValueBytes = sizeof(std::uint32_t);
    if (data.size() % kValueBytes != 0) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "{} bytes of push constants is not whole 32-bit values",
                             data.size());
    }
    if (data.size() > size) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "{} bytes of push constants; there are {}", data.size(),
                             size);
    }
    return {};
}

/// Whether a Metal list's rendering is open, so its render encoder exists.
struct RecordState {
    bool rendering = false;
};

/// Fails with RhiInvalidCall when no rendering is open: Metal records a
/// viewport, scissor, draw or endRendering on the rendering's encoder, which
/// exists only while one is.
[[nodiscard]] inline core::Result<void> checkRendering(const RecordState& state) {
    if (!state.rendering) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "no rendering is open");
    }
    return {};
}

/// Counts a pushConstants call in pushed, the pool's count since its reset,
/// against the pool's pushConstantUpdates; fails with RhiLimitReached at it.
[[nodiscard]] inline core::Result<void> countPushConstants(std::uint32_t& pushed, std::uint32_t pushConstantUpdates) {
    if (pushed >= pushConstantUpdates) {
        return core::failure(core::ErrorCode::RhiLimitReached, "the pool holds at most {} push constant updates",
                             pushConstantUpdates);
    }
    ++pushed;
    return {};
}

/// Fails with RhiUnsupported for more than one indirect draw in a call where
/// the device's Features::multiDrawIndirect is false. Each backend's
/// indirect draws call it first.
[[nodiscard]] inline core::Result<void> checkDrawCount(std::uint32_t drawCount, bool multiDrawIndirect) {
    if (drawCount > 1 && !multiDrawIndirect) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device draws one indirect draw per call (Features::multiDrawIndirect); {} were asked",
                             drawCount);
    }
    return {};
}

/// Fails with RhiUnsupported where the device's Features::drawIndirectCount
/// is false. Each backend's count draws call it first.
[[nodiscard]] inline core::Result<void> checkCountDraws(bool drawIndirectCount) {
    if (!drawIndirectCount) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "the device reads no draw count from a buffer (Features::drawIndirectCount)");
    }
    return {};
}

/// Whether stages lists stage.
[[nodiscard]] constexpr bool takes(const TimestampStages& stages, TimestampStage stage) {
    switch (stage) {
        case TimestampStage::End:
            return stages.end;
        case TimestampStage::Start:
            return stages.start;
        case TimestampStage::Vertex:
            return stages.vertex;
        case TimestampStage::Fragment:
            return stages.fragment;
        case TimestampStage::Compute:
            return stages.compute;
        case TimestampStage::Copy:
            return stages.copy;
        case TimestampStage::Task:
            return stages.task;
        case TimestampStage::Mesh:
            return stages.mesh;
    }
    return false;
}

/// Fails with RhiUnsupported where the device's Features::meshShader is
/// false. Each backend's mesh draws call it first.
[[nodiscard]] inline core::Result<void> checkMeshDraws(bool meshShader) {
    if (!meshShader) {
        return core::failure(core::ErrorCode::RhiUnsupported, "the device has no mesh shaders (Features::meshShader)");
    }
    return {};
}

} // namespace bfr::rhi
