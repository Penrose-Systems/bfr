#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/pipeline.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace bfr::rhi {

/// Fails with RhiInvalidCall when desc asks for a rendering no backend can
/// begin: more color attachments than there are, none at all, a color
/// attachment naming no texture, or, with a swapchain, no color attachment or
/// attachment 0 naming a texture when the swapchain's image is it. Each
/// backend's beginRendering calls it first, then looks its handles up.
[[nodiscard]] inline core::Result<void> checkRenderingDesc(bool hasSwapchain, const RenderingDesc& desc) {
    if (desc.colorAttachmentCount > kMaxColorTargets) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "{} color attachments is more than the {} there are",
                             desc.colorAttachmentCount, kMaxColorTargets);
    }
    if (desc.colorAttachmentCount == 0 && desc.depthStencil.texture.isNull()) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "no attachment to render to");
    }
    if (hasSwapchain && desc.colorAttachmentCount == 0) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "a swapchain's image is color attachment 0, so there is at least one");
    }
    for (std::uint32_t i = 0; i < desc.colorAttachmentCount; ++i) {
        const bool swapchains = i == 0 && hasSwapchain;
        const bool named = !desc.colorAttachments.at(i).texture.isNull();
        if (swapchains && named) {
            return core::failure(core::ErrorCode::RhiInvalidCall,
                                 "attachment 0 is the swapchain's image; its texture is null");
        }
        if (!swapchains && !named) {
            return core::failure(core::ErrorCode::RhiInvalidCall, "color attachment {} names no texture", i);
        }
    }
    return {};
}

/// Fails with RhiInvalidCall when data is not whole 32-bit values or more
/// than the push constants hold. Each backend's pushConstants calls it first.
[[nodiscard]] inline core::Result<void> checkPushConstants(std::span<const std::byte> data) {
    constexpr std::size_t kValueBytes = sizeof(std::uint32_t);
    if (data.size() % kValueBytes != 0) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "{} bytes of push constants is not whole 32-bit values",
                             data.size());
    }
    if (data.size() > kPushConstantsSize) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "{} bytes of push constants; there are {}", data.size(),
                             kPushConstantsSize);
    }
    return {};
}

/// What a recording list has had bound, which the usage checks every backend
/// makes of a draw or dispatch read; each backend keeps one per list.
struct RecordState {
    bool rendering = false;
    bool graphicsBound = false;
    bool computeBound = false;
    bool indexBound = false;
};

/// Fails with RhiInvalidCall when a viewport, scissor or draw is recorded
/// outside rendering.
[[nodiscard]] inline core::Result<void> checkRendering(const RecordState& state) {
    if (!state.rendering) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "no rendering is open");
    }
    return {};
}

/// Fails with RhiInvalidCall when a draw is recorded outside rendering or
/// with no graphics pipeline bound, and, when indexed, no index buffer bound.
[[nodiscard]] inline core::Result<void> checkDraw(const RecordState& state, bool indexed) {
    if (auto checked = checkRendering(state); !checked) {
        return checked;
    }
    if (!state.graphicsBound) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "no graphics pipeline is bound");
    }
    if (indexed && !state.indexBound) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "no index buffer is bound");
    }
    return {};
}

/// Fails with RhiInvalidCall when a dispatch is recorded inside rendering or
/// with no compute pipeline bound.
[[nodiscard]] inline core::Result<void> checkDispatch(const RecordState& state) {
    if (state.rendering) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "a dispatch is recorded outside rendering");
    }
    if (!state.computeBound) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "no compute pipeline is bound");
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

} // namespace bfr::rhi
