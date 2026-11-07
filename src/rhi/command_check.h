#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/pipeline.h"
#include "rhi/enum_check.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace bfr::rhi {

/// Fails with RhiInvalidCall when desc asks for a rendering no backend can
/// begin: a load or store op BFR doesn't have, more color attachments than there are, none at all, a color
/// attachment naming no texture, or, with a swapchain, no color attachment or
/// attachment 0 naming a texture when the swapchain's image is it. Each
/// backend's beginRendering calls it first, then looks its handles up.
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

} // namespace bfr::rhi
