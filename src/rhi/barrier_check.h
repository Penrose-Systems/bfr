#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/barrier.h"
#include "bfr/rhi/texture.h"
#include "rhi/enum_check.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>

namespace bfr::rhi {

/// Fails with RhiInvalidCall when range names a mip level or layer the
/// texture, of mipLevels levels and layers layers, does not have, or a count
/// of 0. Each backend's texture barrier calls it once it has looked the
/// texture up. Vulkan's barrier takes a first level and layer the image has
/// (VUID-VkImageMemoryBarrier2-subresourceRange-01486, -01488), counts that
/// end within it (-01724, -01725) and no count of 0
/// (VUID-VkImageSubresourceRange-levelCount-01720, -layerCount-01721). D3D12
/// reads a NumMipLevels of 0 as naming one subresource by index instead
/// (Enhanced Barriers spec, D3D12_BARRIER_SUBRESOURCE_RANGE), so a count of 0
/// or one cut to 0 at the end would move another subresource. A range past
/// the texture is refused on every backend alike.
[[nodiscard]] inline core::Result<void> checkTextureRange(const TextureRange& range, std::uint32_t mipLevels,
                                                          std::uint32_t layers) {
    const auto within = [](std::uint32_t first, std::uint32_t count, std::uint32_t total) {
        return first < total && count != 0 && (count == kRemaining || std::uint64_t{first} + count <= total);
    };
    if (!within(range.mipLevel, range.mipCount, mipLevels)) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "a barrier moves {} mip levels from level {}; the texture has {}", range.mipCount,
                             range.mipLevel, mipLevels);
    }
    if (!within(range.layer, range.layerCount, layers)) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "a barrier moves {} layers from layer {}; the texture has {}", range.layerCount,
                             range.layer, layers);
    }
    return {};
}

/// The most barriers of each kind a backend translates into one native
/// barrier call, in storage it holds on the stack: a desc with more is
/// recorded in several calls, each its own dependency, so the size bounds the
/// stack and refuses nothing. Not measured: it only needs to hold the few
/// barriers a pass usually has in one call.
inline constexpr std::size_t kBarrierChunk = 16;

/// How many native calls kBarrierChunk storage records desc in.
[[nodiscard]] inline std::size_t barrierChunks(const BarrierDesc& desc) {
    const std::size_t most = std::max(desc.globals.size(), desc.textures.size());
    return (most + kBarrierChunk - 1) / kBarrierChunk;
}

/// The barriers of all that chunk, counted from 0, holds: kBarrierChunk of
/// them, fewer or none at the end.
template <class Barrier>
[[nodiscard]] std::span<const Barrier> barrierChunk(std::span<const Barrier> all, std::size_t chunk) {
    const std::size_t first = std::min(chunk * kBarrierChunk, all.size());
    return all.subspan(first, std::min(kBarrierChunk, all.size() - first));
}

/// Whether a stage or access flag is set: any value but BFR_FALSE, as C
/// writes true.
[[nodiscard]] constexpr bool set(bfr_bool flag) {
    return flag != BFR_FALSE;
}

/// Fails with RhiInvalidCall when a texture barrier of desc has a layout BFR
/// doesn't have. Every backend's barrier calls it before it looks anything
/// up.
[[nodiscard]] inline core::Result<void> checkBarrierLayouts(const BarrierDesc& desc) {
    for (const TextureBarrier& barrier : desc.textures) {
        if (auto checked = checkLayouts(barrier.layoutBefore, barrier.layoutAfter); !checked) {
            return checked;
        }
    }
    return {};
}

} // namespace bfr::rhi
