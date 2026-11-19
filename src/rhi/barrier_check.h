#pragma once

#include "bfr/core/error.h"
#include "bfr/rhi/texture.h"

#include <cstdint>

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

} // namespace bfr::rhi
