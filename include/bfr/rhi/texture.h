#pragma once

#include "bfr/bfr.h"
#include "bfr/bfr_enums.hpp" // IWYU pragma: export
#include "bfr/core/handle.h"
#include "bfr/rhi/format.h"

#include <cstdint>
#include <string_view>

namespace bfr::rhi {

/// Texels the GPU reads and writes. Only named, by a TextureHandle.
struct Texture;

using TextureHandle = core::Handle<Texture>;

/// Faces of a cube: the layers a Cube texture has per cube.
inline constexpr std::uint32_t kCubeFaces = 6;

/// What the GPU does with a texture. Set every use it will have.
struct TextureUsage {
    bool copySrc = false;
    bool copyDst = false;
    bool sampled = false;
    bool storage = false;
    bool colorAttachment = false;
    bool depthStencil = false;
    /// Shaders use atomics on it as a storage texture; Metal needs it set.
    bool storageAtomic = false;
};

/// A bindless slot a resource doesn't have; no table has a slot this high.
inline constexpr std::uint32_t kNoSlot = BFR_NO_SLOT;

/// Every mip level, or every layer, from the first one a TextureRange names.
inline constexpr std::uint32_t kRemaining = BFR_REMAINING;

/// The mip levels and layers of a texture a barrier moves: mipCount levels
/// from mipLevel, and layerCount layers from layer, in every plane. Counts
/// pass through as given; 0 is no count Vulkan takes. A 3D texture has one
/// layer, so a barrier moves every depth slice of its levels.
struct TextureRange {
    std::uint32_t mipLevel = 0;
    std::uint32_t mipCount = kRemaining;
    std::uint32_t layer = 0;
    std::uint32_t layerCount = kRemaining;
};

/// Where a buffer is in the bindless tables api/bfr.yaml states under
/// bfr_pipeline, from creation to destruction: the index shaders use. A table
/// the buffer's usage does not set has no slot for it, and reads kNoSlot.
struct BufferSlots {
    /// Its index in the storage buffer table; only with storage usage.
    std::uint32_t storage = kNoSlot;
    /// Its index in the uniform buffer table; only with uniform usage.
    std::uint32_t uniform = kNoSlot;
};

/// Where a texture is in the bindless tables api/bfr.yaml states under
/// bfr_pipeline, from creation to destruction: the index shaders use. A table
/// the texture's usage does not set has no slot for it, and reads kNoSlot.
struct TextureSlots {
    /// Its index in the texture table; only with sampled usage. A
    /// depth-stencil texture's slot reads its depth; no stencil.
    std::uint32_t sampled = kNoSlot;
    /// Its index in the storage texture table; only with storage usage.
    std::uint32_t storage = kNoSlot;
};

/// How to create a texture, in device memory. Sizes are in texels; height is
/// 1 for D1, depth is 1 but for D3, layers is 1 for D3 and a multiple of 6
/// for Cube. More than one layer, or more than 6 for Cube, makes an array
/// texture, which shaders declare as one.
struct TextureDesc {
    TextureDimension dimension = TextureDimension::D2;
    Format format = Format::Undefined;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t depth = 0;
    std::uint32_t layers = 0;
    std::uint32_t mipLevels = 0;
    std::uint32_t samples = 0;
    TextureUsage usage;
    /// Made with no memory: tiles of SparseMemory are mapped into it by
    /// Device::updateTileMappings, as Caps::sparse and Device::sparseFormat say.
    bool sparse = false;
    /// Its tiles may be mapped to memory other sparse resources' tiles are
    /// mapped to, where Sparse::aliased.
    bool sparseAliased = false;
    /// One of Sparse::tileBytes, or 0 where that list is empty; only when sparse.
    std::uint32_t tileBytes = 0;
    /// What a debugger or profiler shows it as, where Features::objectNames;
    /// read during the call only. Empty is none. The create call fails with
    /// RhiUnsupported for a name where Features::objectNames is false,
    /// RhiInvalidCall for one holding a '\0', which the backend reads as its
    /// end, and RhiDeviceFailed when the backend refuses it.
    std::string_view name{};
};

/// One region copied between a buffer and one layer and mip level of a
/// texture; see api/bfr.yaml for the alignment its offsets keep. Laid out as
/// bfr_buffer_texture_copy.
struct BufferTextureCopy {
    std::uint64_t bufferOffset = 0;
    /// Bytes between the starts of two rows of texel blocks in the buffer.
    std::uint64_t bytesPerRow = 0;
    /// Rows of texels between the starts of two depth slices in the buffer.
    std::uint32_t rowsPerImage = 0;
    std::uint32_t mipLevel = 0;
    std::uint32_t layer = 0;
    TextureAspect aspect = TextureAspect::Color;
    std::uint32_t x = 0;
    std::uint32_t y = 0;
    std::uint32_t z = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t depth = 0;
};

/// One region copied between two textures of one format and sample count,
/// every aspect of it. Laid out as bfr_texture_copy.
struct TextureCopy {
    std::uint32_t srcMipLevel = 0;
    std::uint32_t srcLayer = 0;
    std::uint32_t srcX = 0;
    std::uint32_t srcY = 0;
    std::uint32_t srcZ = 0;
    std::uint32_t dstMipLevel = 0;
    std::uint32_t dstLayer = 0;
    std::uint32_t dstX = 0;
    std::uint32_t dstY = 0;
    std::uint32_t dstZ = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t depth = 0;
};

} // namespace bfr::rhi
