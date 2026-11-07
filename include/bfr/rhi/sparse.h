#pragma once

#include "bfr/core/handle.h"
#include "bfr/rhi/buffer.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/texture.h"

#include <cstdint>
#include <span>

namespace bfr::rhi {

/// Memory that sparse textures' and buffers' tiles are mapped into, in device
/// memory. Only named, by a SparseMemoryHandle.
struct SparseMemory;

using SparseMemoryHandle = core::Handle<SparseMemory>;

/// How to create sparse memory, for the tiles of resources like texture and
/// buffer, either of which may be null but not both: Vulkan takes the
/// memory's type from their requirements, and D3D12, where
/// Sparse::mixedResources is false, the one kind of resource its heap holds.
/// A resource mapped to it later must take what they took, as each backend
/// says.
struct SparseMemoryDesc {
    /// Bytes.
    std::uint64_t size = 0;
    /// One of Sparse::tileBytes, or 0 where that list is empty.
    std::uint32_t tileBytes = 0;
    TextureHandle texture;
    BufferHandle buffer;
};

/// A tile's size, in texels.
struct TileShape {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t depth = 0;
};

/// How one aspect of a sparse texture is cut into tiles, as the backend says.
/// Mip levels from firstTailMip on are its tail, mapped as tiles of
/// tailBytes rather than by region.
struct TextureTiling {
    /// Bytes in one tile.
    std::uint64_t tileBytes = 0;
    TileShape shape;
    std::uint32_t firstTailMip = 0;
    /// Bytes of one layer's tail.
    std::uint64_t tailBytes = 0;
    /// Bytes between the starts of two layers' tails; 0 where the backend gives none.
    std::uint64_t tailStride = 0;
    /// Every layer shares one tail, mapped at layer 0.
    bool singleTail = false;
};

/// How a sparse buffer is cut into tiles.
struct BufferTiling {
    /// Bytes in one tile.
    std::uint64_t tileBytes = 0;
};

/// The texture a sparseFormat call asks about, as TextureDesc would make it.
struct SparseFormatDesc {
    Format format = Format::Undefined;
    TextureDimension dimension = TextureDimension::D2;
    std::uint32_t samples = 0;
    TextureUsage usage;
    TextureAspect aspect = TextureAspect::Color;
    /// As TextureDesc::tileBytes.
    std::uint32_t tileBytes = 0;
};

/// Whether a sparse texture of a format can be made, and its tiles' shape.
struct SparseFormat {
    bool supported = false;
    TileShape shape;
    /// Every layer shares one tail.
    bool singleTail = false;
    /// Every mip level not a whole number of tiles is in the tail.
    bool alignedMipSize = false;
};

/// What a tile mapping does to its region. The values are the C API's.
enum class TileMapping : std::uint32_t {
    /// Maps the region to the memory's tiles.
    Map = 0,
    /// Leaves the region with no memory; memoryTile is unused.
    Unmap = 1,
};

/// One region of a sparse texture mapped to memory or unmapped. Below the
/// tail the region is x, y, z, width, height and depth, in tiles of the mip
/// level; from TextureTiling::firstTailMip on, it is tailTiles tiles from
/// tailTile in the layer's tail. memoryTile is where in memory, in the
/// texture's tiles. Laid out as bfr_texture_tile_region.
struct TextureTileRegion {
    TileMapping mapping = TileMapping::Map;
    TextureAspect aspect = TextureAspect::Color;
    std::uint32_t mipLevel = 0;
    std::uint32_t layer = 0;
    std::uint32_t x = 0;
    std::uint32_t y = 0;
    std::uint32_t z = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t depth = 0;
    std::uint64_t tailTile = 0;
    std::uint64_t tailTiles = 0;
    std::uint64_t memoryTile = 0;
};

/// tiles tiles of a sparse buffer from tile mapped to memory from
/// memoryTile, or unmapped, in the buffer's tiles. Laid out as
/// bfr_buffer_tile_region.
struct BufferTileRegion {
    TileMapping mapping = TileMapping::Map;
    std::uint64_t tile = 0;
    std::uint64_t tiles = 0;
    std::uint64_t memoryTile = 0;
};

/// What Device::updateTileMappings sends to one queue: textureRegions of
/// texture and bufferRegions of buffer mapped to memory or unmapped, after
/// the GPU passes every wait. A null resource has no regions; memory may be
/// null when every region unmaps.
struct TileMappingDesc {
    std::uint32_t queue = 0;
    std::span<const Token> waits;
    SparseMemoryHandle memory;
    TextureHandle texture;
    std::span<const TextureTileRegion> textureRegions;
    BufferHandle buffer;
    std::span<const BufferTileRegion> bufferRegions;
};

} // namespace bfr::rhi
