#pragma once

#include "bfr/core/handle.h"
#include "bfr/rhi/format.h"

#include <cstdint>

namespace bfr::rhi {

/// Texels the GPU reads and writes. Only named, by a TextureHandle.
struct Texture;

using TextureHandle = core::Handle<Texture>;

/// Faces of a cube: the layers a Cube texture has per cube.
inline constexpr std::uint32_t kCubeFaces = 6;

/// The shape of a texture. The values are the C API's.
enum class TextureDimension : std::uint8_t {
    D2 = 0,
    D1 = 1,
    D3 = 2,
    /// Six square faces per cube, in layers: a multiple of 6, face by face.
    Cube = 3,
};

/// How a texture's memory is arranged for what the GPU does with it next.
/// Device::textureBarrier moves a texture between layouts; BFR keeps no
/// record of the one it is in. The values are the C API's.
enum class TextureLayout : std::uint8_t {
    /// Its contents are not kept. Only a layout to move from.
    Undefined = 0,
    CopySrc = 1,
    CopyDst = 2,
    /// Shaders read it through the texture table.
    Sampled = 3,
    /// Shaders read and write it through the storage texture table.
    Storage = 4,
    /// Rendering writes color to it.
    ColorTarget = 5,
    /// Rendering tests and writes depth or stencil in it.
    DepthStencilTarget = 6,
};

/// Which part of a texel a copy moves. The values are the C API's, and so is
/// the size: BufferTextureCopy is read in place from bfr_buffer_texture_copy.
enum class TextureAspect : std::int32_t { // NOLINT(performance-enum-size)
    Color = 0,
    Depth = 1,
    Stencil = 2,
};

/// What the GPU does with a texture. Set every use it will have.
struct TextureUsage {
    bool copySrc = false;
    bool copyDst = false;
    bool sampled = false;
    bool storage = false;
    bool colorAttachment = false;
    bool depthStencil = false;
};

/// Where a texture is in the bindless tables api/bfr.yaml states under
/// bfr_pipeline, from creation to destruction: the index shaders use. A table
/// the texture's usage does not set has no slot for it, and reads 0.
struct TextureSlots {
    /// Its index in the texture table; only with sampled usage.
    std::uint32_t sampled = 0;
    /// Its index in the storage texture table; only with storage usage.
    std::uint32_t storage = 0;
};

/// How to create a texture, in device memory. Sizes are in texels; height is
/// 1 for D1, depth is 1 but for D3, layers is 1 for D3 and a multiple of 6
/// for Cube.
struct TextureDesc {
    TextureDimension dimension = TextureDimension::D2;
    Format format = Format::R8Unorm;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t depth = 0;
    std::uint32_t layers = 0;
    std::uint32_t mipLevels = 0;
    std::uint32_t samples = 0;
    TextureUsage usage;
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
