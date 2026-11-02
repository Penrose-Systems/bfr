#pragma once

#include "bfr/rhi/format.h"
#include "bfr/rhi/texture.h"

#include <algorithm>
#include <array>
#include <cstdint>

namespace bfr::rhi {

/// How a format's texels are grouped for copies: bytes in a block of width by
/// height texels, and the bytes a texel of its depth and stencil take in a
/// buffer, which a copy moves one at a time. No API answers this as a query,
/// so it is the API's own data: vk.xml's blockSize and blockExtent for the
/// format's Vulkan equivalent, and the Vulkan spec's depth/stencil copy sizes
/// (16-bit depth 2 bytes, other depth 4, stencil 1). bytes is 0 for a depth
/// or stencil format, which has no color.
struct FormatBlock {
    Format format = Format::R8Unorm;
    std::uint32_t bytes = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t depthBytes = 0;
    std::uint32_t stencilBytes = 0;
};

inline constexpr std::array kFormatBlocks{
    FormatBlock{.format = Format::R8Unorm, .bytes = 1, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::R8Snorm, .bytes = 1, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::R8Uint, .bytes = 1, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::R8Sint, .bytes = 1, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Rg8Unorm, .bytes = 2, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Rg8Snorm, .bytes = 2, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Rg8Uint, .bytes = 2, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Rg8Sint, .bytes = 2, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Rgba8Unorm, .bytes = 4, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Rgba8Srgb, .bytes = 4, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Rgba8Snorm, .bytes = 4, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Rgba8Uint, .bytes = 4, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Rgba8Sint, .bytes = 4, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Bgra8Unorm, .bytes = 4, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Bgra8Srgb, .bytes = 4, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::R16Unorm, .bytes = 2, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::R16Snorm, .bytes = 2, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::R16Uint, .bytes = 2, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::R16Sint, .bytes = 2, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::R16Float, .bytes = 2, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Rg16Unorm, .bytes = 4, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Rg16Snorm, .bytes = 4, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Rg16Uint, .bytes = 4, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Rg16Sint, .bytes = 4, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Rg16Float, .bytes = 4, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Rgba16Unorm, .bytes = 8, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Rgba16Snorm, .bytes = 8, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Rgba16Uint, .bytes = 8, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Rgba16Sint, .bytes = 8, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Rgba16Float, .bytes = 8, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::R32Uint, .bytes = 4, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::R32Sint, .bytes = 4, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::R32Float, .bytes = 4, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Rg32Uint, .bytes = 8, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Rg32Sint, .bytes = 8, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Rg32Float, .bytes = 8, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Rgb32Uint, .bytes = 12, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Rgb32Sint, .bytes = 12, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Rgb32Float, .bytes = 12, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Rgba32Uint, .bytes = 16, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Rgba32Sint, .bytes = 16, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Rgba32Float, .bytes = 16, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Rgb10a2Unorm, .bytes = 4, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Rgb10a2Uint, .bytes = 4, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Bgr10a2Unorm, .bytes = 4, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Rg11b10Float, .bytes = 4, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Rgb9e5Float, .bytes = 4, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::B5g6r5Unorm, .bytes = 2, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Bgr5a1Unorm, .bytes = 2, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Bgra4Unorm, .bytes = 2, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::D16Unorm, .bytes = 0, .width = 1, .height = 1, .depthBytes = 2, .stencilBytes = 0},
    FormatBlock{
        .format = Format::D24UnormS8Uint, .bytes = 0, .width = 1, .height = 1, .depthBytes = 4, .stencilBytes = 1},
    FormatBlock{.format = Format::D32Float, .bytes = 0, .width = 1, .height = 1, .depthBytes = 4, .stencilBytes = 0},
    FormatBlock{
        .format = Format::D32FloatS8Uint, .bytes = 0, .width = 1, .height = 1, .depthBytes = 4, .stencilBytes = 1},
    FormatBlock{.format = Format::S8Uint, .bytes = 0, .width = 1, .height = 1, .depthBytes = 0, .stencilBytes = 1},
    FormatBlock{.format = Format::Bc1Unorm, .bytes = 8, .width = 4, .height = 4, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Bc1Srgb, .bytes = 8, .width = 4, .height = 4, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Bc2Unorm, .bytes = 16, .width = 4, .height = 4, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Bc2Srgb, .bytes = 16, .width = 4, .height = 4, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Bc3Unorm, .bytes = 16, .width = 4, .height = 4, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Bc3Srgb, .bytes = 16, .width = 4, .height = 4, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Bc4Unorm, .bytes = 8, .width = 4, .height = 4, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Bc4Snorm, .bytes = 8, .width = 4, .height = 4, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Bc5Unorm, .bytes = 16, .width = 4, .height = 4, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Bc5Snorm, .bytes = 16, .width = 4, .height = 4, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Bc6hUfloat, .bytes = 16, .width = 4, .height = 4, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Bc6hSfloat, .bytes = 16, .width = 4, .height = 4, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Bc7Unorm, .bytes = 16, .width = 4, .height = 4, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::Bc7Srgb, .bytes = 16, .width = 4, .height = 4, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Etc2Rgb8Unorm, .bytes = 8, .width = 4, .height = 4, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Etc2Rgb8Srgb, .bytes = 8, .width = 4, .height = 4, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Etc2Rgb8a1Unorm, .bytes = 8, .width = 4, .height = 4, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Etc2Rgb8a1Srgb, .bytes = 8, .width = 4, .height = 4, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Etc2Rgba8Unorm, .bytes = 16, .width = 4, .height = 4, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Etc2Rgba8Srgb, .bytes = 16, .width = 4, .height = 4, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::EacR11Unorm, .bytes = 8, .width = 4, .height = 4, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{.format = Format::EacR11Snorm, .bytes = 8, .width = 4, .height = 4, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::EacRg11Unorm, .bytes = 16, .width = 4, .height = 4, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::EacRg11Snorm, .bytes = 16, .width = 4, .height = 4, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc4x4Unorm, .bytes = 16, .width = 4, .height = 4, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc4x4Srgb, .bytes = 16, .width = 4, .height = 4, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc4x4Float, .bytes = 16, .width = 4, .height = 4, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc5x4Unorm, .bytes = 16, .width = 5, .height = 4, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc5x4Srgb, .bytes = 16, .width = 5, .height = 4, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc5x4Float, .bytes = 16, .width = 5, .height = 4, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc5x5Unorm, .bytes = 16, .width = 5, .height = 5, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc5x5Srgb, .bytes = 16, .width = 5, .height = 5, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc5x5Float, .bytes = 16, .width = 5, .height = 5, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc6x5Unorm, .bytes = 16, .width = 6, .height = 5, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc6x5Srgb, .bytes = 16, .width = 6, .height = 5, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc6x5Float, .bytes = 16, .width = 6, .height = 5, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc6x6Unorm, .bytes = 16, .width = 6, .height = 6, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc6x6Srgb, .bytes = 16, .width = 6, .height = 6, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc6x6Float, .bytes = 16, .width = 6, .height = 6, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc8x5Unorm, .bytes = 16, .width = 8, .height = 5, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc8x5Srgb, .bytes = 16, .width = 8, .height = 5, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc8x5Float, .bytes = 16, .width = 8, .height = 5, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc8x6Unorm, .bytes = 16, .width = 8, .height = 6, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc8x6Srgb, .bytes = 16, .width = 8, .height = 6, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc8x6Float, .bytes = 16, .width = 8, .height = 6, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc8x8Unorm, .bytes = 16, .width = 8, .height = 8, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc8x8Srgb, .bytes = 16, .width = 8, .height = 8, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc8x8Float, .bytes = 16, .width = 8, .height = 8, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc10x5Unorm, .bytes = 16, .width = 10, .height = 5, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc10x5Srgb, .bytes = 16, .width = 10, .height = 5, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc10x5Float, .bytes = 16, .width = 10, .height = 5, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc10x6Unorm, .bytes = 16, .width = 10, .height = 6, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc10x6Srgb, .bytes = 16, .width = 10, .height = 6, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc10x6Float, .bytes = 16, .width = 10, .height = 6, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc10x8Unorm, .bytes = 16, .width = 10, .height = 8, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc10x8Srgb, .bytes = 16, .width = 10, .height = 8, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc10x8Float, .bytes = 16, .width = 10, .height = 8, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc10x10Unorm, .bytes = 16, .width = 10, .height = 10, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc10x10Srgb, .bytes = 16, .width = 10, .height = 10, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc10x10Float, .bytes = 16, .width = 10, .height = 10, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc12x10Unorm, .bytes = 16, .width = 12, .height = 10, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc12x10Srgb, .bytes = 16, .width = 12, .height = 10, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc12x10Float, .bytes = 16, .width = 12, .height = 10, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc12x12Unorm, .bytes = 16, .width = 12, .height = 12, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc12x12Srgb, .bytes = 16, .width = 12, .height = 12, .depthBytes = 0, .stencilBytes = 0},
    FormatBlock{
        .format = Format::Astc12x12Float, .bytes = 16, .width = 12, .height = 12, .depthBytes = 0, .stencilBytes = 0},
};

static_assert(kFormatBlocks.size() == kFormats.size());

/// format's block. Every Format has one.
[[nodiscard]] constexpr const FormatBlock& blockOf(Format format) {
    return *std::ranges::find(kFormatBlocks, format, &FormatBlock::format);
}

/// How many blocks of size texels cover texels.
[[nodiscard]] constexpr std::uint32_t blocksOver(std::uint32_t texels, std::uint32_t size) {
    return (texels + size - 1) / size;
}

/// The bytes a block of format's aspect takes in a buffer.
[[nodiscard]] constexpr std::uint32_t bytesOf(const FormatBlock& block, TextureAspect aspect) {
    switch (aspect) {
        case TextureAspect::Depth:
            return block.depthBytes;
        case TextureAspect::Stencil:
            return block.stencilBytes;
        case TextureAspect::Color:
        default:
            return block.bytes;
    }
}

} // namespace bfr::rhi
