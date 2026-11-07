#pragma once

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <limits>

namespace bfr::rhi {

/// A side of mip level of a texture size texels across: at least 1, and 1
/// past the levels a 32-bit size has.
[[nodiscard]] constexpr std::uint32_t levelSize(std::uint32_t size, std::uint32_t level) {
    return level < std::numeric_limits<std::uint32_t>::digits ? std::max(1U, size >> level) : 1U;
}

/// What remains of a mip level level texels across from offset on, at most
/// wanted: a sparse region is whole tiles or reaches the level's edge, as
/// Vulkan's and Metal's mappings take it in texels.
template <std::unsigned_integral Size>
[[nodiscard]] constexpr Size extentWithin(Size wanted, Size offset, Size level) {
    return offset < level ? std::min(wanted, level - offset) : wanted;
}

} // namespace bfr::rhi
