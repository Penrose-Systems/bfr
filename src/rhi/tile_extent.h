#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

namespace bfr::rhi {

/// A side of mip level of a texture size texels across: at least 1, and 1
/// past the levels a 32-bit size has.
[[nodiscard]] constexpr std::uint32_t levelSize(std::uint32_t size, std::uint32_t level) {
    return level < std::numeric_limits<std::uint32_t>::digits ? std::max(1U, size >> level) : 1U;
}

} // namespace bfr::rhi
