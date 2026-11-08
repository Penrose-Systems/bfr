#pragma once

#include "bfr/rhi/command_list.h"
#include "bfr/rhi/format.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <tuple>

namespace bfr::rhi {

/// The formats whose channels are signed integers.
inline constexpr std::array kSignedIntegerFormats{
    Format::R8Sint,     Format::Rg8Sint,  Format::Rgba8Sint, Format::R16Sint,    Format::Rg16Sint,
    Format::Rgba16Sint, Format::R32Sint,  Format::Rg32Sint,  Format::Rgb32Sint,  Format::Rgba32Sint,
    Format::R64Sint,    Format::Rg64Sint, Format::Rgb64Sint, Format::Rgba64Sint,
};

/// The color formats whose channels are unsigned integers.
inline constexpr std::array kUnsignedIntegerFormats{
    Format::R8Uint,     Format::Rg8Uint,  Format::Rgba8Uint, Format::R16Uint,    Format::Rg16Uint,
    Format::Rgba16Uint, Format::R32Uint,  Format::Rg32Uint,  Format::Rgb32Uint,  Format::Rgba32Uint,
    Format::R64Uint,    Format::Rg64Uint, Format::Rgb64Uint, Format::Rgba64Uint, Format::Rgb10a2Uint,
};

/// Whether format is a color format of integers, signed or not.
[[nodiscard]] constexpr bool integerFormat(Format format) {
    return std::ranges::find(kSignedIntegerFormats, format) != kSignedIntegerFormats.end() ||
           std::ranges::find(kUnsignedIntegerFormats, format) != kUnsignedIntegerFormats.end();
}

/// Red, green, blue and alpha as numbers, which a double holds each of exactly.
using ClearValues = std::array<double, std::tuple_size_v<decltype(ColorAttachment::clearColor)>>;

/// What attachment clears a target of format to, for the backends that take
/// a clear as numbers: clearColor, or, when clearInteger is set,
/// clearColorInteger's bits, read as int32 for a signed integer format and
/// as uint32 for any other.
[[nodiscard]] inline ClearValues clearValues(const ColorAttachment& attachment, Format format) {
    ClearValues values{};
    if (!attachment.clearInteger) {
        std::ranges::copy(attachment.clearColor, values.begin());
        return values;
    }
    const bool signedInteger = std::ranges::find(kSignedIntegerFormats, format) != kSignedIntegerFormats.end();
    std::ranges::transform(attachment.clearColorInteger, values.begin(), [signedInteger](std::uint32_t bits) {
        return signedInteger ? static_cast<double>(static_cast<std::int32_t>(bits)) : static_cast<double>(bits);
    });
    return values;
}

} // namespace bfr::rhi
