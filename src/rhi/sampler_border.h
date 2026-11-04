#pragma once

#include "bfr/rhi/sampler.h"

#include <array>
#include <cstdint>
#include <type_traits>

namespace bfr::rhi {

/// Which of the three border colours every backend names a sampler desc
/// asks for, or Custom for any other: Vulkan and Metal set those three as
/// presets, and Metal has no other.
enum class BorderPreset : std::uint8_t {
    TransparentBlack,
    OpaqueBlack,
    OpaqueWhite,
    Custom,
};

/// The preset desc's border colour is, read from borderColorInteger when
/// borderInteger is set and from borderColor otherwise.
[[nodiscard]] inline BorderPreset borderPreset(const SamplerDesc& desc) {
    const auto match = [](const auto& color) {
        using T = typename std::decay_t<decltype(color)>::value_type;
        const auto is = [&color](T rgb, T alpha) {
            return color == std::array<T, kBorderColorComponents>{rgb, rgb, rgb, alpha};
        };
        if (is(T{0}, T{0})) {
            return BorderPreset::TransparentBlack;
        }
        if (is(T{0}, T{1})) {
            return BorderPreset::OpaqueBlack;
        }
        if (is(T{1}, T{1})) {
            return BorderPreset::OpaqueWhite;
        }
        return BorderPreset::Custom;
    };
    return desc.borderInteger ? match(desc.borderColorInteger) : match(desc.borderColor);
}

} // namespace bfr::rhi
