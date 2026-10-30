#pragma once

#include <concepts>
#include <ranges>
#include <string>
#include <string_view>

namespace bfr::core::text {

/// The items with separator between each pair: join(names, ", ") is "a, b".
template <std::ranges::input_range Items>
    requires std::convertible_to<std::ranges::range_reference_t<const Items&>, std::string_view>
[[nodiscard]] std::string join(const Items& items, std::string_view separator) {
    std::string text;
    bool first = true;
    for (const std::string_view item : items) {
        if (!first) {
            text += separator;
        }
        text += item;
        first = false;
    }
    return text;
}

} // namespace bfr::core::text
