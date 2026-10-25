#pragma once

#include <cstdint>

namespace bfr::core {

/// Names one entry of a Pool. The generation tells an erased entry's handles
/// from its slot's next occupant; a default handle, generation 0, is null.
template <class T>
struct Handle {
    std::uint32_t index = 0;
    std::uint32_t generation = 0;

    [[nodiscard]] constexpr bool isNull() const { return generation == 0; }
    constexpr bool operator==(const Handle&) const = default;
};

} // namespace bfr::core
