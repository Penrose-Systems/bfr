#pragma once

#include <cstdint>

namespace bfr::core {

/// Names one entry of a Pool: 64 bits, the slot's index in the low kIndexBits
/// and its generation above, which tells an erased entry's handles from the
/// slot's next occupant. The same bits as the C API's handle. A default
/// handle, generation 0, is null. Each type is its own C struct, so a handle
/// carries no type tag.
template <class T>
struct Handle {
    static constexpr unsigned kIndexBits = 24;
    static constexpr std::uint64_t kIndexMask = (std::uint64_t{1} << kIndexBits) - 1;
    /// The most slots a pool may have.
    static constexpr std::uint32_t kMaxIndices = std::uint32_t{1} << kIndexBits;

    constexpr Handle() = default;
    constexpr Handle(std::uint32_t index, std::uint64_t generation)
        : bits((generation << kIndexBits) | (index & kIndexMask)) {}
    /// The handle whose bits these are, as the C API carries them.
    [[nodiscard]] static constexpr Handle fromBits(std::uint64_t bits) {
        Handle handle;
        handle.bits = bits;
        return handle;
    }

    [[nodiscard]] constexpr std::uint32_t index() const { return static_cast<std::uint32_t>(bits & kIndexMask); }
    [[nodiscard]] constexpr std::uint64_t generation() const { return bits >> kIndexBits; }
    [[nodiscard]] constexpr bool isNull() const { return generation() == 0; }
    constexpr bool operator==(const Handle&) const = default;

    std::uint64_t bits = 0;
};

} // namespace bfr::core
