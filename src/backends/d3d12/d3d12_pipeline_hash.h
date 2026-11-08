#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace bfr::rhi::d3d12 {

/// A 64-bit FNV-1a hash of what is added, in order: the same values give the
/// same hash in every run, which names a pipeline in a pipeline library.
/// Values are added one scalar at a time, so no struct's padding is hashed.
class PipelineHash {
public:
    void add(std::span<const std::byte> bytes) {
        for (const std::byte byte : bytes) {
            value_ = (value_ ^ std::to_integer<std::uint64_t>(byte)) * kPrime;
        }
    }

    template <class T>
        requires std::is_scalar_v<T>
    void add(T value) {
        add(std::as_bytes(std::span{&value, 1}));
    }

    [[nodiscard]] std::uint64_t value() const { return value_; }

private:
    static constexpr std::uint64_t kOffsetBasis = 0xcbf29ce484222325;
    static constexpr std::uint64_t kPrime = 0x100000001b3;
    std::uint64_t value_ = kOffsetBasis;
};

} // namespace bfr::rhi::d3d12
