#pragma once

#include <array>
#include <cstddef>
#include <span>

namespace bfr::core {

/// Up to Capacity values held in place, for a value made and destroyed on the
/// frame path, which must not allocate or free. push fails rather than grow.
template <class T, std::size_t Capacity>
class InlineVector {
public:
    /// Adds value; false, adding nothing, when full.
    [[nodiscard]] bool push(const T& value) {
        if (size_ == Capacity) {
            return false;
        }
        std::span{values_}[size_++] = value;
        return true;
    }

    [[nodiscard]] std::size_t size() const { return size_; }
    [[nodiscard]] bool empty() const { return size_ == 0; }
    [[nodiscard]] auto begin() const { return values_.begin(); }
    [[nodiscard]] auto end() const { return values_.begin() + static_cast<std::ptrdiff_t>(size_); }

private:
    std::array<T, Capacity> values_{};
    std::size_t size_ = 0;
};

} // namespace bfr::core
