#pragma once

#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

namespace bfr::core {

/// A vector whose storage for capacity values is made once, when it is
/// constructed. push fails rather than grow, and clear keeps the storage, so
/// nothing after construction allocates or frees.
template <class T>
class FixedVector {
public:
    FixedVector() = default;
    explicit FixedVector(std::size_t capacity) : capacity_(capacity) { values_.reserve(capacity); }

    /// Adds value; false, adding nothing, when full.
    [[nodiscard]] bool push(const T& value) {
        if (values_.size() == capacity_) {
            return false;
        }
        values_.push_back(value);
        return true;
    }

    /// Removes every value equal to value.
    void erase(const T& value) { std::erase(values_, value); }
    void clear() { values_.clear(); }

    [[nodiscard]] std::size_t size() const { return values_.size(); }
    [[nodiscard]] bool empty() const { return values_.empty(); }
    [[nodiscard]] T* data() { return values_.data(); }
    [[nodiscard]] const T* data() const { return values_.data(); }
    [[nodiscard]] T& operator[](std::size_t index) { return values_[index]; }
    [[nodiscard]] const T& operator[](std::size_t index) const { return values_[index]; }
    [[nodiscard]] auto begin() { return values_.begin(); }
    [[nodiscard]] auto end() { return values_.end(); }
    [[nodiscard]] auto begin() const { return values_.begin(); }
    [[nodiscard]] auto end() const { return values_.end(); }

private:
    std::size_t capacity_ = 0;
    std::vector<T> values_;
};

} // namespace bfr::core
