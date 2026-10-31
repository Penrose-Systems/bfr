#pragma once

#include <cstddef>

namespace bfr::test {

/// Counts the heap allocations and frees this thread makes through operator
/// new and delete while it lives. The test program replaces the global
/// operators to count them; under a sanitizer, which replaces them itself,
/// nothing is counted. malloc and free, which drivers call, are not counted.
class AllocationCounter {
public:
    AllocationCounter();
    ~AllocationCounter();
    AllocationCounter(const AllocationCounter&) = delete;
    AllocationCounter& operator=(const AllocationCounter&) = delete;
    AllocationCounter(AllocationCounter&&) = delete;
    AllocationCounter& operator=(AllocationCounter&&) = delete;

    /// False under a sanitizer, when nothing is counted.
    [[nodiscard]] static bool counts();
    [[nodiscard]] std::size_t allocations() const;
    [[nodiscard]] std::size_t frees() const;

private:
    std::size_t allocationsBefore_;
    std::size_t freesBefore_;
    bool wasCounting_;
};

} // namespace bfr::test
