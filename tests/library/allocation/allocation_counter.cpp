#include "allocation/allocation_counter.h"

#include <cstddef>
#include <cstdlib>
#include <new>

#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
    #define BFR_TEST_SANITIZED
#elif defined(__has_feature)
    #if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
        #define BFR_TEST_SANITIZED
    #endif
#endif

namespace {

#ifdef BFR_TEST_SANITIZED
constexpr bool kCounts = false;
#else
constexpr bool kCounts = true;
#endif

// This thread's counts, and whether a counter is open on it.
struct ThreadCounts {
    bool counting = false;
    std::size_t allocations = 0;
    std::size_t frees = 0;
};

ThreadCounts& threadCounts() {
    thread_local ThreadCounts counts;
    return counts;
}

} // namespace

namespace bfr::test {

AllocationCounter::AllocationCounter()
    : allocationsBefore_(threadCounts().allocations),
      freesBefore_(threadCounts().frees),
      wasCounting_(threadCounts().counting) {
    threadCounts().counting = true;
}

AllocationCounter::~AllocationCounter() {
    threadCounts().counting = wasCounting_;
}

bool AllocationCounter::counts() {
    return kCounts;
}

std::size_t AllocationCounter::allocations() const {
    return threadCounts().allocations - allocationsBefore_;
}

std::size_t AllocationCounter::frees() const {
    return threadCounts().frees - freesBefore_;
}

} // namespace bfr::test

#ifndef BFR_TEST_SANITIZED

// The replaceable global operators, counting on the thread that asked.
namespace {

void* allocate(std::size_t size) {
    if (ThreadCounts& counts = threadCounts(); counts.counting) {
        ++counts.allocations;
    }
    void* memory = std::malloc(size == 0 ? 1 : size);
    if (memory == nullptr) {
        throw std::bad_alloc{};
    }
    return memory;
}

void* allocateAligned(std::size_t size, std::align_val_t alignment) {
    if (ThreadCounts& counts = threadCounts(); counts.counting) {
        ++counts.allocations;
    }
    const auto align = static_cast<std::size_t>(alignment);
    const std::size_t rounded = ((size == 0 ? 1 : size) + align - 1) / align * align;
    void* memory = std::aligned_alloc(align, rounded);
    if (memory == nullptr) {
        throw std::bad_alloc{};
    }
    return memory;
}

void release(void* memory) noexcept {
    if (memory == nullptr) {
        return;
    }
    if (ThreadCounts& counts = threadCounts(); counts.counting) {
        ++counts.frees;
    }
    std::free(memory);
}

} // namespace

void* operator new(std::size_t size) {
    return allocate(size);
}
void* operator new[](std::size_t size) {
    return allocate(size);
}
void* operator new(std::size_t size, std::align_val_t alignment) {
    return allocateAligned(size, alignment);
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
    return allocateAligned(size, alignment);
}
void* operator new(std::size_t size, const std::nothrow_t& /*unused*/) noexcept {
    try {
        return allocate(size);
    } catch (...) {
        return nullptr;
    }
}
void* operator new[](std::size_t size, const std::nothrow_t& /*unused*/) noexcept {
    try {
        return allocate(size);
    } catch (...) {
        return nullptr;
    }
}
void operator delete(void* memory) noexcept {
    release(memory);
}
void operator delete[](void* memory) noexcept {
    release(memory);
}
void operator delete(void* memory, std::size_t /*size*/) noexcept {
    release(memory);
}
void operator delete[](void* memory, std::size_t /*size*/) noexcept {
    release(memory);
}
void operator delete(void* memory, std::align_val_t /*alignment*/) noexcept {
    release(memory);
}
void operator delete[](void* memory, std::align_val_t /*alignment*/) noexcept {
    release(memory);
}
void operator delete(void* memory, std::size_t /*size*/, std::align_val_t /*alignment*/) noexcept {
    release(memory);
}
void operator delete[](void* memory, std::size_t /*size*/, std::align_val_t /*alignment*/) noexcept {
    release(memory);
}

#endif
