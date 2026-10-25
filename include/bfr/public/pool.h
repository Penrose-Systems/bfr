#pragma once

#include "handle.h"
#include "log.h"

#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace bfr::core {

/// Owns values of T, each named by a Handle<Tag>. The capacity given at
/// construction is allocated up front; past it the pool grows a page at a time
/// and logs a warning with its new capacity, a sign that capacity should be
/// larger. Values never move while they live. Erasing a value moves its slot to
/// the next generation, so the value's handles stop finding anything.
template <class T, class Tag = T>
class Pool {
public:
    static constexpr std::uint32_t kPageSize = 64;

    /// Room for capacity values, rounded up to whole pages. name labels the
    /// growth warning and must outlive the pool.
    Pool(std::string_view name, std::uint32_t capacity) : name_(name) {
        const std::uint32_t pages = (capacity + kPageSize - 1) / kPageSize;
        if (pages != 0) {
            addPages(pages);
        }
    }

    ~Pool() { destroyLive(); }
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;
    Pool(Pool&&) = delete;
    Pool& operator=(Pool&&) = delete;

    /// Makes a value from args and returns its handle.
    template <class... Args>
    Handle<Tag> emplace(Args&&... args) {
        if (free_ == kNoFree) {
            addPages(1);
            BFR_LOG_WARN("Pool {} grew to {} entries", name_, capacity_);
        }
        const std::uint32_t index = free_;
        Entry& entry = entryAt(index);
        const std::uint32_t next = entry.nextFree;
        std::construct_at(&entry.value, std::forward<Args>(args)...);
        free_ = next;
        std::uint32_t& generation = generationAt(index);
        ++generation; // odd: live
        ++size_;
        return {.index = index, .generation = generation};
    }

    /// The value handle names; null when handle is null or erased.
    [[nodiscard]] T* find(Handle<Tag> handle) { return live(handle) ? &entryAt(handle.index).value : nullptr; }

    /// Destroys the value handle names; false when handle is null or erased.
    bool erase(Handle<Tag> handle) {
        if (!live(handle)) {
            return false;
        }
        release(handle.index);
        --size_;
        return true;
    }

    /// Destroys every value; their handles stop finding anything.
    void clear() {
        destroyLive();
        size_ = 0;
    }

    [[nodiscard]] std::uint32_t size() const { return size_; }
    [[nodiscard]] std::uint32_t capacity() const { return capacity_; }

private:
    static_assert(std::has_single_bit(kPageSize));
    static constexpr std::uint32_t kPageShift = std::countr_zero(kPageSize);
    static constexpr std::uint32_t kNoFree = std::numeric_limits<std::uint32_t>::max();

    // A live entry holds its value; a free one, the next free index.
    union Entry {
        Entry() : nextFree(kNoFree) {}
        ~Entry() {}
        Entry(const Entry&) = delete;
        Entry& operator=(const Entry&) = delete;
        Entry(Entry&&) = delete;
        Entry& operator=(Entry&&) = delete;

        std::uint32_t nextFree;
        T value;
    };

    // Generations apart from entries, so checking a handle reads only them. An
    // odd generation is live; 0, a null handle's, never is.
    struct Page {
        std::array<std::uint32_t, kPageSize> generations{};
        std::array<Entry, kPageSize> entries;
    };

    Entry& entryAt(std::uint32_t index) {
        return std::span{pages_[index >> kPageShift]->entries}[index & (kPageSize - 1)];
    }

    std::uint32_t& generationAt(std::uint32_t index) {
        return std::span{pages_[index >> kPageShift]->generations}[index & (kPageSize - 1)];
    }

    [[nodiscard]] bool live(Handle<Tag> handle) {
        return handle.index < capacity_ && (handle.generation & 1U) != 0 &&
               generationAt(handle.index) == handle.generation;
    }

    // Destroys the value and puts its slot at the head of the free list.
    void release(std::uint32_t index) {
        Entry& entry = entryAt(index);
        std::destroy_at(&entry.value);
        entry.nextFree = free_;
        free_ = index;
        ++generationAt(index); // even: free
    }

    void destroyLive() {
        for (std::uint32_t index = 0; index < capacity_; ++index) {
            if ((generationAt(index) & 1U) != 0) {
                release(index);
            }
        }
    }

    // Adds count pages; their slots join the free list in index order.
    void addPages(std::uint32_t count) {
        for (std::uint32_t page = 0; page < count; ++page) {
            pages_.push_back(std::make_unique<Page>());
        }
        const std::uint32_t first = capacity_;
        capacity_ += count * kPageSize;
        for (std::uint32_t index = capacity_; index-- > first;) {
            entryAt(index).nextFree = free_;
            free_ = index;
        }
    }

    std::string_view name_;
    // Grows only when the pool does.
    std::vector<std::unique_ptr<Page>> pages_;
    std::uint32_t capacity_ = 0;
    std::uint32_t size_ = 0;
    std::uint32_t free_ = kNoFree;
};

} // namespace bfr::core
