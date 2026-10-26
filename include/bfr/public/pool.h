#pragma once

#include "handle.h"
#include "log.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace bfr::core {

/// Owns values of T, each named by a Handle<Tag>.
///
/// Room: capacity values are allocated up front, in whole pages. Past that the
/// pool grows a page at a time, logging a warning with its new capacity, a
/// sign that capacity should be larger. It holds at most maxCapacity values,
/// counting retired ones not yet collected; past that emplace fails. Values
/// never move while they live.
///
/// Threads: every call is safe from any thread at once. find takes no lock.
/// The other calls lock only to take or give back a slot; values are made and
/// destroyed outside the lock, so a slow constructor or destructor holds no
/// one up and may use the pool itself. Using one value from two threads at
/// once is the caller's to synchronize.
///
/// Lifetime: retire ends a handle at once, so find stops finding it, but its
/// value lives on until collect is told that the token it was retired with has
/// completed. The device retires against GPU progress, so nothing the GPU may
/// still use is destroyed.
template <class T, class Tag = T>
class Pool {
public:
    static constexpr std::uint32_t kPageSize = 64;

    /// Room for capacity values now, rounded up to whole pages, and for
    /// maxCapacity at most. name labels the growth warning and must outlive the
    /// pool.
    Pool(std::string_view name, std::uint32_t capacity, std::uint32_t maxCapacity)
        : name_(name), maxCapacity_(maxCapacity), pages_(pagesFor(maxCapacity)) {
        const std::scoped_lock lock{mutex_};
        addPages(std::min(pagesFor(capacity), maxPages()));
    }

    ~Pool() { clear(); }
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;
    Pool(Pool&&) = delete;
    Pool& operator=(Pool&&) = delete;

    /// Makes a value from args and returns its handle; a null handle when the
    /// pool is at its maximum.
    template <class... Args>
    Handle<Tag> emplace(Args&&... args) {
        const std::uint32_t index = take();
        if (index == kNoFree) {
            return {};
        }
        Entry& entry = entryAt(index);
        try {
            std::construct_at(&entry.value, std::forward<Args>(args)...);
        } catch (...) {
            const std::scoped_lock lock{mutex_};
            give(index);
            throw;
        }
        // Odd: live. Released, so a find that sees it sees the value made.
        std::atomic<std::uint32_t>& generation = generationAt(index);
        const std::uint32_t live = generation.load(std::memory_order_relaxed) + 1;
        generation.store(live, std::memory_order_release);
        size_.fetch_add(1, std::memory_order_relaxed);
        return {.index = index, .generation = live};
    }

    /// The value handle names; null when handle is null, retired, or never
    /// made by this pool. Takes no lock.
    [[nodiscard]] T* find(Handle<Tag> handle) {
        if ((handle.generation & 1U) == 0 || (handle.index >> kPageShift) >= maxPages()) {
            return nullptr;
        }
        Page* page = pages_[handle.index >> kPageShift].load(std::memory_order_acquire);
        if (page == nullptr) {
            return nullptr;
        }
        const std::uint32_t slot = handle.index & (kPageSize - 1);
        if (std::span{page->generations}[slot].load(std::memory_order_acquire) != handle.generation) {
            return nullptr;
        }
        return &std::span{page->entries}[slot].value;
    }

    /// Ends handle at once; its value is destroyed by the first collect told
    /// that token has completed. False when handle is null or already retired.
    bool retire(Handle<Tag> handle, std::uint64_t token) {
        const std::scoped_lock lock{mutex_};
        // Under the lock, so of two retires of one handle only one passes.
        if (find(handle) == nullptr) {
            return false;
        }
        retireLocked(handle.index, token);
        return true;
    }

    /// Destroys the values retired with a token at or before completed and
    /// frees their slots.
    void collect(std::uint64_t completed) {
        std::vector<std::uint32_t> ready;
        {
            const std::scoped_lock lock{mutex_};
            const auto done = std::ranges::partition(retired_, [completed](const Retired& retired) {
                return retired.token > completed;
            });
            ready.reserve(done.size());
            for (const Retired& retired : done) {
                ready.push_back(retired.index);
            }
            retired_.erase(done.begin(), done.end());
        }
        for (const std::uint32_t index : ready) {
            std::destroy_at(&entryAt(index).value);
        }
        const std::scoped_lock lock{mutex_};
        for (const std::uint32_t index : ready) {
            give(index);
        }
    }

    /// Retires every live value and destroys every retired one, whatever its
    /// token: for when nothing can still be using them.
    void clear() {
        {
            const std::scoped_lock lock{mutex_};
            for (std::uint32_t index = 0; index < pageCount_ * kPageSize; ++index) {
                if ((generationAt(index).load(std::memory_order_relaxed) & 1U) != 0) {
                    retireLocked(index, 0);
                }
            }
        }
        collect(std::numeric_limits<std::uint64_t>::max());
    }

    /// Live values: made and not yet retired.
    [[nodiscard]] std::uint32_t size() const { return size_.load(std::memory_order_relaxed); }

    [[nodiscard]] std::uint32_t capacity() const {
        const std::scoped_lock lock{mutex_};
        return pageCount_ * kPageSize;
    }

private:
    static_assert(std::has_single_bit(kPageSize));
    static constexpr std::uint32_t kPageShift = std::countr_zero(kPageSize);
    static constexpr std::uint32_t kNoFree = std::numeric_limits<std::uint32_t>::max();

    // A live or retired entry holds its value; a free one, the next free index.
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
    // odd generation is live; even is free or retired; 0, a null handle's, is
    // never live.
    struct Page {
        std::array<std::atomic<std::uint32_t>, kPageSize> generations{};
        std::array<Entry, kPageSize> entries;
    };

    struct Retired {
        std::uint32_t index;
        std::uint64_t token;
    };

    static std::uint32_t pagesFor(std::uint32_t count) { return (count + kPageSize - 1) / kPageSize; }

    [[nodiscard]] std::uint32_t maxPages() const { return static_cast<std::uint32_t>(pages_.size()); }

    Page& pageAt(std::uint32_t index) { return *pages_[index >> kPageShift].load(std::memory_order_acquire); }

    Entry& entryAt(std::uint32_t index) { return std::span{pageAt(index).entries}[index & (kPageSize - 1)]; }

    std::atomic<std::uint32_t>& generationAt(std::uint32_t index) {
        return std::span{pageAt(index).generations}[index & (kPageSize - 1)];
    }

    // Takes a free slot, growing a page if there is none; kNoFree at the maximum.
    std::uint32_t take() {
        const std::scoped_lock lock{mutex_};
        if (taken_ == maxCapacity_) {
            return kNoFree;
        }
        if (free_ == kNoFree) {
            addPages(1);
            BFR_LOG_WARN("Pool {} grew to {} entries", name_, pageCount_ * kPageSize);
        }
        const std::uint32_t index = free_;
        free_ = entryAt(index).nextFree;
        ++taken_;
        return index;
    }

    // Puts a slot whose value is gone at the head of the free list. Locked.
    void give(std::uint32_t index) {
        pushFree(index);
        --taken_;
    }

    // Puts a slot at the head of the free list. Locked.
    void pushFree(std::uint32_t index) {
        entryAt(index).nextFree = free_;
        free_ = index;
    }

    // Moves a live slot to its next, even, generation and queues its value.
    // Locked.
    void retireLocked(std::uint32_t index, std::uint64_t token) {
        std::atomic<std::uint32_t>& generation = generationAt(index);
        generation.store(generation.load(std::memory_order_relaxed) + 1, std::memory_order_release);
        retired_.push_back({.index = index, .token = token});
        size_.fetch_sub(1, std::memory_order_relaxed);
    }

    // Adds count pages; their slots join the free list in index order. Locked.
    void addPages(std::uint32_t count) {
        const std::uint32_t first = pageCount_ * kPageSize;
        for (std::uint32_t added = 0; added < count; ++added) {
            owned_.push_back(std::make_unique<Page>());
            pages_[pageCount_].store(owned_.back().get(), std::memory_order_release);
            ++pageCount_;
        }
        for (std::uint32_t index = pageCount_ * kPageSize; index-- > first;) {
            pushFree(index);
        }
    }

    std::string_view name_;
    std::uint32_t maxCapacity_;
    // One per page the pool may ever have, made once and never resized, so find
    // reads it without the lock. A page is published here once it is ready.
    std::vector<std::atomic<Page*>> pages_;
    std::atomic<std::uint32_t> size_{0};
    // Guards everything below.
    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<Page>> owned_;
    std::uint32_t pageCount_ = 0;
    std::uint32_t free_ = kNoFree;
    // Slots off the free list: live, being made, or retired and not collected.
    std::uint32_t taken_ = 0;
    std::vector<Retired> retired_;
};

} // namespace bfr::core
