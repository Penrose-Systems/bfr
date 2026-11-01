#pragma once

#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

namespace bfr::core {

/// count slots, numbered from 0, handed out in chains: a chain is its first
/// slot, and next gives the one after each, kEnd after the last.
///
/// Why: the frame path takes and gives back slots, for command lists and the
/// swapchains they render to, and must not allocate or free. The links are one
/// array made here, so taking and giving are only index updates under a lock,
/// and a caller keeps whatever a slot holds in its own array of count, by slot.
///
/// Threads: every call is safe from any thread at once. Walking a chain with
/// next is the chain owner's to synchronize against giving it back.
class SlotChains {
public:
    static constexpr std::uint32_t kEnd = std::numeric_limits<std::uint32_t>::max();

    explicit SlotChains(std::uint32_t count) : next_(count), free_(count == 0 ? kEnd : 0) {
        for (std::uint32_t slot = 0; slot < count; ++slot) {
            std::span{next_}[slot] = slot + 1 == count ? kEnd : slot + 1;
        }
        available_ = count;
    }

    /// A chain of count free slots; kEnd for none. Nothing when fewer than count are free.
    [[nodiscard]] std::optional<std::uint32_t> take(std::uint32_t count) {
        const std::scoped_lock lock{mutex_};
        if (count > available_) {
            return std::nullopt;
        }
        const std::uint32_t first = count == 0 ? kEnd : free_;
        std::uint32_t last = kEnd;
        for (std::uint32_t taken = 0; taken < count; ++taken) {
            last = free_;
            free_ = std::span{next_}[free_];
        }
        if (last != kEnd) {
            std::span{next_}[last] = kEnd;
        }
        available_ -= count;
        return first;
    }

    /// Takes one free slot and puts it at the head of the chain head names,
    /// which may be kEnd. Nothing, leaving head as it was, when none is free.
    [[nodiscard]] std::optional<std::uint32_t> push(std::uint32_t& head) {
        const std::scoped_lock lock{mutex_};
        if (available_ == 0) {
            return std::nullopt;
        }
        const std::uint32_t slot = free_;
        free_ = std::span{next_}[slot];
        std::span{next_}[slot] = head;
        head = slot;
        --available_;
        return slot;
    }

    /// Gives back the whole chain from first, which may be kEnd.
    void give(std::uint32_t first) {
        const std::scoped_lock lock{mutex_};
        for (std::uint32_t slot = first; slot != kEnd;) {
            const std::uint32_t after = std::span{next_}[slot];
            std::span{next_}[slot] = free_;
            free_ = slot;
            ++available_;
            slot = after;
        }
    }

    /// The slot after slot in its chain; kEnd after the last.
    [[nodiscard]] std::uint32_t next(std::uint32_t slot) const { return std::span{next_}[slot]; }

    [[nodiscard]] std::uint32_t available() const {
        const std::scoped_lock lock{mutex_};
        return available_;
    }

private:
    // Made once, here, at its final size.
    std::vector<std::uint32_t> next_;
    mutable std::mutex mutex_;
    std::uint32_t free_;
    std::uint32_t available_ = 0;
};

} // namespace bfr::core
