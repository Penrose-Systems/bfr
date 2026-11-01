#pragma once

#include <cstdint>
#include <mutex>
#include <vector>

namespace bfr::core {

/// A device's submission IDs, each naming a point in its submit history: at
/// that point, the timeline value last submitted on every queue. A handle
/// table retires an object against mark(), taken as the object is destroyed,
/// and collects what completed() says every queue has passed, so an object
/// lives until the GPU has run every submit made before it was destroyed.
///
/// Submits record nothing here. Only destroying and collecting do, and
/// neither is on the frame path. Every call is safe from any thread.
class Submissions {
public:
    explicit Submissions(std::uint32_t queueCount) : queueCount_(queueCount), passed_(queueCount) {}

    /// The ID of now: submitted(queue) gives the timeline value last
    /// submitted on queue. IDs count up from 1.
    template <class Submitted>
    [[nodiscard]] std::uint64_t mark(const Submitted& submitted) {
        const std::scoped_lock lock{mutex_};
        for (std::uint32_t queue = 0; queue < queueCount_; ++queue) {
            values_.push_back(submitted(queue));
        }
        return nextId_++;
    }

    /// The newest ID every queue has passed, 0 before any has. completedValue
    /// (queue) gives the timeline value the GPU has passed on queue; it is
    /// asked only when an ID is outstanding, and must not call this.
    template <class CompletedValue>
    [[nodiscard]] std::uint64_t completed(const CompletedValue& completedValue) {
        const std::scoped_lock lock{mutex_};
        if (values_.empty()) {
            return nextId_ - 1;
        }
        for (std::uint32_t queue = 0; queue < queueCount_; ++queue) {
            passed_[queue] = completedValue(queue);
        }
        // A queue's values only rise from one ID to the next, so the IDs
        // passed are the oldest ones.
        std::size_t done = 0;
        while (done < values_.size() && isPassed(done)) {
            done += queueCount_;
        }
        values_.erase(values_.begin(), values_.begin() + static_cast<std::ptrdiff_t>(done));
        firstId_ += done / queueCount_;
        return firstId_ - 1;
    }

private:
    // Whether every queue has passed the ID whose values start at first.
    [[nodiscard]] bool isPassed(std::size_t first) const {
        for (std::uint32_t queue = 0; queue < queueCount_; ++queue) {
            if (values_[first + queue] > passed_[queue]) {
                return false;
            }
        }
        return true;
    }

    mutable std::mutex mutex_;
    std::uint32_t queueCount_;
    // The first outstanding ID, and the next one mark gives.
    std::uint64_t firstId_ = 1;
    std::uint64_t nextId_ = 1;
    // queueCount_ values for each outstanding ID, oldest first.
    std::vector<std::uint64_t> values_;
    // What completedValue gave for each queue, made once.
    std::vector<std::uint64_t> passed_;
};

} // namespace bfr::core
