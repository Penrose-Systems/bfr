#include "core/submissions.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <tuple>
#include <vector>

using bfr::core::Submissions;

namespace {

constexpr std::uint32_t kQueues = 2;

// The timeline values each queue last submitted and has passed, as a backend
// reports them, and which queues completed asked.
struct Timelines {
    std::array<std::uint64_t, kQueues> submitted{};
    std::array<std::uint64_t, kQueues> passed{};
    std::vector<std::uint32_t> asked;

    [[nodiscard]] auto submittedReader() const {
        return [this](std::uint32_t queue) {
            return submitted.at(queue);
        };
    }
    auto reader() {
        return [this](std::uint32_t queue) {
            asked.push_back(queue);
            return passed.at(queue);
        };
    }
};

} // namespace

TEST_CASE("Submissions has passed everything before any ID is marked", "[core][unit]") {
    Submissions submissions{kQueues};
    Timelines timelines;

    CHECK(submissions.completed(timelines.reader()) == 0);
    CHECK(timelines.asked.empty());
}

TEST_CASE("Submissions numbers the points it marks from 1", "[core][unit]") {
    Submissions submissions{kQueues};
    const Timelines timelines;

    CHECK(submissions.mark(timelines.submittedReader()) == 1);
    CHECK(submissions.mark(timelines.submittedReader()) == 2);
}

TEST_CASE("Submissions has passed an ID once every queue has passed what it had submitted then", "[core][unit]") {
    Submissions submissions{kQueues};
    Timelines timelines;
    timelines.submitted = {1, 0};
    const std::uint64_t first = submissions.mark(timelines.submittedReader());
    timelines.submitted = {2, 1};
    const std::uint64_t second = submissions.mark(timelines.submittedReader());

    CHECK(submissions.completed(timelines.reader()) == 0);

    timelines.passed = {1, 0};
    CHECK(submissions.completed(timelines.reader()) == first);

    // Queue 0 has passed the second ID's value, but queue 1 has not.
    timelines.passed = {2, 0};
    CHECK(submissions.completed(timelines.reader()) == first);

    timelines.passed = {2, 1};
    CHECK(submissions.completed(timelines.reader()) == second);
}

TEST_CASE("Submissions has passed an ID marked before any submit at once", "[core][unit]") {
    Submissions submissions{kQueues};
    Timelines timelines;
    const std::uint64_t id = submissions.mark(timelines.submittedReader());

    CHECK(submissions.completed(timelines.reader()) == id);
}

TEST_CASE("Submissions asks the queues only while an ID is outstanding", "[core][unit]") {
    Submissions submissions{kQueues};
    Timelines timelines;
    timelines.submitted = {0, 1};
    std::ignore = submissions.mark(timelines.submittedReader());
    timelines.passed = {0, 1};

    CHECK(submissions.completed(timelines.reader()) == 1);
    CHECK(timelines.asked == std::vector<std::uint32_t>{0, 1});

    timelines.asked.clear();
    CHECK(submissions.completed(timelines.reader()) == 1);
    CHECK(timelines.asked.empty());
}
