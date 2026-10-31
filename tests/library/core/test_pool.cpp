#include "bfr/core/handle.h"
#include "bfr/core/log.h"
#include "bfr/core/pool.h"
#include "log_capture.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using bfr::core::Handle;
using bfr::core::Pool;

namespace {

// More than any test here makes.
constexpr std::uint32_t kLimit = 16 * Pool<int>::kPageSize;

} // namespace

TEST_CASE("Pool finds what it stores", "[core][unit]") {
    Pool<int> pool{"test", 1, kLimit};

    const Handle<int> handle = pool.emplace(4);

    CHECK_FALSE(handle.isNull());
    REQUIRE(pool.find(handle) != nullptr);
    CHECK(*pool.find(handle) == 4);
    CHECK(pool.size() == 1);
}

TEST_CASE("Pool visits every live value and no retired one", "[core][unit]") {
    Pool<int> pool{"test", 1, kLimit};
    const Handle<int> kept = pool.emplace(4);
    const Handle<int> gone = pool.emplace(5);
    REQUIRE(pool.retire(gone, 1));

    std::vector<int> seen;
    pool.forEach([&seen](int& value) { seen.push_back(value); });

    CHECK(seen == std::vector<int>{4});
    CHECK(pool.find(kept) != nullptr);
}

TEST_CASE("Pool finds nothing for a retired handle", "[core][unit]") {
    Pool<int> pool{"test", 1, kLimit};
    const Handle<int> handle = pool.emplace(4);

    CHECK(pool.retire(handle, 0));

    CHECK(pool.find(handle) == nullptr);
    CHECK_FALSE(pool.retire(handle, 0));
    CHECK(pool.size() == 0);
}

TEST_CASE("Pool keeps a retired value until its submission ID completes", "[core][unit]") {
    constexpr std::uint64_t kSubmissionId = 5;
    const auto shared = std::make_shared<int>(0);
    Pool<std::shared_ptr<int>> pool{"test", 1, kLimit};
    REQUIRE(pool.retire(pool.emplace(shared), kSubmissionId));

    pool.collect(kSubmissionId - 1);
    CHECK(shared.use_count() == 2);

    pool.collect(kSubmissionId);
    CHECK(shared.use_count() == 1);
}

TEST_CASE("Pool erase destroys the value before it returns", "[core][unit]") {
    const auto shared = std::make_shared<int>(0);
    Pool<std::shared_ptr<int>> pool{"test", 1, 1};
    const Handle<std::shared_ptr<int>> handle = pool.emplace(shared);

    CHECK(pool.erase(handle));

    CHECK(shared.use_count() == 1);
    CHECK(pool.find(handle) == nullptr);
    CHECK(pool.size() == 0);
    CHECK_FALSE(pool.erase(handle));
    CHECK_FALSE(pool.retire(handle, 0));
    CHECK_FALSE(pool.erase(Handle<std::shared_ptr<int>>{}));
    // Its slot is free again, so the maximum of one is not reached.
    CHECK_FALSE(pool.emplace(shared).isNull());
}

TEST_CASE("Pool returns a null handle at its maximum, exactly", "[core][unit]") {
    // Not a whole number of pages, so rounding would show.
    constexpr std::uint32_t kMax = 3;
    Pool<int> pool{"test", 1, kMax};
    for (std::uint32_t added = 0; added < kMax; ++added) {
        REQUIRE_FALSE(pool.emplace(0).isNull());
    }

    CHECK(pool.emplace(0).isNull());
}

TEST_CASE("Pool counts retired values against its maximum until collected", "[core][unit]") {
    Pool<int> pool{"test", 1, 1};
    REQUIRE(pool.retire(pool.emplace(0), 1));

    CHECK(pool.emplace(0).isNull());
    pool.collect(1);
    CHECK_FALSE(pool.emplace(0).isNull());
}

TEST_CASE("Pool gives a reused slot a new generation", "[core][unit]") {
    Pool<int> pool{"test", 1, kLimit};
    const Handle<int> first = pool.emplace(1);
    REQUIRE(pool.retire(first, 0));
    pool.collect(0);

    const Handle<int> second = pool.emplace(2);

    CHECK(second.index() == first.index());
    CHECK(second.generation() != first.generation());
    CHECK(pool.find(first) == nullptr);
    REQUIRE(pool.find(second) != nullptr);
    CHECK(*pool.find(second) == 2);
}

TEST_CASE("Handle packs its index under its generation", "[core][unit]") {
    constexpr std::uint32_t kIndex = 5;
    constexpr std::uint64_t kGeneration = std::uint64_t{1} << 36;
    const Handle<int> handle{kIndex, kGeneration};

    CHECK(handle.index() == kIndex);
    CHECK(handle.generation() == kGeneration);
    CHECK(handle.bits == (kGeneration << Handle<int>::kIndexBits | kIndex));
    CHECK_FALSE(handle.isNull());
}

TEST_CASE("Pool finds nothing for a generation past 32 bits that a slot never had", "[core][unit]") {
    Pool<int> pool{"test", 1, kLimit};
    const Handle<int> handle = pool.emplace(1);
    const Handle<int> forged{handle.index(), handle.generation() + (std::uint64_t{1} << 32)};

    CHECK(forged != handle);
    CHECK(pool.find(forged) == nullptr);
    CHECK_FALSE(pool.retire(forged, 0));
}

TEST_CASE("Pool caps its maximum at what a handle can name", "[core][unit]") {
    const Pool<int> pool{"test", 0, std::numeric_limits<std::uint32_t>::max()};

    CHECK(pool.capacity() == 0);
    CHECK(Handle<int>::kMaxIndices == (std::uint32_t{1} << Handle<int>::kIndexBits));
}

TEST_CASE("Pool finds nothing for a null handle", "[core][unit]") {
    Pool<int> pool{"test", 1, kLimit};

    CHECK(Handle<int>{}.isNull());
    CHECK(pool.find(Handle<int>{}) == nullptr);
    pool.emplace(1);
    CHECK(pool.find(Handle<int>{}) == nullptr);
    CHECK_FALSE(pool.retire(Handle<int>{}, 0));
}

TEST_CASE("Pool preallocates its capacity", "[core][unit]") {
    if (!bfr::core::kLoggingEnabled) {
        SKIP("logging is compiled out");
    }
    std::ostringstream log;
    const bfr::core::Logger logger = loggingInto(log);
    constexpr std::uint32_t kCapacity = 100;
    Pool<int> pool{"test", kCapacity, kLimit, logger};
    REQUIRE(pool.capacity() >= kCapacity);

    for (std::uint32_t added = 0; added < pool.capacity(); ++added) {
        pool.emplace(0);
    }

    CHECK(log.str().empty());
}

TEST_CASE("Pool warns with its new capacity when it grows", "[core][unit]") {
    if (!bfr::core::kLoggingEnabled) {
        SKIP("logging is compiled out");
    }
    std::ostringstream log;
    const bfr::core::Logger logger = loggingInto(log);
    Pool<int> pool{"surfaces", 1, kLimit, logger};
    const std::uint32_t before = pool.capacity();

    for (std::uint32_t added = 0; added <= before; ++added) {
        pool.emplace(0);
    }

    CHECK(pool.capacity() == before + Pool<int>::kPageSize);
    CHECK_THAT(log.str(), Catch::Matchers::ContainsSubstring("Pool surfaces grew to " +
                                                             std::to_string(before + Pool<int>::kPageSize)));
}

TEST_CASE("Pool keeps entries in place as it grows", "[core][unit]") {
    Pool<int> pool{"test", 1, kLimit};
    const Handle<int> first = pool.emplace(1);
    const int* address = pool.find(first);

    for (std::uint32_t added = 0; added < 2 * Pool<int>::kPageSize; ++added) {
        pool.emplace(0);
    }

    CHECK(pool.find(first) == address);
}

TEST_CASE("Pool clear destroys every value", "[core][unit]") {
    const auto shared = std::make_shared<int>(0);
    Pool<std::shared_ptr<int>> pool{"test", 1, kLimit};
    pool.emplace(shared);
    pool.emplace(shared);

    pool.clear();

    CHECK(shared.use_count() == 1);
    CHECK(pool.size() == 0);
}

TEST_CASE("Pool clear destroys retired values whatever their submission ID", "[core][unit]") {
    const auto shared = std::make_shared<int>(0);
    Pool<std::shared_ptr<int>> pool{"test", 1, kLimit};
    REQUIRE(pool.retire(pool.emplace(shared), std::numeric_limits<std::uint64_t>::max()));

    pool.clear();

    CHECK(shared.use_count() == 1);
}

namespace {

// Uses its pool while it is made and destroyed, which a pool that held its
// lock meanwhile would deadlock on.
struct UsesPool {
    explicit UsesPool(Pool<UsesPool>& owner) : pool(&owner), capacitySeen(owner.capacity()) {}
    ~UsesPool() { static_cast<void>(pool->capacity()); }
    UsesPool(const UsesPool&) = delete;
    UsesPool& operator=(const UsesPool&) = delete;
    UsesPool(UsesPool&&) = delete;
    UsesPool& operator=(UsesPool&&) = delete;

    Pool<UsesPool>* pool;
    std::uint32_t capacitySeen;
};

} // namespace

TEST_CASE("Pool lets a value use the pool while it is made and destroyed", "[core][unit]") {
    Pool<UsesPool> pool{"test", 1, kLimit};

    const Handle<UsesPool> handle = pool.emplace(pool);
    REQUIRE(pool.find(handle) != nullptr);
    CHECK(pool.find(handle)->capacitySeen == Pool<UsesPool>::kPageSize);
    CHECK(pool.retire(handle, 0));
    pool.collect(0);
    CHECK(pool.size() == 0);
}

TEST_CASE("Pool destroys its values when it goes", "[core][unit]") {
    const auto shared = std::make_shared<int>(0);
    {
        Pool<std::shared_ptr<int>> pool{"test", 1, kLimit};
        pool.emplace(shared);
    }

    CHECK(shared.use_count() == 1);
}

namespace {

// Enough that the pool grows while several threads use it.
constexpr std::uint32_t kEachThread = 3 * Pool<int>::kPageSize;

// Emplaces kEachThread values from first on, finding each and ending every
// other one, by retire and collect or by erase in turn, so that their slots
// are reused; returns how many calls went wrong.
std::uint32_t useFromOneThread(Pool<std::uint32_t>& pool, std::uint32_t first) {
    std::uint32_t wrong = 0;
    std::vector<Handle<std::uint32_t>> kept;
    kept.reserve(kEachThread / 2);
    for (std::uint32_t value = first; value < first + kEachThread; ++value) {
        const Handle<std::uint32_t> handle = pool.emplace(value);
        const std::uint32_t* found = pool.find(handle);
        wrong += (found == nullptr || *found != value) ? 1U : 0U;
        if (value % 4 == 0) {
            wrong += pool.retire(handle, 0) ? 0U : 1U;
            pool.collect(0);
        }
        else if (value % 2 == 0) {
            wrong += pool.erase(handle) ? 0U : 1U;
        }
        else {
            kept.push_back(handle);
        }
    }
    for (const Handle<std::uint32_t> handle : kept) {
        wrong += pool.find(handle) == nullptr ? 1U : 0U;
    }
    return wrong;
}

} // namespace

TEST_CASE("Pool takes emplace, find, retire, collect and erase from many threads at once", "[core][unit]") {
    constexpr std::uint32_t kThreads = 4;
    Pool<std::uint32_t> pool{"test", 1, kLimit};
    std::atomic<std::uint32_t> wrong{0};
    {
        std::vector<std::jthread> threads;
        threads.reserve(kThreads);
        for (std::uint32_t thread = 0; thread < kThreads; ++thread) {
            threads.emplace_back([&pool, &wrong, thread] { wrong += useFromOneThread(pool, thread * kEachThread); });
        }
    }

    CHECK(wrong == 0);
    CHECK(pool.size() == kThreads * kEachThread / 2);
}
