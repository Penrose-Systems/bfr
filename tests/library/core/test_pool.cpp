#include "handle.h"
#include "log.h"
#include "log_capture.h"
#include "pool.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>

using bfr::core::Handle;
using bfr::core::Pool;

TEST_CASE("Pool finds what it stores", "[core][unit]") {
    Pool<int> pool{"test", 1};

    const Handle<int> handle = pool.emplace(4);

    CHECK_FALSE(handle.isNull());
    REQUIRE(pool.find(handle) != nullptr);
    CHECK(*pool.find(handle) == 4);
    CHECK(pool.size() == 1);
}

TEST_CASE("Pool finds nothing for an erased handle", "[core][unit]") {
    Pool<int> pool{"test", 1};
    const Handle<int> handle = pool.emplace(4);

    CHECK(pool.erase(handle));

    CHECK(pool.find(handle) == nullptr);
    CHECK_FALSE(pool.erase(handle));
    CHECK(pool.size() == 0);
}

TEST_CASE("Pool gives a reused slot a new generation", "[core][unit]") {
    Pool<int> pool{"test", 1};
    const Handle<int> first = pool.emplace(1);
    REQUIRE(pool.erase(first));

    const Handle<int> second = pool.emplace(2);

    CHECK(second.index == first.index);
    CHECK(second.generation != first.generation);
    CHECK(pool.find(first) == nullptr);
    REQUIRE(pool.find(second) != nullptr);
    CHECK(*pool.find(second) == 2);
}

TEST_CASE("Pool finds nothing for a null handle", "[core][unit]") {
    Pool<int> pool{"test", 1};

    CHECK(Handle<int>{}.isNull());
    CHECK(pool.find(Handle<int>{}) == nullptr);
    pool.emplace(1);
    CHECK(pool.find(Handle<int>{}) == nullptr);
    CHECK_FALSE(pool.erase(Handle<int>{}));
}

TEST_CASE("Pool preallocates its capacity", "[core][unit]") {
    if (!bfr::core::kLoggingEnabled) {
        SKIP("logging is compiled out");
    }
    std::ostringstream log;
    captureLogInto(log);
    constexpr std::uint32_t kCapacity = 100;
    Pool<int> pool{"test", kCapacity};
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
    captureLogInto(log);
    Pool<int> pool{"surfaces", 1};
    const std::uint32_t before = pool.capacity();

    for (std::uint32_t added = 0; added <= before; ++added) {
        pool.emplace(0);
    }

    CHECK(pool.capacity() == before + Pool<int>::kPageSize);
    CHECK_THAT(log.str(), Catch::Matchers::ContainsSubstring("Pool surfaces grew to " +
                                                             std::to_string(before + Pool<int>::kPageSize)));
}

TEST_CASE("Pool keeps entries in place as it grows", "[core][unit]") {
    Pool<int> pool{"test", 1};
    const Handle<int> first = pool.emplace(1);
    const int* address = pool.find(first);

    for (std::uint32_t added = 0; added < 2 * Pool<int>::kPageSize; ++added) {
        pool.emplace(0);
    }

    CHECK(pool.find(first) == address);
}

TEST_CASE("Pool clear destroys every value", "[core][unit]") {
    const auto shared = std::make_shared<int>(0);
    Pool<std::shared_ptr<int>> pool{"test", 1};
    pool.emplace(shared);
    pool.emplace(shared);

    pool.clear();

    CHECK(shared.use_count() == 1);
    CHECK(pool.size() == 0);
}

TEST_CASE("Pool destroys its values when it goes", "[core][unit]") {
    const auto shared = std::make_shared<int>(0);
    {
        Pool<std::shared_ptr<int>> pool{"test", 1};
        pool.emplace(shared);
    }

    CHECK(shared.use_count() == 1);
}
