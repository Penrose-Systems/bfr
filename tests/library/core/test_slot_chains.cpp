#include "core/slot_chains.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <set>
#include <vector>

using bfr::core::SlotChains;

namespace {

constexpr std::uint32_t kSlots = 8;

std::vector<std::uint32_t> walk(const SlotChains& slots, std::uint32_t first) {
    std::vector<std::uint32_t> chain;
    for (std::uint32_t slot = first; slot != SlotChains::kEnd; slot = slots.next(slot)) {
        chain.push_back(slot);
    }
    return chain;
}

} // namespace

TEST_CASE("SlotChains hands out distinct slots in chains of the size asked for", "[core][unit]") {
    SlotChains slots{kSlots};
    const auto first = slots.take(3);
    const auto second = slots.take(5);
    REQUIRE(first.has_value());
    REQUIRE(second.has_value());
    const auto a = walk(slots, *first);
    const auto b = walk(slots, *second);
    CHECK(a.size() == 3);
    CHECK(b.size() == 5);
    std::set<std::uint32_t> all{a.begin(), a.end()};
    all.insert(b.begin(), b.end());
    CHECK(all.size() == kSlots);
    CHECK(slots.available() == 0);
}

TEST_CASE("SlotChains refuses more slots than are free, and takes none", "[core][unit]") {
    SlotChains slots{kSlots};
    REQUIRE(slots.take(kSlots - 1).has_value());
    CHECK_FALSE(slots.take(2).has_value());
    CHECK(slots.available() == 1);
    CHECK(slots.take(0) == SlotChains::kEnd);
}

TEST_CASE("SlotChains takes back a chain for the next take", "[core][unit]") {
    SlotChains slots{kSlots};
    const auto first = slots.take(kSlots);
    REQUIRE(first.has_value());
    slots.give(*first);
    CHECK(slots.available() == kSlots);
    const auto again = slots.take(kSlots);
    REQUIRE(again.has_value());
    CHECK(walk(slots, *again).size() == kSlots);
    slots.give(SlotChains::kEnd);
    CHECK(slots.available() == 0);
}

TEST_CASE("SlotChains pushes slots onto a chain one at a time", "[core][unit]") {
    SlotChains slots{2};
    std::uint32_t head = SlotChains::kEnd;
    const auto first = slots.push(head);
    const auto second = slots.push(head);
    REQUIRE(first.has_value());
    REQUIRE(second.has_value());
    CHECK(walk(slots, head) == std::vector{*second, *first});
    const std::uint32_t full = head;
    CHECK_FALSE(slots.push(head).has_value());
    CHECK(head == full);
    slots.give(head);
    CHECK(slots.available() == 2);
}
