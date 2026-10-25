#include "text.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <string>
#include <vector>

TEST_CASE("join puts the separator between each pair of items", "[core][unit]") {
    const std::vector<std::string> items{"a", "b", "c"};
    CHECK(bfr::core::text::join(items, ", ") == "a, b, c");
}

TEST_CASE("join of one item is the item alone", "[core][unit]") {
    const std::array items{"a"};
    CHECK(bfr::core::text::join(items, ", ") == "a");
}

TEST_CASE("join of no items is empty", "[core][unit]") {
    const std::vector<std::string> items;
    CHECK(bfr::core::text::join(items, ", ").empty());
}
