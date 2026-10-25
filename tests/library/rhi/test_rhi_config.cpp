#include "config.h"
#include "config_file.h"
#include "rhi_config.h"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>

TEST_CASE("RhiConfig takes each value its section sets", "[rhi][unit]") {
    const auto path = writeConfig("bfr_rhi_config.json",
                                  R"({"rhi": {"version": 1, "validation": true, "swapchainImages": 4}})");
    const auto config = bfr::config::Config::load(path);
    std::filesystem::remove(path);
    REQUIRE(config.has_value());
    bfr::rhi::RhiConfig rhi;

    rhi.set(*config);

    CHECK(rhi.validation);
    CHECK(rhi.swapchainImages == 4);
}
