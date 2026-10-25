#include "config.h"
#include "config_file.h"
#include "error.h"
#include "log.h"
#include "log_capture.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdint>
#include <filesystem>
#include <sstream>

using bfr::config::Config;

TEST_CASE("Config reads a value from its file", "[config][unit]") {
    const auto path = writeConfig("bfr_config_reads.json", R"({"rhi": {"version": 1, "validation": true}})");

    const auto config = Config::load(path);

    std::filesystem::remove(path);
    REQUIRE(config.has_value());
    CHECK(config->section("rhi", 1).value("validation", false));
}

TEST_CASE("Config warns when a section's version differs and still reads it", "[config][unit]") {
    if (!bfr::core::kLoggingEnabled) {
        SKIP("logging is compiled out");
    }
    const auto path = writeConfig("bfr_config_drift.json", R"({"rhi": {"version": 2, "validation": true}})");
    const auto config = Config::load(path);
    std::filesystem::remove(path);
    REQUIRE(config.has_value());
    std::ostringstream output;
    captureLogInto(output);

    const auto rhi = config->section("rhi", 1);

    CHECK(output.str() == "Config section rhi is version 2; the code reads version 1\n");
    CHECK(rhi.value("validation", false));
}

TEST_CASE("Config warns when a section has no version and still reads it", "[config][unit]") {
    if (!bfr::core::kLoggingEnabled) {
        SKIP("logging is compiled out");
    }
    const auto path = writeConfig("bfr_config_unversioned.json", R"({"rhi": {"validation": true}})");
    const auto config = Config::load(path);
    std::filesystem::remove(path);
    REQUIRE(config.has_value());
    std::ostringstream output;
    captureLogInto(output);

    const auto rhi = config->section("rhi", 1);

    CHECK(output.str() == "Config section rhi has no version; the code reads version 1\n");
    CHECK(rhi.value("validation", false));
}

TEST_CASE("Config warns about a value of the wrong type and gives the default", "[config][unit]") {
    const auto path = writeConfig("bfr_config_wrong_type.json", R"({"rhi": {"version": 1, "validation": "yes"}})");
    const auto config = Config::load(path);
    std::filesystem::remove(path);
    REQUIRE(config.has_value());
    const auto rhi = config->section("rhi", 1);
    std::ostringstream output;
    captureLogInto(output);

    CHECK(rhi.value("validation", true));
    if (bfr::core::kLoggingEnabled) {
        CHECK_THAT(output.str(), Catch::Matchers::ContainsSubstring("rhi.validation") &&
                                     Catch::Matchers::ContainsSubstring("type must be boolean, but is string"));
    }
}

TEST_CASE("Config warns about a whole number that is negative, fractional or too large and gives the default",
          "[config][unit]") {
    const auto path = writeConfig("bfr_config_whole_number.json",
                                  R"({"rhi": {"version": 1, "negative": -1, "fraction": 2.5, "large": 4294967296}})");
    const auto config = Config::load(path);
    std::filesystem::remove(path);
    REQUIRE(config.has_value());
    const auto rhi = config->section("rhi", 1);
    std::ostringstream output;
    captureLogInto(output);
    constexpr std::uint32_t kDefault = 3;

    CHECK(rhi.value("negative", kDefault) == kDefault);
    CHECK(rhi.value("fraction", kDefault) == kDefault);
    CHECK(rhi.value("large", kDefault) == kDefault);
    if (bfr::core::kLoggingEnabled) {
        CHECK_THAT(output.str(), Catch::Matchers::ContainsSubstring("rhi.negative: -1 is not a whole number from 0 to "
                                                                    "4294967295") &&
                                     Catch::Matchers::ContainsSubstring("rhi.fraction: 2.5") &&
                                     Catch::Matchers::ContainsSubstring("rhi.large: 4294967296"));
    }
}

TEST_CASE("Config fails to load a file that isn't there", "[config][unit]") {
    const auto path = std::filesystem::temp_directory_path() / "bfr_config_missing.json";
    std::filesystem::remove(path);

    const auto config = Config::load(path);

    REQUIRE_FALSE(config.has_value());
    CHECK(config.error().code == bfr::core::ErrorCode::ConfigLoadFailed);
    CHECK_THAT(config.error().message, Catch::Matchers::ContainsSubstring(path.string()) &&
                                           Catch::Matchers::ContainsSubstring("No such file or directory"));
}

TEST_CASE("Config fails to load a file that isn't JSON and says where", "[config][unit]") {
    const auto path = writeConfig("bfr_config_broken.json", R"({"rhi": {"version": 1,}})");

    const auto config = Config::load(path);

    std::filesystem::remove(path);
    REQUIRE_FALSE(config.has_value());
    CHECK(config.error().code == bfr::core::ErrorCode::ConfigLoadFailed);
    CHECK_THAT(config.error().message, Catch::Matchers::ContainsSubstring(path.string()) &&
                                           Catch::Matchers::ContainsSubstring("line 1, column 23"));
}
