#include "log.h"
#include "log_capture.h"

#include <catch2/catch_test_macros.hpp>

#include <sstream>

TEST_CASE("Log writes each message to the logger's sinks", "[core][unit]") {
    if (!bfr::core::kLoggingEnabled) {
        SKIP("logging is compiled out");
    }
    bfr::core::Log::init(bfr::core::LogConfig(BFR_LOGLEVEL_TRACE));
    std::ostringstream output;
    captureLogInto(output);

    BFR_LOG_INFO("hello {}", 7);

    REQUIRE(output.str() == "hello 7\n");
}

TEST_CASE("Log compiled out never evaluates its arguments", "[core][unit]") {
    if (bfr::core::kLoggingEnabled) {
        SKIP("logging is compiled in");
    }
    bool evaluated = false;
    const auto evaluate = [&evaluated] {
        evaluated = true;
        return 0;
    };

    BFR_LOG_INFO("{}", evaluate());

    REQUIRE_FALSE(evaluated);
    // The probe does set the flag, so the check above can fail.
    evaluate();
    REQUIRE(evaluated);
}

TEST_CASE("Log at level off drops every message", "[core][unit]") {
    bfr::core::Log::init(bfr::core::LogConfig(BFR_LOGLEVEL_OFF));
    std::ostringstream output;
    captureLogInto(output);

    BFR_LOG_CRITICAL("dropped");

    REQUIRE(output.str().empty());
}
