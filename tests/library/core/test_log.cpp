#include "log.h"
#include "log_capture.h"

#include <catch2/catch_test_macros.hpp>

#include <sstream>

using bfr::core::Logger;
using bfr::core::Severity;

TEST_CASE("Log sends each message its logger wants", "[core][unit]") {
    if (!bfr::core::kLoggingEnabled) {
        SKIP("logging is compiled out");
    }
    std::ostringstream output;
    const Logger logger = loggingInto(output, Severity::Info);

    BFR_LOG_INFO(logger, "hello {}", 7);
    BFR_LOG_ERROR(logger, "and {}", "this");

    REQUIRE(output.str() == "hello 7\nand this\n");
}

TEST_CASE("Log never formats a message its logger doesn't want", "[core][unit]") {
    std::ostringstream output;
    const Logger logger = loggingInto(output, Severity::Warning);
    bool evaluated = false;
    const auto evaluate = [&evaluated] {
        evaluated = true;
        return 0;
    };

    BFR_LOG_DEBUG(logger, "{}", evaluate());

    CHECK(output.str().empty());
    REQUIRE_FALSE(evaluated);
    // The probe does set the flag, so the check above can fail.
    evaluate();
    REQUIRE(evaluated);
}

TEST_CASE("Log without a callback sends nothing", "[core][unit]") {
    const Logger logger;

    CHECK_FALSE(logger.wants(Severity::Error));
    CHECK_FALSE(Logger::none().wants(Severity::Error));
    BFR_LOG_ERROR(logger, "dropped");
}

TEST_CASE("Log compiled out never evaluates its arguments", "[core][unit]") {
    if (bfr::core::kLoggingEnabled) {
        SKIP("logging is compiled in");
    }
    std::ostringstream output;
    const Logger logger = loggingInto(output);
    bool evaluated = false;
    const auto evaluate = [&evaluated] {
        evaluated = true;
        return 0;
    };

    BFR_LOG_ERROR(logger, "{}", evaluate());

    CHECK(output.str().empty());
    REQUIRE_FALSE(evaluated);
    evaluate();
    REQUIRE(evaluated);
}
