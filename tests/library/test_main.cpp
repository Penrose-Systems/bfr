#include "test_backends.h"

#include <catch2/catch_session.hpp>

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

// The backends --backend named; empty for every backend the build has.
std::vector<std::string>& chosenBackends() {
    static std::vector<std::string> chosen;
    return chosen;
}

// What --api-version named; empty for each backend's own.
std::string& chosenVersion() {
    static std::string chosen;
    return chosen;
}

// "1.2" as a version; nullopt for anything else.
std::optional<bfr::test::ApiVersion> parseVersion(std::string_view text) {
    bfr::test::ApiVersion version;
    const std::size_t dot = text.find('.');
    if (dot == std::string_view::npos) {
        return std::nullopt;
    }
    const char* end = text.data() + text.size();
    const auto major = std::from_chars(text.data(), text.data() + dot, version.major);
    const auto minor = std::from_chars(text.data() + dot + 1, end, version.minor);
    if (major.ec != std::errc{} || major.ptr != text.data() + dot || minor.ec != std::errc{} || minor.ptr != end) {
        return std::nullopt;
    }
    return version;
}

// The newest version each backend's tests are written for. D3D12 and Metal
// take none yet.
constexpr bfr::test::ApiVersion kVulkanTests{.major = 1, .minor = 4};

} // namespace

namespace bfr::test {

bool runsOnBackend(std::string_view backend) {
    const auto& chosen = chosenBackends();
    return chosen.empty() || std::ranges::find(chosen, backend) != chosen.end();
}

ApiVersion testApiVersion(std::string_view backend) {
    const std::string& chosen = chosenVersion();
    if (chosen.empty()) {
        return backend == "vulkan" ? kVulkanTests : ApiVersion{};
    }
    return *parseVersion(chosen);
}

} // namespace bfr::test

// The main every test program runs. A backend that needs its environment set
// up does so from its own tests/library/backends/<name>/test_environment.cpp,
// linked beside this.
int main(int argc, char* argv[]) {
    Catch::Session session;
    session.cli(session.cli() |
                Catch::Clara::Opt(chosenBackends(),
                                  "name")["--backend"]("run the agnostic cases on this backend; repeat for several. "
                                                       "Without it, they run on every backend the build has.") |
                Catch::Clara::Opt(chosenVersion(), "major.minor")["--api-version"](
                    "the API version every case asks its backend for. Without it, each backend's newest."));
    const int parsed = session.applyCommandLine(argc, argv);
    if (parsed != 0) {
        return parsed;
    }
    if (!chosenVersion().empty() && !parseVersion(chosenVersion())) {
        std::cerr << "--api-version takes major.minor, such as 1.2, not " << chosenVersion() << "\n";
        return 1;
    }
    return session.run();
}
