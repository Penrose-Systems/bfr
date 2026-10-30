#include "test_backends.h"

#include <catch2/catch_session.hpp>

#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

namespace {

// The backends --backend named; empty for every backend the build has.
std::vector<std::string>& chosenBackends() {
    static std::vector<std::string> chosen;
    return chosen;
}

} // namespace

namespace bfr::test {

bool runsOnBackend(std::string_view backend) {
    const auto& chosen = chosenBackends();
    return chosen.empty() || std::ranges::find(chosen, backend) != chosen.end();
}

} // namespace bfr::test

// The main every test program runs. A backend that needs its environment set
// up does so from its own tests/library/backends/<name>/test_environment.cpp,
// linked beside this.
int main(int argc, char* argv[]) {
    Catch::Session session;
    session.cli(session.cli() | Catch::Clara::Opt(chosenBackends(), "name")["--backend"](
                                    "run the agnostic cases on this backend; repeat for several. "
                                    "Without it, they run on every backend the build has."));
    const int parsed = session.applyCommandLine(argc, argv);
    if (parsed != 0) {
        return parsed;
    }
    return session.run();
}
