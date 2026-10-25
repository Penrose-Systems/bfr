// Indexes one past the end of a vector; the standard library's assertions must
// stop it. A failed assertion aborts, which ctest counts as a failure whatever
// the output, so the abort becomes an ordinary exit.
#include <csignal>
#include <cstddef>
#include <cstdlib>
#include <vector>

int main(int argc, char**) {
    std::signal(SIGABRT, [](int) { std::_Exit(1); });
    const std::vector<int> values(1);
    return values[static_cast<std::size_t>(argc)];
}
