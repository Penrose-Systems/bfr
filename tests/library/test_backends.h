#pragma once

// Which backends the agnostic test cases run on: every backend the build has,
// or only those the test program's --backend options name. ctest's default
// entries pass --backend null; its gpu entries name a GPU backend.

#include <string_view>

namespace bfr::test {

/// Whether the agnostic cases run on the backend named, as name(Backend)
/// spells it.
[[nodiscard]] bool runsOnBackend(std::string_view backend);

} // namespace bfr::test
