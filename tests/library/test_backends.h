#pragma once

// Which backends the agnostic test cases run on: every backend the build has,
// or only those the test program's --backend options name. ctest's entries
// name one: vulkan on lavapipe by default, each GPU backend under gpu. Vulkan's
// entries run again with --api-version 1.2, BFR's minimum.

#include <cstdint>
#include <string_view>

namespace bfr::test {

/// An API version in AdapterInfo's terms: 1, 2 is Vulkan 1.2.
struct ApiVersion {
    std::uint32_t major = 0;
    std::uint32_t minor = 0;
};

/// Whether the agnostic cases run on the backend named, as name(Backend)
/// spells it.
[[nodiscard]] bool runsOnBackend(std::string_view backend);

/// The version the cases ask the backend named for: the test program's
/// --api-version ("1.2") where it names one, else the newest that backend's
/// tests are written for.
[[nodiscard]] ApiVersion testApiVersion(std::string_view backend);

} // namespace bfr::test
