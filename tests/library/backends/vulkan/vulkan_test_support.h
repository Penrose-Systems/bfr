#pragma once

// Helpers for the Vulkan backend's tests: the layers they load and the
// environment they set, on top of the RHI tests' helpers.

#include "bfr/rhi/rhi.h"
#include "rhi/rhi_test_support.h"
#include "scoped_env.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

namespace bfr::rhi::vulkan::test {

using namespace bfr::rhi::test; // NOLINT(google-build-using-namespace)

/// A description on the Vulkan backend with no logger.
inline const RhiDesc kVulkan{.backend = Backend::Vulkan};

/// recordingInto on the Vulkan backend.
inline RhiDesc recordingInto(std::vector<std::string>& messages) {
    return bfr::rhi::test::recordingInto(Backend::Vulkan, messages);
}

/// recordingErrorsInto on the Vulkan backend.
inline RhiDesc recordingErrorsInto(std::vector<std::string>& errors) {
    return bfr::rhi::test::recordingErrorsInto(Backend::Vulkan, errors);
}

/// The CPU adapter, lavapipe, which every Vulkan test machine has. Tests of
/// what the backend does with a driver's answers use it: the adapter override
/// layer changes what it reports, and it has one queue family.
inline std::size_t cpuAdapter(const Rhi& rhi) {
    const auto adapters = rhi.adapters();
    const auto found = std::ranges::find(adapters, AdapterType::Cpu, &Adapter::type);
    REQUIRE(found != adapters.end());
    return static_cast<std::size_t>(found - adapters.begin());
}

/// For one scope, the loader reports itself as version ("1.4.319"), through
/// the loader version layer.
struct LoaderVersion {
    explicit LoaderVersion(const std::string& version) : reported{"BFR_LOADER_VERSION", version} {}

    ScopedEnv layer{"XDG_DATA_HOME", BFR_TEST_LOADER_VERSION_DATA};
    ScopedEnv reported;
};

/// Turns the adapter override layer on for one scope, after vcpkg's layers.
/// Layers named in VK_INSTANCE_LAYERS sit above those the Rhi enables, the
/// first nearest the application, so a layer named in above sees what this one
/// changes.
struct AdapterOverrideLayer {
    explicit AdapterOverrideLayer(const std::string& above = {})
        : enable{"VK_INSTANCE_LAYERS", (above.empty() ? "" : above + ":") + "VK_LAYER_BFR_adapter_override"} {}

    ScopedEnv path{"VK_ADD_LAYER_PATH",
                   std::string{std::getenv("VK_ADD_LAYER_PATH")} + ":" + BFR_TEST_ADAPTER_LAYER_PATH};
    ScopedEnv enable;
};

} // namespace bfr::rhi::vulkan::test
