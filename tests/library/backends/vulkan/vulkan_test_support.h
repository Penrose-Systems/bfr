#pragma once

// Helpers for the Vulkan backend's tests: the layers they load and the
// environment they set, on top of the public API tests' helpers.

#include "bfr.hpp"
#include "rhi/bfr_test_support.h"
#include "scoped_env.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace bfr::test::vulkan {

/// A description on the Vulkan backend at the version the tests ask for, with
/// no message callback.
inline InstanceDesc vulkanDesc() {
    return describe(Backend::Vulkan);
}

/// vulkanDesc at major.minor.
inline InstanceDesc vulkanAt(std::uint32_t major, std::uint32_t minor) {
    InstanceDesc desc = vulkanDesc();
    desc.apiMajor = major;
    desc.apiMinor = minor;
    return desc;
}

/// recordingInto on the Vulkan backend.
inline InstanceDesc recordingInto(std::vector<std::string>& messages) {
    return bfr::test::recordingInto(Backend::Vulkan, messages);
}

/// recordingErrorsInto on the Vulkan backend.
inline InstanceDesc recordingErrorsInto(std::vector<std::string>& errors) {
    return bfr::test::recordingErrorsInto(Backend::Vulkan, errors);
}

/// The CPU adapter, lavapipe, which every Vulkan test machine has. Tests of
/// what the backend does with a driver's answers use it: the adapter override
/// layer changes what it reports, and it has one queue family.
inline std::uint32_t cpuAdapter(const Instance& instance) {
    const std::vector<AdapterInfo> adapters = adaptersOf(instance);
    const auto found = std::ranges::find(adapters, AdapterType::Cpu, &AdapterInfo::type);
    REQUIRE(found != adapters.end());
    return static_cast<std::uint32_t>(found - adapters.begin());
}

/// For one scope, the loader reports itself as version ("1.4.319"), through
/// the loader version layer.
struct LoaderVersion {
    explicit LoaderVersion(const std::string& version) : reported{"BFR_LOADER_VERSION", version} {}

    ScopedEnv layer{"XDG_DATA_HOME", BFR_TEST_LOADER_VERSION_DATA};
    ScopedEnv reported;
};

/// Turns the adapter override layer on for one scope, after vcpkg's layers.
/// Layers named in VK_INSTANCE_LAYERS sit above those the Instance enables, the
/// first nearest the application, so a layer named in above sees what this one
/// changes.
struct AdapterOverrideLayer {
    explicit AdapterOverrideLayer(const std::string& above = {})
        : enable{"VK_INSTANCE_LAYERS", (above.empty() ? "" : above + ":") + "VK_LAYER_BFR_adapter_override"} {}

    ScopedEnv path{"VK_LAYER_PATH", std::string{std::getenv("VK_LAYER_PATH")} + ":" + BFR_TEST_ADAPTER_LAYER_PATH};
    ScopedEnv enable;
};

} // namespace bfr::test::vulkan
