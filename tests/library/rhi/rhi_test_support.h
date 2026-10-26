#pragma once

// Helpers for the RHI tests.

#include "error.h"
#include "rhi.h"
#include "rhi_config.h"
#include "scoped_env.h"
#include "window.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace bfr::rhi::test {

inline const RhiConfig kValidation{.validation = true};

/// For a device that presents to no window yet.
inline const core::window::NativeWindow kNoWindow{};

/// The images a test swapchain asks for.
inline constexpr std::uint32_t kSwapchainImages = 3;

/// A description that records every validation message the Rhi reports.
inline RhiDesc recordingInto(std::vector<std::string>& messages) {
    return {.onValidationMessage = [&messages](bool, std::string_view message) {
        messages.emplace_back(message);
    }};
}

/// A description that records only the errors validation reports: the loader
/// warns about layers turned on from the environment.
inline RhiDesc recordingErrorsInto(std::vector<std::string>& errors) {
    return {.onValidationMessage = [&errors](bool isError, std::string_view message) {
        if (isError) {
            errors.emplace_back(message);
        }
    }};
}

/// The code and message of a failed result, for INFO; empty on success.
template <class T>
std::string failureOf(const core::Result<T>& result) {
    return result.has_value() ? std::string{}
                              : std::string{core::name(result.error().code)} + ": " + result.error().message;
}

/// The CPU adapter, lavapipe, which every test machine has and whose driver
/// presents to headless windows. Tests of what the RHI does with a driver's
/// answers about a headless window use it: NVIDIA's driver is never asked.
inline std::size_t headlessAdapter(const Rhi& rhi) {
    const auto adapters = rhi.adapters();
    const auto found = std::ranges::find(adapters, AdapterType::Cpu, &Adapter::type);
    REQUIRE(found != adapters.end());
    return static_cast<std::size_t>(found - adapters.begin());
}

/// A device on the first adapter that presents to window. An adapter may not
/// present to it at all: NVIDIA's driver has no headless surfaces.
inline std::unique_ptr<Device> presentingDevice(Rhi& rhi, const core::window::NativeWindow& window) {
    for (std::size_t adapter = 0; adapter < rhi.adapters().size(); ++adapter) {
        auto device = rhi.createDevice(adapter, window);
        if (device.has_value()) {
            return std::move(*device);
        }
        INFO(rhi.adapters()[adapter].name << ' ' << failureOf(device));
        REQUIRE(device.error().code == core::ErrorCode::RhiPresentUnsupported);
    }
    FAIL("no adapter presents to the window");
    return nullptr;
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

} // namespace bfr::rhi::test
