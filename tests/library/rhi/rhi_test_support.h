#pragma once

// Helpers for the RHI tests: those that speak only to the interface and run on
// every backend the build has. A backend's own helpers sit under
// backends/<name>.

#include "bfr/core/error.h"
#include "bfr/core/window.h"
#include "bfr/rhi/rhi.h"
#include "bfr/rhi/rhi_config.h"
#include "bfr/rhi/swapchain.h"
#include "test_backends.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace bfr::rhi::test {

inline const RhiConfig kValidation{.validation = true};

/// For a device that presents to no window yet.
inline const core::window::NativeWindow kNoWindow{};

/// What a test swapchain asks for: the defaults.
inline constexpr SwapchainRequest kSwapchainRequest{};

/// Each backend the agnostic cases run on (test_backends.h), for a test case
/// that runs once on each: `const Backend backend = GENERATE(eachBackend());`.
inline auto eachBackend() {
    std::vector<Backend> chosen;
    std::ranges::copy_if(backends(), std::back_inserter(chosen),
                         [](Backend backend) { return bfr::test::runsOnBackend(name(backend)); });
    INFO("no backend --backend named is in this build");
    REQUIRE_FALSE(chosen.empty());
    return Catch::Generators::from_range(chosen);
}

/// A description on backend that records every warning and error the Rhi and
/// validation report.
inline RhiDesc recordingInto(Backend backend, std::vector<std::string>& messages) {
    return {.backend = backend,
            .logger = {[&messages](core::Severity, std::string_view message) { messages.emplace_back(message); },
                       core::Severity::Warning}};
}

/// A description on backend that records only errors: a backend's loader may
/// warn about what the environment turned on.
inline RhiDesc recordingErrorsInto(Backend backend, std::vector<std::string>& errors) {
    return {.backend = backend,
            .logger = {[&errors](core::Severity, std::string_view message) { errors.emplace_back(message); },
                       core::Severity::Error}};
}

/// The code and message of a failed result, for INFO; empty on success.
template <class T>
std::string failureOf(const core::Result<T>& result) {
    return result.has_value() ? std::string{}
                              : std::string{core::name(result.error().code)} + ": " + result.error().message;
}

/// The first adapter that presents to a headless window. An adapter may not
/// present to one at all: NVIDIA's driver has no headless surfaces.
inline std::size_t headlessAdapter(const Rhi& rhi) {
    for (std::size_t adapter = 0; adapter < rhi.adapters().size(); ++adapter) {
        const auto caps = rhi.surfaceCaps(adapter, core::window::Headless{});
        if (caps.has_value()) {
            return adapter;
        }
        INFO(rhi.adapters()[adapter].name << ' ' << failureOf(caps));
        REQUIRE(caps.error().code == core::ErrorCode::RhiPresentUnsupported);
    }
    FAIL("no adapter presents to a headless window");
    return 0;
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

} // namespace bfr::rhi::test
