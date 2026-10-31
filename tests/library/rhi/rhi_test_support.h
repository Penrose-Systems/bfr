#pragma once

// Helpers for the RHI tests: those that speak only to the interface and run on
// every backend the build has. A backend's own helpers sit under
// backends/<name>.

#include "bfr/core/error.h"
#include "bfr/core/window.h"
#include "bfr/rhi/instance.h"
#include "bfr/rhi/rhi_config.h"
#include "bfr/rhi/swapchain.h"
#include "null/null_instance.h"
#include "test_backends.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace bfr::rhi::test {

inline const RhiConfig kValidation{.validation = true};

/// For a device that presents to no window yet.
inline const core::window::NativeWindow kNoWindow{};

/// A backend the agnostic cases run on: one the build has, or the null one,
/// which only these tests reach and which has no Backend.
struct TestBackend {
    std::string_view name;
    std::optional<Backend> backend;
};

[[nodiscard]] inline std::string_view name(const TestBackend& backend) {
    return backend.name;
}

/// What createInstance needs to make an Instance on a TestBackend.
struct TestInstanceDesc {
    InstanceDesc desc;
    bool null = false;
};

/// Each backend the agnostic cases run on (test_backends.h), for a test case
/// that runs once on each: `const TestBackend backend = GENERATE(eachBackend());`.
inline auto eachBackend() {
    std::vector<TestBackend> chosen;
    for (const Backend backend : backends()) {
        chosen.push_back({.name = name(backend), .backend = backend});
    }
    chosen.push_back({.name = "null", .backend = std::nullopt});
    std::erase_if(chosen, [](const TestBackend& backend) { return !bfr::test::runsOnBackend(backend.name); });
    INFO("no backend --backend named is in this build");
    REQUIRE_FALSE(chosen.empty());
    return Catch::Generators::from_range(chosen);
}

/// An Instance on backend: the null one when it has no Backend.
[[nodiscard]] inline core::Result<std::unique_ptr<Instance>> createInstance(const TestInstanceDesc& test,
                                                                            const RhiConfig& config) {
    if (test.null) {
        return null::NullInstance::create(test.desc, config);
    }
    return Instance::create(test.desc, config);
}

/// A description on backend, with nothing recorded.
inline TestInstanceDesc describe(const TestBackend& backend) {
    return {.desc = {.backend = backend.backend.value_or(Backend{})}, .null = !backend.backend.has_value()};
}

/// A description on backend that records every warning and error the Instance and
/// validation report.
inline InstanceDesc recordingInto(Backend backend, std::vector<std::string>& messages) {
    return {.backend = backend,
            .logger = {[&messages](core::Severity, std::string_view message) { messages.emplace_back(message); },
                       core::Severity::Warning}};
}

/// recordingInto on a TestBackend.
inline TestInstanceDesc recordingInto(const TestBackend& backend, std::vector<std::string>& messages) {
    return {.desc = recordingInto(backend.backend.value_or(Backend{}), messages), .null = !backend.backend.has_value()};
}

/// A description on backend that records only errors: a backend's loader may
/// warn about what the environment turned on.
inline InstanceDesc recordingErrorsInto(Backend backend, std::vector<std::string>& errors) {
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
inline std::size_t headlessAdapter(const Instance& rhi) {
    for (std::size_t adapter = 0; adapter < rhi.adapters().size(); ++adapter) {
        const auto caps = rhi.swapchainCaps(adapter, core::window::Headless{});
        if (caps.has_value()) {
            return adapter;
        }
        INFO(rhi.adapters()[adapter].name << ' ' << failureOf(caps));
        REQUIRE(caps.error().code == core::ErrorCode::RhiPresentUnsupported);
    }
    FAIL("no adapter presents to a headless window");
    return 0;
}

/// What a test swapchain on a headless window of rhi asks for: the defaults,
/// with the fewest images the surface reports it takes, as a caller must set.
inline SwapchainDesc swapchainDesc(const Instance& rhi) {
    const auto caps = rhi.swapchainCaps(headlessAdapter(rhi), core::window::Headless{});
    INFO(failureOf(caps));
    REQUIRE(caps.has_value());
    return {.imageCount = caps->minImageCount};
}

/// A device on the first adapter that presents to window. An adapter may not
/// present to it at all: NVIDIA's driver has no headless surfaces.
inline std::unique_ptr<Device> presentingDevice(Instance& rhi, const core::window::NativeWindow& window) {
    for (std::size_t adapter = 0; adapter < rhi.adapters().size(); ++adapter) {
        auto device = rhi.createDevice(adapter, window, {});
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
