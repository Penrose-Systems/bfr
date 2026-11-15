#pragma once

// Helpers for the library tests that speak only to the public API, bfr.hpp,
// and run on every backend the build has. A backend's own helpers sit under
// backends/<name>.

#include "bfr.hpp"
#include "test_backends.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace bfr::test {

// Lists each test command pool holds, more than any test records between resets.
inline constexpr std::uint32_t kPoolLists = 8;

/// Validation on, for createInstance.
inline constexpr bool kValidation = true;

// What the tests' devices hold, more than any test uses; a test of one of
// these maximums sets its own with testDeviceWith.
inline constexpr std::uint32_t kTestLists = 64;
inline constexpr std::uint32_t kTestPools = 16;
inline constexpr std::uint32_t kTestSwapchains = 4;
inline constexpr std::uint32_t kTestTableSlots = 256;
inline constexpr std::uint32_t kTestSamplers = 16;
inline constexpr std::uint32_t kTestTimestampPools = 4;
inline constexpr std::uint32_t kTestQueryPools = 4;
inline constexpr std::uint32_t kTestObjects = 1024;
// Queues at rank 0 and level normal, as many as a test opens.
inline constexpr std::array<QueueDesc, 4> kTestQueues{};
// The longest wait there is, so a slow GPU never fails a test.
inline constexpr std::uint64_t kTestWait = std::numeric_limits<std::uint64_t>::max();

/// For a device that presents to no window yet: all zero opens it without asking.
inline constexpr Window kNoWindow{};
inline constexpr Window kHeadless{.system = WindowSystem::Headless};

/// What the tests' devices hold; testDeviceOn and testDeviceWith name the
/// adapter and window.
inline constexpr DeviceDesc kTestDevice{.graphicsQueues = std::span{kTestQueues}.first(1),
                                        .maxCommandLists = kTestLists,
                                        .maxCommandPools = kTestPools,
                                        .maxSwapchains = kTestSwapchains,
                                        .maxBindlessTextures = kTestTableSlots,
                                        .maxBindlessStorageTextures = kTestTableSlots,
                                        .maxBindlessStorageBuffers = kTestTableSlots,
                                        .maxBindlessSamplers = kTestSamplers,
                                        .maxTimestampPools = kTestTimestampPools,
                                        .maxQueryPools = kTestQueryPools,
                                        .maxBuffers = kTestObjects,
                                        .maxTextures = kTestObjects,
                                        .maxShaders = kTestObjects,
                                        .maxPipelines = kTestObjects,
                                        .maxPipelineCaches = kTestObjects,
                                        .maxAccelerationStructures = kTestObjects};

/// kTestDevice on adapter, presenting to window.
inline DeviceDesc testDeviceOn(std::uint32_t adapter, const Window& window) {
    DeviceDesc desc = kTestDevice;
    desc.adapter = adapter;
    desc.window = window;
    return desc;
}

/// testDeviceOn with change made to it.
template <class Change>
DeviceDesc testDeviceWith(std::uint32_t adapter, const Window& window, Change change) {
    DeviceDesc desc = testDeviceOn(adapter, window);
    change(desc);
    return desc;
}

/// A backend the agnostic cases run on: one the build has.
struct TestBackend {
    std::string_view name;
    Backend backend{};
};

[[nodiscard]] inline std::string_view name(const TestBackend& backend) {
    return backend.name;
}

/// Each backend the agnostic cases run on (test_backends.h), for a test case
/// that runs once on each: `const TestBackend backend = GENERATE(eachBackend());`.
inline auto eachBackend() {
    const auto backends = bfr::backends();
    REQUIRE(backends.has_value());
    std::vector<TestBackend> chosen;
    for (const Backend backend : *backends) {
        if (runsOnBackend(bfr::name(backend))) {
            chosen.push_back({.name = bfr::name(backend), .backend = backend});
        }
    }
    INFO("no backend --backend named is in this build");
    REQUIRE_FALSE(chosen.empty());
    return Catch::Generators::from_range(chosen);
}

/// A description on backend at the version the tests ask it for
/// (testApiVersion), with nothing recorded.
inline InstanceDesc describe(Backend backend) {
    const ApiVersion version = testApiVersion(bfr::name(backend));
    return {.backend = backend, .apiMajor = version.major, .apiMinor = version.minor};
}

/// describe on a TestBackend.
inline InstanceDesc describe(const TestBackend& backend) {
    return describe(backend.backend);
}

/// A MessageCallback that appends each message to the std::vector<std::string>
/// userData points at.
inline void recordInto(MessageSeverity /*severity*/, std::string_view message, void* userData) {
    static_cast<std::vector<std::string>*>(userData)->emplace_back(message);
}

/// A description on backend that records every warning and error the Instance and
/// validation report.
inline InstanceDesc recordingInto(Backend backend, std::vector<std::string>& messages) {
    InstanceDesc desc = describe(backend);
    desc.messageCallback = recordInto;
    desc.messageUserData = &messages;
    desc.minSeverity = MessageSeverity::Warning;
    return desc;
}

/// recordingInto on a TestBackend.
inline InstanceDesc recordingInto(const TestBackend& backend, std::vector<std::string>& messages) {
    return recordingInto(backend.backend, messages);
}

/// A description on backend that records only errors: a backend's loader may
/// warn about what the environment turned on.
inline InstanceDesc recordingErrorsInto(Backend backend, std::vector<std::string>& errors) {
    InstanceDesc desc = recordingInto(backend, errors);
    desc.minSeverity = MessageSeverity::Error;
    return desc;
}

/// An Instance as desc says, with validation on or off.
[[nodiscard]] inline Expected<Instance> createInstance(InstanceDesc desc, bool validation) {
    desc.validation = validation;
    return Instance::create(desc);
}

/// The code and message of a failed result, for INFO; empty on success.
template <class T>
std::string failureOf(const Expected<T>& result) {
    return result.has_value() ? std::string{}
                              : std::string{bfr::name(result.error().code)} + ": " + result.error().message.text();
}

/// The adapters instance lists; the test fails when it can't list them.
inline std::vector<AdapterInfo> adaptersOf(const Instance& instance) {
    auto adapters = instance.adapters();
    INFO(failureOf(adapters));
    REQUIRE(adapters.has_value());
    return std::move(*adapters);
}

/// The first adapter that presents to window. An adapter may not present to
/// one at all: NVIDIA's driver has no headless surfaces.
inline std::uint32_t presentingAdapter(const Instance& instance, const Window& window) {
    const std::vector<AdapterInfo> adapters = adaptersOf(instance);
    for (std::uint32_t adapter = 0; adapter < adapters.size(); ++adapter) {
        const auto caps = instance.swapchainCaps(adapter, {.window = window});
        if (caps.has_value()) {
            return adapter;
        }
        INFO(adapters[adapter].name << ' ' << failureOf(caps));
        REQUIRE(caps.error().code == Result::ErrorPresentUnsupported);
    }
    FAIL("no adapter presents to the window");
    return 0;
}

/// The first adapter that presents to a headless window.
inline std::uint32_t headlessAdapter(const Instance& instance) {
    return presentingAdapter(instance, kHeadless);
}

/// A device on the first adapter that presents to window.
inline Device presentingDevice(const Instance& instance, const Window& window) {
    auto device = instance.createDevice(testDeviceOn(presentingAdapter(instance, window), window));
    INFO(failureOf(device));
    REQUIRE(device.has_value());
    return std::move(*device);
}

} // namespace bfr::test
