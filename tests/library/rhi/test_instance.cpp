// What every backend's Instance does, through the interface alone. Each case runs
// on every backend the build has; nothing here names one.

#include "bfr/core/error.h"
#include "bfr/core/window.h"
#include "bfr/rhi/instance.h"
#include "rhi_test_support.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_container_properties.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cstddef>
#include <format>
#include <string>
#include <vector>

using namespace bfr::rhi;
using namespace bfr::rhi::test;

TEST_CASE("Instance has a name for each backend", "[rhi][unit]") {
    for (const Backend backend : backends()) {
        CHECK(name(backend) != "unknown");
    }
}

TEST_CASE("Instance fails with RhiBackendUnavailable for a backend this build lacks", "[rhi][unit]") {
    for (const Backend backend : {Backend::Vulkan, Backend::D3D12, Backend::Metal}) {
        if (std::ranges::contains(backends(), backend)) {
            continue;
        }
        const auto rhi = Instance::create({.backend = backend}, {});

        INFO(name(backend));
        REQUIRE_FALSE(rhi.has_value());
        CHECK(rhi.error().code == bfr::core::ErrorCode::RhiBackendUnavailable);
        CHECK_THAT(rhi.error().message,
                   Catch::Matchers::Equals(std::format("backend {} is not in this build", name(backend))));
    }
}

TEST_CASE("Instance fails with RhiBackendUnavailable for a value that is no backend", "[rhi][unit]") {
    const auto rhi = Instance::create({.backend = static_cast<Backend>(0)}, {});

    REQUIRE_FALSE(rhi.has_value());
    CHECK(rhi.error().code == bfr::core::ErrorCode::RhiBackendUnavailable);
    CHECK_THAT(rhi.error().message, Catch::Matchers::Equals("backend 0 is not one BFR has"));
}

TEST_CASE("Instance with validation lists an adapter and reports nothing", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;

    const auto rhi = createInstance(recordingInto(backend, messages), kValidation);

    INFO(name(backend) << ' ' << failureOf(rhi));
    REQUIRE(rhi.has_value());
    CHECK_FALSE((*rhi)->adapters().empty());
    for (const Adapter& adapter : (*rhi)->adapters()) {
        CHECK_FALSE(adapter.name.empty());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Instance fails to create a device on an adapter it does not list", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    const auto rhi = createInstance(recordingInto(backend, messages), kValidation);
    INFO(name(backend) << ' ' << failureOf(rhi));
    REQUIRE(rhi.has_value());

    const auto device = (*rhi)->createDevice((*rhi)->adapters().size(), kNoWindow, {});

    REQUIRE_FALSE(device.has_value());
    CHECK(device.error().code == bfr::core::ErrorCode::RhiAdapterOutOfRange);
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Instance creates a device on every adapter it lists", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    const auto rhi = createInstance(recordingInto(backend, messages), kValidation);
    INFO(name(backend) << ' ' << failureOf(rhi));
    REQUIRE(rhi.has_value());
    REQUIRE_FALSE((*rhi)->adapters().empty());

    for (std::size_t adapter = 0; adapter < (*rhi)->adapters().size(); ++adapter) {
        const auto device = (*rhi)->createDevice(adapter, kNoWindow, {});
        INFO((*rhi)->adapters()[adapter].name << ' ' << failureOf(device));
        REQUIRE(device.has_value());
        CHECK(*device != nullptr);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Instance creates a device that presents to a headless window", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const auto rhi = createInstance(recordingInto(backend, messages), kValidation);
        INFO(name(backend) << ' ' << failureOf(rhi));
        REQUIRE(rhi.has_value());

        const auto device = presentingDevice(**rhi, bfr::core::window::Headless{});

        REQUIRE(device != nullptr);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}
