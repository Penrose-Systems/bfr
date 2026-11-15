// What every backend's Instance does, through the interface alone. Each case runs
// on every backend the build has; nothing here names one.

#include "bfr.hpp"
#include "bfr_test_support.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_container_properties.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cstdint>
#include <format>
#include <string>
#include <vector>

using namespace bfr;
using namespace bfr::test;

TEST_CASE("Instance has a name for each backend", "[rhi][unit]") {
    const auto built = backends();
    REQUIRE(built.has_value());
    for (const Backend backend : *built) {
        CHECK_FALSE(name(backend).empty());
    }
}

TEST_CASE("Instance fails with RhiBackendUnavailable for a backend this build lacks", "[rhi][unit]") {
    const auto built = backends();
    REQUIRE(built.has_value());
    for (const Backend backend : {Backend::Vulkan, Backend::D3d12, Backend::Metal}) {
        if (std::ranges::contains(*built, backend)) {
            continue;
        }
        const auto rhi = Instance::create({.backend = backend});

        INFO(name(backend));
        REQUIRE_FALSE(rhi.has_value());
        CHECK(rhi.error().code == Result::ErrorBackendUnavailable);
        CHECK_THAT(rhi.error().message.text(),
                   Catch::Matchers::Equals(std::format("backend {} is not in this build", name(backend))));
    }
}

TEST_CASE("Instance fails with RhiInvalidCall for a value that is no backend", "[rhi][unit]") {
    const auto rhi = Instance::create({.backend = static_cast<Backend>(0)});

    REQUIRE_FALSE(rhi.has_value());
    CHECK(rhi.error().code == Result::ErrorInvalidArgument);
    CHECK_THAT(rhi.error().message.text(), Catch::Matchers::Equals("backend 0 is not one BFR has"));
}

TEST_CASE("Instance with validation lists an adapter and reports nothing", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;

    const auto rhi = createInstance(recordingInto(backend, messages), kValidation);

    INFO(name(backend) << ' ' << failureOf(rhi));
    REQUIRE(rhi.has_value());
    const std::vector<AdapterInfo> adapters = adaptersOf(*rhi);
    CHECK_FALSE(adapters.empty());
    for (const AdapterInfo& adapter : adapters) {
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

    const auto device = rhi->createDevice(testDeviceOn(static_cast<std::uint32_t>(adaptersOf(*rhi).size()), kNoWindow));

    REQUIRE_FALSE(device.has_value());
    CHECK(device.error().code == Result::ErrorInvalidArgument);
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Instance creates a device on every adapter it lists", "[rhi][agnostic]") {
    const TestBackend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    const auto rhi = createInstance(recordingInto(backend, messages), kValidation);
    INFO(name(backend) << ' ' << failureOf(rhi));
    REQUIRE(rhi.has_value());
    const std::vector<AdapterInfo> adapters = adaptersOf(*rhi);
    REQUIRE_FALSE(adapters.empty());

    for (std::uint32_t adapter = 0; adapter < adapters.size(); ++adapter) {
        const auto device = rhi->createDevice(testDeviceOn(adapter, kNoWindow));
        INFO(adapters[adapter].name << ' ' << failureOf(device));
        REQUIRE(device.has_value());
        CHECK(static_cast<bool>(*device));
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

        const auto device = presentingDevice(*rhi, kHeadless);

        REQUIRE(static_cast<bool>(device));
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}
