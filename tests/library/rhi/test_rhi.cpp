#include "error.h"
#include "log.h"
#include "log_capture.h"
#include "rhi.h"
#include "rhi_test_support.h"
#include "scoped_env.h"
#include "window.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_container_properties.hpp>
#include <catch2/matchers/catch_matchers_quantifiers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using namespace bfr::rhi;
using namespace bfr::rhi::test;

namespace {

// The first line of log that starts with prefix; empty when none does.
std::string lineStartingWith(const std::string& log, std::string_view prefix) {
    std::istringstream lines{log};
    for (std::string line; std::getline(lines, line);) {
        if (line.starts_with(prefix)) {
            return line;
        }
    }
    return {};
}

} // namespace

TEST_CASE("Rhi with validation lists an adapter and reports nothing", "[rhi][gpu]") {
    std::vector<std::string> messages;

    const auto rhi = Rhi::create(recordingInto(messages), kValidation);

    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());
    CHECK_FALSE((*rhi)->adapters().empty());
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Rhi with validation reports a broken layer manifest", "[rhi][gpu]") {
    // The loader skips a manifest it cannot parse and reports it while it
    // creates the rhi.
    const auto layers = std::filesystem::temp_directory_path() / "bfr_broken_layer";
    std::filesystem::create_directories(layers);
    std::ofstream{layers / "broken.json"} << "{ not json";
    const char* vcpkgLayers = std::getenv("VK_ADD_LAYER_PATH");
    REQUIRE(vcpkgLayers != nullptr);
    const ScopedEnv layerPath("VK_ADD_LAYER_PATH", std::string{vcpkgLayers} + ":" + layers.string());
    std::vector<std::string> messages;

    const auto rhi = Rhi::create(recordingInto(messages), kValidation);

    std::filesystem::remove_all(layers);
    REQUIRE(rhi.has_value());
    CHECK_THAT(messages, Catch::Matchers::AnyMatch(Catch::Matchers::ContainsSubstring("broken.json")));
}

TEST_CASE("Rhi with validation fails when the layer is missing", "[rhi][gpu]") {
    const ScopedEnv hidden("VK_LOADER_LAYERS_DISABLE", "VK_LAYER_KHRONOS_validation");
    std::vector<std::string> messages;

    const auto rhi = Rhi::create(recordingInto(messages), kValidation);

    REQUIRE_FALSE(rhi.has_value());
    CHECK(rhi.error().code == bfr::core::ErrorCode::RhiValidationUnavailable);
}

TEST_CASE("Rhi with validation fails when the layer's library is missing", "[rhi][gpu]") {
    // The loader lists the layer from its manifest, then fails to load it.
    const auto layers = std::filesystem::temp_directory_path() / "bfr_missing_layer_library";
    std::filesystem::create_directories(layers);
    std::ofstream{layers / "validation.json"} << R"({
        "file_format_version": "1.2.0",
        "layer": {"name": "VK_LAYER_KHRONOS_validation", "type": "GLOBAL",
                  "library_path": "./libVkLayer_missing.so", "api_version": "1.4.0",
                  "implementation_version": "1", "description": "Names a library that does not exist"}
    })";
    const char* vcpkgLayers = std::getenv("VK_ADD_LAYER_PATH");
    REQUIRE(vcpkgLayers != nullptr);
    // First in the path, so the loader takes this manifest over vcpkg's.
    const ScopedEnv layerPath("VK_ADD_LAYER_PATH", layers.string() + ":" + vcpkgLayers);
    std::vector<std::string> messages;

    const auto rhi = Rhi::create(recordingInto(messages), kValidation);

    std::filesystem::remove_all(layers);
    INFO(failureOf(rhi));
    REQUIRE_FALSE(rhi.has_value());
    CHECK(rhi.error().code == bfr::core::ErrorCode::RhiValidationUnavailable);
    // The loader's message names the library it failed to load.
    CHECK_THAT(rhi.error().message, Catch::Matchers::ContainsSubstring("libVkLayer_missing.so"));
}

TEST_CASE("Rhi fails when no driver is installed", "[rhi][gpu]") {
    const ScopedEnv noDriver("VK_DRIVER_FILES", "/nonexistent/bfr_no_driver.json");
    std::vector<std::string> messages;

    const auto rhi = Rhi::create(recordingInto(messages), kValidation);

    REQUIRE_FALSE(rhi.has_value());
    CHECK(rhi.error().code == bfr::core::ErrorCode::RhiInstanceFailed);
}

TEST_CASE("Rhi fails to create a device on an adapter it does not list", "[rhi][gpu]") {
    std::vector<std::string> messages;
    const auto rhi = Rhi::create(recordingInto(messages), kValidation);
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());

    const auto device = (*rhi)->createDevice((*rhi)->adapters().size(), kNoWindow);

    REQUIRE_FALSE(device.has_value());
    CHECK(device.error().code == bfr::core::ErrorCode::RhiAdapterOutOfRange);
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Rhi creates a device on every adapter it lists", "[rhi][gpu]") {
    std::vector<std::string> messages;
    const auto rhi = Rhi::create(recordingInto(messages), kValidation);
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());
    REQUIRE_FALSE((*rhi)->adapters().empty());

    for (std::size_t adapter = 0; adapter < (*rhi)->adapters().size(); ++adapter) {
        const auto device = (*rhi)->createDevice(adapter, kNoWindow);
        INFO((*rhi)->adapters()[adapter].name << ' ' << failureOf(device));
        REQUIRE(device.has_value());
        CHECK(*device != nullptr);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

// Hidden: only the rhi.loader_* ctest entries run these, each with
// LD_LIBRARY_PATH at a broken loader, which glibc reads only when the process
// starts.
TEST_CASE("Rhi reports why each loader library failed", "[rhi][unit][.]") {
    const auto rhi = Rhi::create({}, {});

    REQUIRE_FALSE(rhi.has_value());
    CHECK(rhi.error().code == bfr::core::ErrorCode::RhiLoaderFailed);
    CHECK_THAT(rhi.error().message, Catch::Matchers::ContainsSubstring("libvulkan.so.1: file too short") &&
                                        Catch::Matchers::ContainsSubstring("libvulkan.so: file too short"));
}

TEST_CASE("Rhi reports a loader library without vkGetInstanceProcAddr", "[rhi][unit][.]") {
    const auto rhi = Rhi::create({}, {});

    REQUIRE_FALSE(rhi.has_value());
    CHECK(rhi.error().code == bfr::core::ErrorCode::RhiLoaderFailed);
    CHECK_THAT(rhi.error().message,
               Catch::Matchers::ContainsSubstring("libvulkan.so.1: undefined symbol: vkGetInstanceProcAddr") &&
                   Catch::Matchers::ContainsSubstring("libvulkan.so: undefined symbol: vkGetInstanceProcAddr"));
}

// Hidden: only the rhi.fake_loader_* ctest entries run these, each with
// LD_LIBRARY_PATH at a loader that misbehaves as BFR_FAKE_LOADER_MODE says.
TEST_CASE("Rhi with validation reports a failed layer listing", "[rhi][unit][.]") {
    const auto rhi = Rhi::create({}, kValidation);

    REQUIRE_FALSE(rhi.has_value());
    CHECK(rhi.error().code == bfr::core::ErrorCode::RhiValidationUnavailable);
    CHECK_THAT(rhi.error().message,
               Catch::Matchers::ContainsSubstring("vkEnumerateInstanceLayerProperties: VK_ERROR_UNKNOWN"));
}

TEST_CASE("Rhi with validation reports a failed extension listing", "[rhi][unit][.]") {
    const auto rhi = Rhi::create({}, kValidation);

    REQUIRE_FALSE(rhi.has_value());
    CHECK(rhi.error().code == bfr::core::ErrorCode::RhiValidationUnavailable);
    CHECK_THAT(rhi.error().message,
               Catch::Matchers::ContainsSubstring("vkEnumerateInstanceExtensionProperties: VK_ERROR_UNKNOWN"));
}

TEST_CASE("Rhi with validation lists again when a listing grows", "[rhi][unit][.]") {
    const auto rhi = Rhi::create({}, kValidation);

    // Past both listings: only the fake vkCreateInstance fails.
    INFO(failureOf(rhi));
    REQUIRE_FALSE(rhi.has_value());
    CHECK(rhi.error().code == bfr::core::ErrorCode::RhiInstanceFailed);
    CHECK_THAT(rhi.error().message,
               Catch::Matchers::ContainsSubstring("vkCreateInstance: VK_ERROR_INITIALIZATION_FAILED"));
}

// Hidden: only the rhi.portability_driver ctest entry runs this, with
// VK_DRIVER_FILES at lavapipe marked as a portability driver.
TEST_CASE("Rhi lists an adapter from a portability driver", "[rhi][gpu][.]") {
    const auto rhi = Rhi::create({}, {});

    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());
    CHECK_FALSE((*rhi)->adapters().empty());
}

TEST_CASE("Rhi skips an adapter it fails to check, and says which call failed", "[rhi][gpu]") {
    const AdapterOverrideLayer adapterOverride;
    // Mesa's device-select layer lists device extensions itself, ignores the
    // failure, and allocates from the count it never got.
    const ScopedEnv noDeviceSelect("NODEVICE_SELECT", "1");
    const ScopedEnv failing("BFR_FAIL", "vkEnumerateDeviceExtensionProperties");
    std::vector<std::string> messages;

    const auto rhi = Rhi::create(recordingInto(messages), {});

    REQUIRE_FALSE(rhi.has_value());
    CHECK(rhi.error().code == bfr::core::ErrorCode::RhiNoAdapter);
    CHECK_THAT(rhi.error().message, Catch::Matchers::StartsWith("no usable adapter") &&
                                        Catch::Matchers::ContainsSubstring(
                                            "skipped, vkEnumerateDeviceExtensionProperties: VK_ERROR_UNKNOWN"));
    if (bfr::core::kLoggingEnabled) {
        CHECK_THAT(messages, Catch::Matchers::AnyMatch(Catch::Matchers::ContainsSubstring("skipped, ")));
    }
}

TEST_CASE("Rhi fails with RhiNoAdapter when every adapter is below the floor", "[rhi][gpu]") {
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv version("BFR_ADAPTER_MINOR", "2");

    const auto rhi = Rhi::create({}, {});

    REQUIRE_FALSE(rhi.has_value());
    CHECK(rhi.error().code == bfr::core::ErrorCode::RhiNoAdapter);
    CHECK_THAT(rhi.error().message, Catch::Matchers::ContainsSubstring("Vulkan 1.2, needs 1.3"));
}

TEST_CASE("Rhi fails with RhiNoAdapter when an adapter lacks a floor feature, and names it", "[rhi][gpu]") {
    const std::string feature =
        GENERATE("descriptorIndexing", "runtimeDescriptorArray", "descriptorBindingPartiallyBound",
                 "descriptorBindingVariableDescriptorCount", "descriptorBindingUpdateUnusedWhilePending",
                 "descriptorBindingSampledImageUpdateAfterBind", "descriptorBindingStorageImageUpdateAfterBind",
                 "descriptorBindingStorageBufferUpdateAfterBind", "shaderSampledImageArrayNonUniformIndexing",
                 "shaderStorageBufferArrayNonUniformIndexing", "shaderStorageImageArrayNonUniformIndexing",
                 "timelineSemaphore", "bufferDeviceAddress", "dynamicRendering", "synchronization2", "maintenance4");
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv hidden("BFR_ADAPTER_NO_FEATURE", feature);

    const auto rhi = Rhi::create({}, {});

    INFO(feature);
    REQUIRE_FALSE(rhi.has_value());
    CHECK(rhi.error().code == bfr::core::ErrorCode::RhiNoAdapter);
    CHECK_THAT(rhi.error().message, Catch::Matchers::ContainsSubstring("below the floor, lacks " + feature));
}

TEST_CASE("Rhi lists a Vulkan 1.3 adapter without an extension that 1.3 made core", "[rhi][gpu]") {
    const std::string extension = GENERATE("VK_KHR_dynamic_rendering", "VK_KHR_synchronization2");
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv version("BFR_ADAPTER_MINOR", "3");
    const ScopedEnv hidden("BFR_ADAPTER_HIDE", extension);

    const auto rhi = Rhi::create({}, {});

    INFO(extension << ' ' << failureOf(rhi));
    REQUIRE(rhi.has_value());
    CHECK_FALSE((*rhi)->adapters().empty());
}

TEST_CASE("Rhi fails with RhiNoAdapter when a Vulkan 1.3 adapter lacks the swapchain extension", "[rhi][gpu]") {
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv version("BFR_ADAPTER_MINOR", "3");
    const ScopedEnv hidden("BFR_ADAPTER_HIDE", "VK_KHR_swapchain");

    const auto rhi = Rhi::create({}, {});

    REQUIRE_FALSE(rhi.has_value());
    CHECK(rhi.error().code == bfr::core::ErrorCode::RhiNoAdapter);
    CHECK_THAT(rhi.error().message, Catch::Matchers::ContainsSubstring("lacks VK_KHR_swapchain"));
}

TEST_CASE("Rhi fails with RhiNoAdapter when no queue family runs compute", "[rhi][gpu]") {
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv noCompute("BFR_ADAPTER_NO_COMPUTE", "1");

    const auto rhi = Rhi::create({}, {});

    REQUIRE_FALSE(rhi.has_value());
    CHECK(rhi.error().code == bfr::core::ErrorCode::RhiNoAdapter);
    CHECK_THAT(rhi.error().message, Catch::Matchers::ContainsSubstring("no queue family runs compute"));
}

TEST_CASE("Rhi fails with RhiNoAdapter when no queue family runs graphics", "[rhi][gpu]") {
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv noGraphics("BFR_ADAPTER_NO_GRAPHICS", "1");

    const auto rhi = Rhi::create({}, {});

    REQUIRE_FALSE(rhi.has_value());
    CHECK(rhi.error().code == bfr::core::ErrorCode::RhiNoAdapter);
    CHECK_THAT(rhi.error().message, Catch::Matchers::ContainsSubstring("no queue family runs graphics"));
}

TEST_CASE("Rhi creates a device that presents to a headless window", "[rhi][gpu]") {
    std::vector<std::string> messages;
    {
        const auto rhi = Rhi::create(recordingInto(messages), kValidation);
        INFO(failureOf(rhi));
        REQUIRE(rhi.has_value());

        const auto device = presentingDevice(**rhi, bfr::core::window::Headless{});

        REQUIRE(device != nullptr);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("Rhi fails with RhiPresentUnsupported when no queue family presents to the window", "[rhi][gpu]") {
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv unsupported("BFR_PRESENT_UNSUPPORTED", "1");
    const auto rhi = Rhi::create({}, {});
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());

    const auto device = (*rhi)->createDevice(headlessAdapter(**rhi), bfr::core::window::Headless{});

    REQUIRE_FALSE(device.has_value());
    CHECK(device.error().code == bfr::core::ErrorCode::RhiPresentUnsupported);
    CHECK_THAT(device.error().message, Catch::Matchers::ContainsSubstring("presents to the window"));
}

TEST_CASE("Rhi fails with RhiSurfaceFailed when the API refuses a surface for the device's window", "[rhi][gpu]") {
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv refused("BFR_FAIL", "vkCreateHeadlessSurfaceEXT");
    const auto rhi = Rhi::create({}, {});
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());

    const auto device = (*rhi)->createDevice(0, bfr::core::window::Headless{});

    REQUIRE_FALSE(device.has_value());
    CHECK(device.error().code == bfr::core::ErrorCode::RhiSurfaceFailed);
    CHECK_THAT(device.error().message,
               Catch::Matchers::ContainsSubstring("vkCreateHeadlessSurfaceEXT: VK_ERROR_UNKNOWN"));
}

// In these two, lavapipe reports NVIDIA's driver ID and stands in for NVIDIA's
// driver, which has no headless surfaces and crashes when a loader before
// 1.4.319 asks it about one.
TEST_CASE("Rhi doesn't ask NVIDIA's driver about a headless window on a loader that lets it crash", "[rhi][gpu]") {
    const LoaderVersion loader{"1.4.318"};
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv nvidia("BFR_ADAPTER_DRIVER_ID", std::to_string(VK_DRIVER_ID_NVIDIA_PROPRIETARY));
    std::ostringstream log;
    const auto rhi = Rhi::create({.logger = loggingInto(log)}, {});
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());

    const auto device = (*rhi)->createDevice(headlessAdapter(**rhi), bfr::core::window::Headless{});

    REQUIRE_FALSE(device.has_value());
    CHECK(device.error().code == bfr::core::ErrorCode::RhiPresentUnsupported);
    if (bfr::core::kLoggingEnabled) {
        CHECK_THAT(log.str(), Catch::Matchers::ContainsSubstring(
                                  "does not present to headless surfaces; not asked, since loader 1.4.318"));
    }
}

TEST_CASE("Rhi asks NVIDIA's driver about a headless window on a loader that answers for it", "[rhi][gpu]") {
    const LoaderVersion loader{"1.4.319"};
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv nvidia("BFR_ADAPTER_DRIVER_ID", std::to_string(VK_DRIVER_ID_NVIDIA_PROPRIETARY));
    const auto rhi = Rhi::create({}, {});
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());

    const auto device = (*rhi)->createDevice(headlessAdapter(**rhi), bfr::core::window::Headless{});

    INFO(failureOf(device));
    CHECK(device.has_value());
}

TEST_CASE("Rhi fails with RhiDeviceFailed when the API refuses the device and leaves nothing behind", "[rhi][gpu]") {
    const AdapterOverrideLayer adapterOverride{"VK_LAYER_KHRONOS_validation"};
    const ScopedEnv failing("BFR_FAIL", "vkCreateDevice");
    std::vector<std::string> errors;
    {
        const auto rhi = Rhi::create(recordingErrorsInto(errors), kValidation);
        INFO(failureOf(rhi));
        REQUIRE(rhi.has_value());

        const auto device = (*rhi)->createDevice(0, kNoWindow);

        REQUIRE_FALSE(device.has_value());
        CHECK(device.error().code == bfr::core::ErrorCode::RhiDeviceFailed);
        CHECK_THAT(device.error().message,
                   Catch::Matchers::ContainsSubstring("vkCreateDevice on " + (*rhi)->adapters()[0].name +
                                                      ": VK_ERROR_UNKNOWN"));
    }
    // The loader reports the refusal; anything more would be the RHI misusing
    // what the failed call left.
    CHECK_THAT(errors, Catch::Matchers::AllMatch(Catch::Matchers::ContainsSubstring("Failed to create device chain")));
}

TEST_CASE("Rhi creates a valid device on an adapter that lists VK_KHR_portability_subset", "[rhi][gpu]") {
    const AdapterOverrideLayer adapterOverride{"VK_LAYER_KHRONOS_validation"};
    const ScopedEnv added("BFR_ADAPTER_ADD", "VK_KHR_portability_subset");
    // Errors only: the loader also reports layer warnings.
    std::vector<std::string> errors;
    const auto rhi = Rhi::create(recordingErrorsInto(errors), kValidation);
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());

    const auto device = (*rhi)->createDevice(0, kNoWindow);

    INFO(failureOf(device));
    REQUIRE(device.has_value());
    CHECK_THAT(errors, Catch::Matchers::IsEmpty());
}

TEST_CASE("Rhi logs the loader and each adapter it lists", "[rhi][gpu]") {
    if (!bfr::core::kLoggingEnabled) {
        SKIP("logging is compiled out");
    }
    std::ostringstream log;

    const auto rhi = Rhi::create({.logger = loggingInto(log)}, {});

    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());
    CHECK_THAT(log.str(), Catch::Matchers::ContainsSubstring("Vulkan loader 1.") &&
                              Catch::Matchers::ContainsSubstring("Vulkan instance 1.3, validation off"));
    for (const Adapter& adapter : (*rhi)->adapters()) {
        CHECK_THAT(lineStartingWith(log.str(), "adapter " + adapter.name + ", "),
                   Catch::Matchers::EndsWith(": listed"));
    }
}

TEST_CASE("Rhi logs why an adapter is below the floor", "[rhi][gpu]") {
    if (!bfr::core::kLoggingEnabled) {
        SKIP("logging is compiled out");
    }
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv version("BFR_ADAPTER_MINOR", "2");
    std::ostringstream log;

    const auto rhi = Rhi::create({.logger = loggingInto(log)}, {});

    REQUIRE_FALSE(rhi.has_value());
    CHECK_THAT(log.str(), Catch::Matchers::ContainsSubstring(", Vulkan 1.2: below the floor, Vulkan 1.2, needs 1.3"));
}

TEST_CASE("Rhi logs the validation layer's version", "[rhi][gpu]") {
    if (!bfr::core::kLoggingEnabled) {
        SKIP("logging is compiled out");
    }
    std::ostringstream log;

    const auto rhi = Rhi::create({.logger = loggingInto(log)}, kValidation);

    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());
    CHECK_THAT(log.str(), Catch::Matchers::ContainsSubstring("VK_LAYER_KHRONOS_validation 1."));
}

TEST_CASE("Rhi logs the device it creates, its queue families and its extensions", "[rhi][gpu]") {
    if (!bfr::core::kLoggingEnabled) {
        SKIP("logging is compiled out");
    }
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv added("BFR_ADAPTER_ADD", "VK_KHR_portability_subset");
    // Outlives the Rhi, which logs as it goes.
    std::ostringstream log;
    const auto rhi = Rhi::create({.logger = loggingInto(log)}, {});
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());

    // lavapipe has one queue family.
    const auto device = (*rhi)->createDevice(headlessAdapter(**rhi), kNoWindow);

    INFO(failureOf(device));
    REQUIRE(device.has_value());
    CHECK_THAT(log.str(), Catch::Matchers::ContainsSubstring(
                              "device on " + (*rhi)->adapters()[headlessAdapter(**rhi)].name +
                              ", queue families 0, extensions: VK_KHR_swapchain, VK_KHR_portability_subset"));
}
