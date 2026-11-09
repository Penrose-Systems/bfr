// What the Vulkan backend's Instance does with its loader, layers, drivers and
// adapters: Vulkan's own behaviour, tested through the interface.

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/window.h"
#include "bfr/rhi/acceleration_structure.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/instance.h"
#include "bfr/rhi/swapchain.h"
#include "bfr/rhi/timestamp.h"
#include "log_capture.h"
#include "rhi/rhi_test_support.h"
#include "scoped_env.h"
#include "shaders/spirv.h"
#include "test_backends.h"
#include "vulkan_test_support.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_container_properties.hpp>
#include <catch2/matchers/catch_matchers_quantifiers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <vulkan/vulkan_core.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace bfr::rhi;
using namespace bfr::rhi::vulkan::test;

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

// A description on the Vulkan backend that logs into log.
InstanceDesc vulkanLoggingInto(std::ostringstream& log) {
    InstanceDesc desc = vulkanDesc();
    desc.logger = loggingInto(log);
    return desc;
}

} // namespace

TEST_CASE("Vulkan with validation reports a broken layer manifest", "[vulkan][gpu]") {
    // The loader skips a manifest it cannot parse and reports it while it
    // creates the rhi.
    const auto layers = std::filesystem::temp_directory_path() / "bfr_broken_layer";
    std::filesystem::create_directories(layers);
    std::ofstream{layers / "broken.json"} << "{ not json";
    const char* vcpkgLayers = std::getenv("VK_ADD_LAYER_PATH");
    REQUIRE(vcpkgLayers != nullptr);
    const ScopedEnv layerPath("VK_ADD_LAYER_PATH", std::string{vcpkgLayers} + ":" + layers.string());
    std::vector<std::string> messages;

    const auto rhi = Instance::create(recordingInto(messages), kValidation);

    std::filesystem::remove_all(layers);
    REQUIRE(rhi.has_value());
    CHECK_THAT(messages, Catch::Matchers::AnyMatch(Catch::Matchers::ContainsSubstring("broken.json")));
}

TEST_CASE("Vulkan with validation fails when the layer is missing", "[vulkan][gpu]") {
    const ScopedEnv hidden("VK_LOADER_LAYERS_DISABLE", "VK_LAYER_KHRONOS_validation");
    std::vector<std::string> messages;

    const auto rhi = Instance::create(recordingInto(messages), kValidation);

    REQUIRE_FALSE(rhi.has_value());
    CHECK(rhi.error().code == bfr::core::ErrorCode::RhiValidationUnavailable);
}

TEST_CASE("Vulkan with validation fails when the layer's library is missing", "[vulkan][gpu]") {
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

    const auto rhi = Instance::create(recordingInto(messages), kValidation);

    std::filesystem::remove_all(layers);
    INFO(failureOf(rhi));
    REQUIRE_FALSE(rhi.has_value());
    CHECK(rhi.error().code == bfr::core::ErrorCode::RhiValidationUnavailable);
    // The loader's message names the library it failed to load.
    CHECK_THAT(rhi.error().message.text(), Catch::Matchers::ContainsSubstring("libVkLayer_missing.so"));
}

TEST_CASE("Vulkan fails when no driver is installed", "[vulkan][gpu]") {
    const ScopedEnv noDriver("VK_DRIVER_FILES", "/nonexistent/bfr_no_driver.json");
    std::vector<std::string> messages;

    const auto rhi = Instance::create(recordingInto(messages), kValidation);

    REQUIRE_FALSE(rhi.has_value());
    CHECK(rhi.error().code == bfr::core::ErrorCode::RhiInstanceFailed);
}

// Hidden: only the vulkan.loader_* ctest entries run these, each with
// LD_LIBRARY_PATH at a broken loader, which glibc reads only when the process
// starts.
TEST_CASE("Vulkan reports why each loader library failed", "[vulkan][unit][.]") {
    const auto rhi = Instance::create(vulkanDesc(), {});

    REQUIRE_FALSE(rhi.has_value());
    CHECK(rhi.error().code == bfr::core::ErrorCode::RhiLoaderFailed);
    CHECK_THAT(rhi.error().message.text(), Catch::Matchers::ContainsSubstring("libvulkan.so.1: file too short") &&
                                               Catch::Matchers::ContainsSubstring("libvulkan.so: file too short"));
}

TEST_CASE("Vulkan reports a loader library without vkGetInstanceProcAddr", "[vulkan][unit][.]") {
    const auto rhi = Instance::create(vulkanDesc(), {});

    REQUIRE_FALSE(rhi.has_value());
    CHECK(rhi.error().code == bfr::core::ErrorCode::RhiLoaderFailed);
    CHECK_THAT(rhi.error().message.text(),
               Catch::Matchers::ContainsSubstring("libvulkan.so.1: undefined symbol: vkGetInstanceProcAddr") &&
                   Catch::Matchers::ContainsSubstring("libvulkan.so: undefined symbol: vkGetInstanceProcAddr"));
}

// Hidden: only the vulkan.fake_loader_* ctest entries run these, each with
// LD_LIBRARY_PATH at a loader that misbehaves as BFR_FAKE_LOADER_MODE says.
TEST_CASE("Vulkan reports a failed extension listing", "[vulkan][unit][.]") {
    const auto rhi = Instance::create(vulkanDesc(), kValidation);

    REQUIRE_FALSE(rhi.has_value());
    CHECK(rhi.error().code == bfr::core::ErrorCode::RhiInstanceFailed);
    CHECK_THAT(rhi.error().message.text(),
               Catch::Matchers::ContainsSubstring("vkEnumerateInstanceExtensionProperties: VK_ERROR_UNKNOWN"));
}

TEST_CASE("Vulkan lists again when a listing grows", "[vulkan][unit][.]") {
    const auto rhi = Instance::create(vulkanDesc(), kValidation);

    // Past the listing: only the fake vkCreateInstance fails.
    INFO(failureOf(rhi));
    REQUIRE_FALSE(rhi.has_value());
    CHECK(rhi.error().code == bfr::core::ErrorCode::RhiInstanceFailed);
    CHECK_THAT(rhi.error().message.text(),
               Catch::Matchers::ContainsSubstring("vkCreateInstance: VK_ERROR_INITIALIZATION_FAILED"));
}

// Hidden: only the vulkan.portability_driver ctest entry runs this, with
// VK_DRIVER_FILES at lavapipe marked as a portability driver.
TEST_CASE("Vulkan lists an adapter from a portability driver", "[vulkan][gpu][.]") {
    const auto rhi = Instance::create(vulkanDesc(), {});

    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());
    CHECK_FALSE((*rhi)->adapters().empty());
}

TEST_CASE("Vulkan passes the caller's application and engine to VkApplicationInfo", "[vulkan][gpu]") {
    constexpr std::uint32_t kApplicationVersion = 3;
    constexpr std::uint32_t kEngineVersion = 7;
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv expected("BFR_EXPECT_APPLICATION", "Test App,3,Test Engine,7");
    InstanceDesc named = vulkanDesc();
    named.applicationName = "Test App";
    named.applicationVersion = kApplicationVersion;
    named.engineName = "Test Engine";
    named.engineVersion = kEngineVersion;

    const auto rhi = Instance::create(named, {});
    INFO(failureOf(rhi));
    CHECK(rhi.has_value());
    const auto unnamed = Instance::create(vulkanDesc(), {});
    REQUIRE_FALSE(unnamed.has_value());
    CHECK(unnamed.error().code == bfr::core::ErrorCode::RhiInstanceFailed);
}

TEST_CASE("Vulkan skips an adapter it fails to check, and says which call failed", "[vulkan][gpu]") {
    const AdapterOverrideLayer adapterOverride;
    // Mesa's device-select layer lists device extensions itself, ignores the
    // failure, and allocates from the count it never got.
    const ScopedEnv noDeviceSelect("NODEVICE_SELECT", "1");
    const ScopedEnv failing("BFR_FAIL", "vkEnumerateDeviceExtensionProperties");
    std::vector<std::string> messages;

    const auto rhi = Instance::create(recordingInto(messages), {});

    REQUIRE_FALSE(rhi.has_value());
    CHECK(rhi.error().code == bfr::core::ErrorCode::RhiNoAdapter);
    CHECK_THAT(rhi.error().message.text(), Catch::Matchers::StartsWith("no usable adapter") &&
                                               Catch::Matchers::ContainsSubstring(
                                                   "skipped, vkEnumerateDeviceExtensionProperties: VK_ERROR_UNKNOWN"));
    if (bfr::core::kLoggingEnabled) {
        CHECK_THAT(messages, Catch::Matchers::AnyMatch(Catch::Matchers::ContainsSubstring("skipped, ")));
    }
}

TEST_CASE("Vulkan fails with RhiUnsupported for a version below BFR's minimum", "[vulkan][unit]") {
    const auto [major, minor] = GENERATE(std::pair{0U, 0U}, std::pair{1U, 0U}, std::pair{1U, 1U});

    const auto rhi = Instance::create(vulkanAt(major, minor), {});

    INFO(major << '.' << minor);
    REQUIRE_FALSE(rhi.has_value());
    CHECK(rhi.error().code == bfr::core::ErrorCode::RhiUnsupported);
    CHECK_THAT(rhi.error().message.text(),
               Catch::Matchers::Equals(std::format("Vulkan {}.{} is below BFR's minimum, 1.2", major, minor)));
}

TEST_CASE("Vulkan fails with RhiInvalidCall for a version Vulkan can't name", "[vulkan][unit]") {
    // A major past 7 bits, a minor past 10 (vulkan_core.h:62-63).
    constexpr std::uint32_t kPastMajor = 128;
    constexpr std::uint32_t kPastMinor = 1024;
    const auto [major, minor] = GENERATE_COPY(std::pair{kPastMajor, 0U}, std::pair{1U, kPastMinor});

    const auto rhi = Instance::create(vulkanAt(major, minor), {});

    INFO(major << '.' << minor);
    REQUIRE_FALSE(rhi.has_value());
    CHECK(rhi.error().code == bfr::core::ErrorCode::RhiInvalidCall);
}

TEST_CASE("Vulkan makes the instance at a version past every adapter's, and opens a valid device on each",
          "[vulkan][gpu]") {
    // Past any Vulkan there is; the loader takes any version
    // (initialization.adoc:883-910).
    constexpr std::uint32_t kPastNewest = 9;
    std::vector<std::string> errors;
    InstanceDesc desc = recordingErrorsInto(errors);
    desc.apiMajor = 1;
    desc.apiMinor = kPastNewest;

    const auto rhi = Instance::create(desc, kValidation);

    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());
    for (std::size_t adapter = 0; adapter < (*rhi)->adapters().size(); ++adapter) {
        const auto device = (*rhi)->createDevice(adapter, kNoWindow, kTestDevice);
        INFO((*rhi)->adapters()[adapter].name << ' ' << failureOf(device));
        CHECK(device.has_value());
    }
    CHECK_THAT(errors, Catch::Matchers::IsEmpty());
}

TEST_CASE("Vulkan at 1.2 reports nothing for a format 1.3 added that the adapter lists no extension for",
          "[vulkan][caps][gpu]") {
    const auto [extension, format] = GENERATE(std::pair{"VK_EXT_4444_formats", Format::Bgra4Unorm},
                                              std::pair{"VK_EXT_texture_compression_astc_hdr", Format::Astc4x4Float});
    const AdapterOverrideLayer adapterOverride{"VK_LAYER_KHRONOS_validation"};
    const ScopedEnv hidden("BFR_ADAPTER_HIDE", extension);
    std::vector<std::string> errors;
    {
        InstanceDesc desc = recordingErrorsInto(errors);
        desc.apiMajor = 1;
        desc.apiMinor = 2;
        const auto rhi = Instance::create(desc, kValidation);
        INFO(failureOf(rhi));
        REQUIRE(rhi.has_value());
        const auto caps = (*rhi)->formatCaps(0, format);

        INFO(extension);
        REQUIRE(caps.has_value());
        CHECK_FALSE((caps->sampled || caps->storage || caps->colorAttachment || caps->copySrc || caps->copyDst));
    }
    CHECK_THAT(errors, Catch::Matchers::IsEmpty());
}

TEST_CASE("Vulkan at 1.2 reports a format through the 32-bit query without VK_KHR_format_feature_flags2",
          "[vulkan][caps][gpu]") {
    const AdapterOverrideLayer adapterOverride{"VK_LAYER_KHRONOS_validation"};
    const ScopedEnv hidden("BFR_ADAPTER_HIDE", "VK_KHR_format_feature_flags2");
    std::vector<std::string> errors;
    {
        InstanceDesc desc = recordingErrorsInto(errors);
        desc.apiMajor = 1;
        desc.apiMinor = 2;
        const auto rhi = Instance::create(desc, kValidation);
        INFO(failureOf(rhi));
        REQUIRE(rhi.has_value());

        // Every Vulkan device samples, filters, blends and renders to it
        // (formats.adoc:3509, the required format support table).
        const auto caps = (*rhi)->formatCaps(0, Format::Rgba8Unorm);

        REQUIRE(caps.has_value());
        CHECK(caps->sampled);
        CHECK(caps->filterLinear);
        CHECK(caps->colorAttachment);
        CHECK(caps->blend);
    }
    CHECK_THAT(errors, Catch::Matchers::IsEmpty());
}

TEST_CASE("Vulkan fails with RhiNoAdapter when every adapter is below BFR's minimum", "[vulkan][gpu]") {
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv version("BFR_ADAPTER_MINOR", "1");

    const auto rhi = Instance::create(vulkanDesc(), {});

    REQUIRE_FALSE(rhi.has_value());
    CHECK(rhi.error().code == bfr::core::ErrorCode::RhiNoAdapter);
    CHECK_THAT(rhi.error().message.text(),
               Catch::Matchers::ContainsSubstring("below BFR's minimum, Vulkan 1.1, needs 1.2"));
}

TEST_CASE("Vulkan fails with RhiNoAdapter when an adapter lacks a feature BFR needs, at 1.2 and later, and names it",
          "[vulkan][gpu]") {
    const std::string feature = GENERATE(
        "descriptorIndexing", "runtimeDescriptorArray", "descriptorBindingPartiallyBound",
        "descriptorBindingSampledImageUpdateAfterBind", "descriptorBindingStorageImageUpdateAfterBind",
        "descriptorBindingStorageBufferUpdateAfterBind", "shaderSampledImageArrayNonUniformIndexing",
        "shaderStorageBufferArrayNonUniformIndexing", "timelineSemaphore", "dynamicRendering", "synchronization2");
    // At 1.2 the extensions 1.3 made core report dynamicRendering and
    // synchronization2.
    const std::uint32_t minor = GENERATE(2U, bfr::test::testApiVersion("vulkan").minor);
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv hidden("BFR_ADAPTER_NO_FEATURE", feature);

    const auto rhi = Instance::create(vulkanAt(1, minor), {});

    INFO(feature << " at 1." << minor);
    REQUIRE_FALSE(rhi.has_value());
    CHECK(rhi.error().code == bfr::core::ErrorCode::RhiNoAdapter);
    CHECK_THAT(rhi.error().message.text(), Catch::Matchers::ContainsSubstring("below BFR's minimum, lacks " + feature));
}

TEST_CASE("Vulkan lists no 1.2 adapter without an extension 1.3 made core that BFR needs", "[vulkan][gpu]") {
    const auto [extension, feature] = GENERATE(std::pair{"VK_KHR_dynamic_rendering", "dynamicRendering"},
                                               std::pair{"VK_KHR_synchronization2", "synchronization2"});
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv hidden("BFR_ADAPTER_HIDE", extension);

    const auto rhi = Instance::create(vulkanAt(1, 2), {});

    INFO(extension);
    REQUIRE_FALSE(rhi.has_value());
    CHECK(rhi.error().code == bfr::core::ErrorCode::RhiNoAdapter);
    CHECK_THAT(rhi.error().message.text(),
               Catch::Matchers::ContainsSubstring(std::string{"below BFR's minimum, lacks "} + feature));
}

TEST_CASE("Vulkan lists an adapter without non-uniform storage texture indexing, and reports it", "[vulkan][gpu]") {
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv hidden("BFR_ADAPTER_NO_FEATURE", "shaderStorageImageArrayNonUniformIndexing");

    const auto rhi = Instance::create(vulkanDesc(), {});

    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());
    for (const Adapter& adapter : (*rhi)->adapters()) {
        INFO(adapter.name);
        CHECK_FALSE(adapter.caps.features.storageTextureNonUniformIndexing);
    }
}

TEST_CASE("Vulkan without VK_EXT_mesh_shader reports no mesh shaders, and refuses mesh pipelines and draws",
          "[vulkan][gpu]") {
    const AdapterOverrideLayer adapterOverride{"VK_LAYER_KHRONOS_validation"};
    const ScopedEnv hidden("BFR_ADAPTER_HIDE", "VK_EXT_mesh_shader");
    std::vector<std::string> errors;
    {
        auto created = Instance::create(recordingErrorsInto(errors), kValidation);
        INFO(failureOf(created));
        REQUIRE(created.has_value());
        const std::unique_ptr<Instance> instance = std::move(*created);
        const Caps& caps = instance->adapters()[0].caps;
        CHECK_FALSE(caps.features.meshShader);
        CHECK_FALSE(caps.features.taskShader);
        CHECK(caps.mesh.maxMeshOutputVertices == 0);
        CHECK_FALSE(caps.timestamps.insideRendering.mesh);
        auto made = instance->createDevice(0, kNoWindow, kTestDevice);
        INFO(failureOf(made));
        REQUIRE(made.has_value());
        Device& device = **made;
        GraphicsPipelineDesc meshDesc{.vertex = {},
                                      .fragment = {.code = std::as_bytes(std::span{bfr::test::spirv::kColorFrag}),
                                                   .entryPoint = "main"},
                                      .mesh = {.code = std::as_bytes(std::span{bfr::test::spirv::kTriangleMesh}),
                                               .entryPoint = "main"},
                                      .meshWorkgroupSize = {1, 1, 1},
                                      .topology = Topology::TriangleList,
                                      .primitiveRestart = PrimitiveRestart::Off,
                                      .raster = {.cull = CullMode::None},
                                      .depthStencil = {},
                                      .colorTargetCount = 1,
                                      .samples = 1,
                                      .sampleMask = BFR_MASK_ALL};
        meshDesc.colorTargets[0] = {.format = Format::Rgba8Unorm, .blend = {.writeMask = BFR_MASK_ALL}};
        const auto pipeline = device.createGraphicsPipeline(meshDesc);
        REQUIRE_FALSE(pipeline.has_value());
        CHECK(pipeline.error().code == bfr::core::ErrorCode::RhiUnsupported);
        const auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        const auto list = device.beginCommands(*pool);
        REQUIRE(list.has_value());

        const auto drawn = device.drawMesh(*list, 1, 1, 1);
        const auto indirect = device.drawMeshIndirect(*list, BufferHandle{}, 0, 1, 0);
        const auto counted = device.drawMeshIndirectCount(*list, BufferHandle{}, 0, BufferHandle{}, 0, 1, 0);
        const auto timestamps = device.createTimestampPool({.queue = 0, .count = 1});
        REQUIRE(timestamps.has_value());
        const auto timed = device.writeTimestamp(*list, *timestamps, 0, TimestampStage::Mesh,
                                                 TimestampGranularity::Relaxed);

        for (const auto* result : {&drawn, &indirect, &counted, &timed}) {
            REQUIRE_FALSE(result->has_value());
            CHECK(result->error().code == bfr::core::ErrorCode::RhiUnsupported);
        }
        CHECK(device.destroyTimestampPool(*timestamps).has_value());
        CHECK(device.destroyCommandList(*list).has_value());
        CHECK(device.destroyCommandPool(*pool).has_value());
    }
    CHECK_THAT(errors, Catch::Matchers::IsEmpty());
}

TEST_CASE("Vulkan without VK_KHR_acceleration_structure reports none, and refuses structures and their table",
          "[vulkan][gpu]") {
    const AdapterOverrideLayer adapterOverride{"VK_LAYER_KHRONOS_validation"};
    const ScopedEnv hidden("BFR_ADAPTER_HIDE", "VK_KHR_acceleration_structure");
    std::vector<std::string> errors;
    {
        auto created = Instance::create(recordingErrorsInto(errors), kValidation);
        INFO(failureOf(created));
        REQUIRE(created.has_value());
        const std::unique_ptr<Instance> instance = std::move(*created);
        const Caps& caps = instance->adapters()[0].caps;
        CHECK_FALSE(caps.features.accelerationStructure);
        CHECK_FALSE(caps.features.rayQuery);
        CHECK_FALSE(caps.features.rayQueryGraphics);
        CHECK(caps.rayTracing.maxInstances == 0);
        CHECK(caps.rayTracing.instanceSize == 0);
        CHECK(caps.bindless.maxAccelerationStructures == 0);
        CHECK_FALSE(caps.timestamps.outsideRendering.accelerationStructureBuild);
        const auto tabled = instance->createDevice(0, kNoWindow, testDeviceWith([](DeviceDesc& desc) {
                                                       desc.maxBindlessAccelerationStructures = 1;
                                                   }));
        REQUIRE_FALSE(tabled.has_value());
        CHECK(tabled.error().code == bfr::core::ErrorCode::RhiLimitReached);
        auto made = instance->createDevice(0, kNoWindow, kTestDevice);
        INFO(failureOf(made));
        REQUIRE(made.has_value());
        Device& device = **made;
        const auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        const auto list = device.beginCommands(*pool);
        REQUIRE(list.has_value());
        std::array<std::byte, 1> out{};

        const auto sized = device.accelerationStructureSizes({});
        const auto structure = device.createAccelerationStructure({.size = 1});
        const auto written = device.writeInstances({}, out);
        const auto built = device.buildAccelerationStructure(*list, {});
        const auto copied = device.copyAccelerationStructure(*list, {}, {});
        const auto timestamps = device.createTimestampPool({.queue = 0, .count = 1});
        REQUIRE(timestamps.has_value());
        const auto timed = device.writeTimestamp(*list, *timestamps, 0, TimestampStage::AccelerationStructureBuild,
                                                 TimestampGranularity::Relaxed);

        CHECK(sized.error().code == bfr::core::ErrorCode::RhiUnsupported);
        CHECK(structure.error().code == bfr::core::ErrorCode::RhiUnsupported);
        for (const auto* result : {&written, &built, &copied, &timed}) {
            REQUIRE_FALSE(result->has_value());
            CHECK(result->error().code == bfr::core::ErrorCode::RhiUnsupported);
        }
        CHECK(device.destroyTimestampPool(*timestamps).has_value());
        CHECK(device.destroyCommandList(*list).has_value());
        CHECK(device.destroyCommandPool(*pool).has_value());
    }
    CHECK_THAT(errors, Catch::Matchers::IsEmpty());
}

TEST_CASE("Vulkan without VK_KHR_ray_tracing_pipeline reports none, and refuses pipelines and traces",
          "[vulkan][gpu]") {
    const AdapterOverrideLayer adapterOverride{"VK_LAYER_KHRONOS_validation"};
    const ScopedEnv hidden("BFR_ADAPTER_HIDE", "VK_KHR_ray_tracing_pipeline");
    std::vector<std::string> errors;
    {
        auto created = Instance::create(recordingErrorsInto(errors), kValidation);
        INFO(failureOf(created));
        REQUIRE(created.has_value());
        const std::unique_ptr<Instance> instance = std::move(*created);
        const Caps& caps = instance->adapters()[0].caps;
        CHECK_FALSE(caps.features.rayTracingPipeline);
        CHECK_FALSE(caps.features.traceRaysIndirect);
        CHECK_FALSE(caps.features.rayPrimitiveCulling);
        CHECK(caps.rayTracing.identifierSize == 0);
        CHECK(caps.rayTracing.maxRecursionDepth == 0);
        CHECK_FALSE(caps.timestamps.outsideRendering.rayTracing);
        auto made = instance->createDevice(0, kNoWindow, kTestDevice);
        INFO(failureOf(made));
        REQUIRE(made.has_value());
        Device& device = **made;
        const auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        const auto list = device.beginCommands(*pool);
        REQUIRE(list.has_value());

        const auto pipeline = device.createRayTracingPipeline({});
        const auto traced = device.traceRays(*list, {});
        const auto indirect = device.traceRaysIndirect(*list, BufferHandle{}, 0);
        const auto timestamps = device.createTimestampPool({.queue = 0, .count = 1});
        REQUIRE(timestamps.has_value());
        const auto timed = device.writeTimestamp(*list, *timestamps, 0, TimestampStage::RayTracing,
                                                 TimestampGranularity::Relaxed);

        CHECK(pipeline.error().code == bfr::core::ErrorCode::RhiUnsupported);
        for (const auto* result : {&traced, &indirect, &timed}) {
            REQUIRE_FALSE(result->has_value());
            CHECK(result->error().code == bfr::core::ErrorCode::RhiUnsupported);
        }
        CHECK(device.destroyTimestampPool(*timestamps).has_value());
        CHECK(device.destroyCommandList(*list).has_value());
        CHECK(device.destroyCommandPool(*pool).has_value());
    }
    CHECK_THAT(errors, Catch::Matchers::IsEmpty());
}

TEST_CASE("Vulkan without VK_KHR_ray_tracing_position_fetch reports no position fetch", "[vulkan][gpu]") {
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv hidden("BFR_ADAPTER_HIDE", "VK_KHR_ray_tracing_position_fetch");
    auto created = Instance::create(vulkanDesc(), {});
    INFO(failureOf(created));
    REQUIRE(created.has_value());
    const std::unique_ptr<Instance> instance = std::move(*created);
    CHECK_FALSE(instance->adapters()[0].caps.features.rayPositionFetch);
}

TEST_CASE("Vulkan without VK_KHR_pipeline_library reports no pipeline libraries, and refuses them", "[vulkan][gpu]") {
    const AdapterOverrideLayer adapterOverride{"VK_LAYER_KHRONOS_validation"};
    const ScopedEnv hidden("BFR_ADAPTER_HIDE", "VK_KHR_pipeline_library");
    std::vector<std::string> errors;
    {
        auto created = Instance::create(recordingErrorsInto(errors), kValidation);
        INFO(failureOf(created));
        REQUIRE(created.has_value());
        const std::unique_ptr<Instance> instance = std::move(*created);
        const Caps& caps = instance->adapters()[0].caps;
        CHECK_FALSE(caps.features.rayTracingPipelineLibrary);
        CHECK_FALSE(caps.features.pipelineLibraryIdentifiers);
        if (!caps.features.rayTracingPipeline) {
            SKIP("the adapter has no ray tracing pipelines");
        }
        auto made = instance->createDevice(0, kNoWindow, kTestDevice);
        INFO(failureOf(made));
        REQUIRE(made.has_value());
        const auto library = (**made).createRayTracingPipeline({.library = true});
        REQUIRE_FALSE(library.has_value());
        CHECK(library.error().code == bfr::core::ErrorCode::RhiUnsupported);
        const PipelineHandle named{};
        const auto linking = (**made).createRayTracingPipeline({.libraries = std::span{&named, 1}});
        REQUIRE_FALSE(linking.has_value());
        CHECK(linking.error().code == bfr::core::ErrorCode::RhiUnsupported);
    }
    CHECK_THAT(errors, Catch::Matchers::IsEmpty());
}

TEST_CASE("Vulkan without VK_EXT_opacity_micromap reports no micromaps, and refuses them", "[vulkan][gpu]") {
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv hidden("BFR_ADAPTER_HIDE", "VK_EXT_opacity_micromap");
    auto created = Instance::create(vulkanDesc(), {});
    INFO(failureOf(created));
    REQUIRE(created.has_value());
    const std::unique_ptr<Instance> instance = std::move(*created);
    const Caps& caps = instance->adapters()[0].caps;
    CHECK_FALSE(caps.features.opacityMicromap);
    CHECK(caps.rayTracing.maxOpacity2StateSubdivisionLevel == 0);
    CHECK(caps.rayTracing.maxOpacity4StateSubdivisionLevel == 0);
    if (!caps.features.accelerationStructure) {
        SKIP("the adapter has no acceleration structures");
    }
    auto made = instance->createDevice(0, kNoWindow, testDeviceWith([](DeviceDesc& device) {
                                           device.maxAccelerationStructurePropertyPools = 1;
                                       }));
    INFO(failureOf(made));
    REQUIRE(made.has_value());
    Device& device = **made;
    const auto code = [](const auto& result) {
        return result.has_value() ? bfr::core::ErrorCode{} : result.error().code;
    };
    constexpr std::uint64_t kSmall = 256;
    using bfr::core::ErrorCode;
    CHECK(code(device.createAccelerationStructure(
              {.type = AccelerationStructureType::OpacityMicromap, .size = kSmall})) == ErrorCode::RhiUnsupported);
    CHECK(code(device.accelerationStructureSizes(
              {.type = AccelerationStructureType::Bottom, .flags = {.allowDisableOpacityMicromaps = true}})) ==
          ErrorCode::RhiUnsupported);
    CHECK(code(device.createAccelerationStructurePropertyPool({.count = 1, .opacityMicromaps = true})) ==
          ErrorCode::RhiUnsupported);
    std::vector<std::byte> out(caps.rayTracing.instanceSize);
    const AccelerationStructureInstance disabled{.disableOpacityMicromaps = true};
    CHECK(code(device.writeInstances(std::span{&disabled, 1}, out)) == ErrorCode::RhiUnsupported);
    if (caps.features.rayTracingPipeline) {
        CHECK(code(device.createRayTracingPipeline({.allowOpacityMicromaps = true})) == ErrorCode::RhiUnsupported);
    }
}

TEST_CASE("Vulkan without VK_NV_ray_tracing_motion_blur reports no motion", "[vulkan][gpu]") {
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv hidden("BFR_ADAPTER_HIDE", "VK_NV_ray_tracing_motion_blur");
    auto created = Instance::create(vulkanDesc(), {});
    INFO(failureOf(created));
    REQUIRE(created.has_value());
    const std::unique_ptr<Instance> instance = std::move(*created);
    const Caps& caps = instance->adapters()[0].caps;
    CHECK_FALSE(caps.features.rayTracingMotion);
    CHECK_FALSE(caps.features.rayTracingMotionIndirect);
    CHECK(caps.rayTracing.maxMotionKeyframes == 0);
    CHECK(caps.rayTracing.motionInstanceSize == 0);
}

TEST_CASE("Vulkan refuses motion of other than the two keyframes it takes", "[vulkan][gpu]") {
    auto created = Instance::create(vulkanDesc(), {});
    INFO(failureOf(created));
    REQUIRE(created.has_value());
    const std::unique_ptr<Instance> instance = std::move(*created);
    const Caps& caps = instance->adapters()[0].caps;
    if (!caps.features.rayTracingMotion) {
        SKIP("the adapter has no structures with motion");
    }
    CHECK(caps.rayTracing.maxMotionKeyframes == 2);
    auto made = instance->createDevice(0, kNoWindow, kTestDevice);
    INFO(failureOf(made));
    REQUIRE(made.has_value());
    Device& device = **made;
    constexpr std::uint64_t kSmall = 256;
    constexpr std::uint32_t kThree = 3;
    const auto structure = device.createAccelerationStructure({.size = kSmall, .motionKeyframes = kThree});
    REQUIRE_FALSE(structure.has_value());
    CHECK(structure.error().code == bfr::core::ErrorCode::RhiInvalidCall);
    const std::array<MotionKeyframe, kThree> keyframes{};
    const AccelerationStructureMotionInstance moving{.motion = InstanceMotion::Matrix, .keyframeCount = kThree};
    std::vector<std::byte> out(caps.rayTracing.motionInstanceSize);
    const auto written = device.writeMotionInstances(std::span{&moving, 1}, keyframes, out);
    REQUIRE_FALSE(written.has_value());
    CHECK(written.error().code == bfr::core::ErrorCode::RhiInvalidCall);
}

TEST_CASE("Vulkan refuses opacity micromap work it has no call for", "[vulkan][gpu]") {
    auto created = Instance::create(vulkanDesc(), {});
    INFO(failureOf(created));
    REQUIRE(created.has_value());
    const std::unique_ptr<Instance> instance = std::move(*created);
    if (!instance->adapters()[0].caps.features.opacityMicromap) {
        SKIP("the adapter has no opacity micromaps");
    }
    auto made = instance->createDevice(0, kNoWindow, testDeviceWith([](DeviceDesc& device) {
                                           device.maxAccelerationStructurePropertyPools = 1;
                                       }));
    INFO(failureOf(made));
    REQUIRE(made.has_value());
    Device& device = **made;
    const auto code = [](const auto& result) {
        return result.has_value() ? bfr::core::ErrorCode{} : result.error().code;
    };
    constexpr std::uint64_t kSmall = 256;
    const auto micromap = device.createAccelerationStructure(
        {.type = AccelerationStructureType::OpacityMicromap, .size = kSmall});
    INFO(failureOf(micromap));
    REQUIRE(micromap.has_value());
    // Vulkan gives a micromap no address, writes only its compacted and
    // serialized sizes, and builds it whole.
    CHECK(code(device.accelerationStructureReference(*micromap)) == bfr::core::ErrorCode::RhiInvalidCall);
    if (instance->adapters()[0].caps.features.accelerationStructureCurrentSize) {
        CHECK(code(device.createAccelerationStructurePropertyPool(
                  {.property = AccelerationStructureProperty::CurrentSize, .count = 1, .opacityMicromaps = true})) ==
              bfr::core::ErrorCode::RhiInvalidCall);
    }
    const auto pool = device.createCommandPool({.queue = 0, .listCount = 1});
    REQUIRE(pool.has_value());
    const auto list = device.beginCommands(*pool);
    REQUIRE(list.has_value());
    CHECK(code(device.buildAccelerationStructure(*list,
                                                 {.dst = *micromap,
                                                  .src = *micromap,
                                                  .update = true,
                                                  .inputs = {.type = AccelerationStructureType::OpacityMicromap}})) ==
          bfr::core::ErrorCode::RhiInvalidCall);
    CHECK(device.destroyCommandList(*list).has_value());
    CHECK(device.destroyCommandPool(*pool).has_value());
    CHECK(device.destroyAccelerationStructure(*micromap).has_value());
}

TEST_CASE("Vulkan refuses a ray tracing stack size past the 32 bits it takes", "[vulkan][gpu]") {
    auto created = Instance::create(vulkanDesc(), {});
    INFO(failureOf(created));
    REQUIRE(created.has_value());
    const std::unique_ptr<Instance> instance = std::move(*created);
    if (!instance->adapters()[0].caps.features.rayTracingPipeline) {
        SKIP("the adapter has no ray tracing pipelines");
    }
    auto made = instance->createDevice(0, kNoWindow, kTestDevice);
    INFO(failureOf(made));
    REQUIRE(made.has_value());
    constexpr std::uint64_t kPast32Bits = std::uint64_t{1} << 32U;
    const auto pipeline = (**made).createRayTracingPipeline({.stackSize = kPast32Bits});
    REQUIRE_FALSE(pipeline.has_value());
    CHECK(pipeline.error().code == bfr::core::ErrorCode::RhiInvalidCall);
}

TEST_CASE("Vulkan with acceleration structures but no table updated after bind has no table, and makes top structures "
          "with no slot",
          "[vulkan][gpu]") {
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv hidden("BFR_ADAPTER_NO_FEATURE", "descriptorBindingAccelerationStructureUpdateAfterBind");
    auto created = Instance::create(vulkanDesc(), {});
    INFO(failureOf(created));
    REQUIRE(created.has_value());
    const std::unique_ptr<Instance> instance = std::move(*created);
    const Caps& caps = instance->adapters()[0].caps;
    if (!caps.features.accelerationStructure) {
        SKIP("the adapter has no acceleration structures");
    }
    CHECK(caps.bindless.maxAccelerationStructures == 0);
    const auto tabled = instance->createDevice(0, kNoWindow, testDeviceWith([](DeviceDesc& desc) {
                                                   desc.maxBindlessAccelerationStructures = 1;
                                               }));
    REQUIRE_FALSE(tabled.has_value());
    CHECK(tabled.error().code == bfr::core::ErrorCode::RhiLimitReached);
    auto made = instance->createDevice(0, kNoWindow, kTestDevice);
    INFO(failureOf(made));
    REQUIRE(made.has_value());
    Device& device = **made;
    // A top structure is made with no slot.
    constexpr std::uint64_t kSmallStructure = 256;
    const auto top = device.createAccelerationStructure(
        {.type = AccelerationStructureType::Top, .size = kSmallStructure});
    INFO(failureOf(top));
    REQUIRE(top.has_value());
    const auto slot = device.accelerationStructureSlot(*top);
    REQUIRE_FALSE(slot.has_value());
    CHECK(slot.error().code == bfr::core::ErrorCode::RhiInvalidCall);
    CHECK(device.destroyAccelerationStructure(*top).has_value());
}

TEST_CASE("Vulkan with mesh shaders but no task shaders refuses a task pipeline and a task timestamp",
          "[vulkan][gpu]") {
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv hidden("BFR_ADAPTER_NO_FEATURE", "taskShader");
    auto created = Instance::create(vulkanDesc(), {});
    INFO(failureOf(created));
    REQUIRE(created.has_value());
    const std::unique_ptr<Instance> instance = std::move(*created);
    const Caps& caps = instance->adapters()[0].caps;
    if (!caps.features.meshShader) {
        SKIP("the adapter has no mesh shaders");
    }
    CHECK_FALSE(caps.features.taskShader);
    CHECK(caps.mesh.maxTaskWorkgroupTotalCount == 0);
    auto made = instance->createDevice(0, kNoWindow, kTestDevice);
    INFO(failureOf(made));
    REQUIRE(made.has_value());
    Device& device = **made;
    GraphicsPipelineDesc taskDesc{
        .vertex = {},
        .fragment = {.code = std::as_bytes(std::span{bfr::test::spirv::kColorFrag}), .entryPoint = "main"},
        .task = {.code = std::as_bytes(std::span{bfr::test::spirv::kTriangleTask}), .entryPoint = "main"},
        .taskWorkgroupSize = {1, 1, 1},
        .mesh = {.code = std::as_bytes(std::span{bfr::test::spirv::kTriangleMesh}), .entryPoint = "main"},
        .meshWorkgroupSize = {1, 1, 1},
        .topology = Topology::TriangleList,
        .primitiveRestart = PrimitiveRestart::Off,
        .raster = {.cull = CullMode::None},
        .depthStencil = {},
        .colorTargetCount = 1,
        .samples = 1,
        .sampleMask = BFR_MASK_ALL};
    taskDesc.colorTargets[0] = {.format = Format::Rgba8Unorm, .blend = {.writeMask = BFR_MASK_ALL}};

    const auto pipeline = device.createGraphicsPipeline(taskDesc);

    REQUIRE_FALSE(pipeline.has_value());
    CHECK(pipeline.error().code == bfr::core::ErrorCode::RhiUnsupported);
    const auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
    REQUIRE(pool.has_value());
    const auto list = device.beginCommands(*pool);
    REQUIRE(list.has_value());
    const auto timestamps = device.createTimestampPool({.queue = 0, .count = 1});
    REQUIRE(timestamps.has_value());
    const auto timed = device.writeTimestamp(*list, *timestamps, 0, TimestampStage::Task,
                                             TimestampGranularity::Relaxed);
    REQUIRE_FALSE(timed.has_value());
    CHECK(timed.error().code == bfr::core::ErrorCode::RhiUnsupported);
    CHECK(device.destroyTimestampPool(*timestamps).has_value());
    CHECK(device.destroyCommandList(*list).has_value());
    CHECK(device.destroyCommandPool(*pool).has_value());
}

TEST_CASE("Vulkan at 1.2 without VK_EXT_extended_dynamic_state reports no dynamic topology and bakes it in",
          "[vulkan][gpu]") {
    const AdapterOverrideLayer adapterOverride{"VK_LAYER_KHRONOS_validation"};
    const ScopedEnv hidden("BFR_ADAPTER_HIDE", "VK_EXT_extended_dynamic_state");
    std::vector<std::string> errors;
    {
        InstanceDesc desc = recordingErrorsInto(errors);
        desc.apiMajor = 1;
        desc.apiMinor = 2;
        auto created = Instance::create(desc, kValidation);
        INFO(failureOf(created));
        REQUIRE(created.has_value());
        const std::unique_ptr<Instance> instance = std::move(*created);
        CHECK_FALSE(instance->adapters()[0].caps.features.dynamicTopology);
        auto made = instance->createDevice(0, kNoWindow, kTestDevice);
        INFO(failureOf(made));
        REQUIRE(made.has_value());
        Device& device = **made;
        GraphicsPipelineDesc pipelineDesc{.vertex = {.code = std::as_bytes(std::span{bfr::test::spirv::kBindlessVert}),
                                                     .entryPoint = "main"},
                                          .fragment = {.code = std::as_bytes(std::span{bfr::test::spirv::kColorFrag}),
                                                       .entryPoint = "main"},
                                          .topology = Topology::TriangleStrip,
                                          .primitiveRestart = PrimitiveRestart::Off,
                                          .raster = {.cull = CullMode::None},
                                          .depthStencil = {},
                                          .colorTargetCount = 1,
                                          .samples = 1,
                                          .sampleMask = BFR_MASK_ALL};
        pipelineDesc.colorTargets[0] = {.format = Format::Rgba8Unorm, .blend = {.writeMask = BFR_MASK_ALL}};
        const auto pipeline = device.createGraphicsPipeline(pipelineDesc);
        INFO(failureOf(pipeline));
        REQUIRE(pipeline.has_value());
        const auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        REQUIRE(pool.has_value());
        const auto list = device.beginCommands(*pool);
        REQUIRE(list.has_value());

        CHECK(device.bindPipeline(*list, *pipeline).has_value());
        const auto set = device.setTopology(*list, Topology::TriangleList);
        REQUIRE_FALSE(set.has_value());
        CHECK(set.error().code == bfr::core::ErrorCode::RhiUnsupported);

        CHECK(device.destroyCommandList(*list).has_value());
        CHECK(device.destroyCommandPool(*pool).has_value());
        CHECK(device.destroyPipeline(*pipeline).has_value());
    }
    CHECK_THAT(errors, Catch::Matchers::IsEmpty());
}

TEST_CASE("Vulkan lists a 1.3 adapter without an extension that 1.3 made core", "[vulkan][gpu]") {
    const std::string extension = GENERATE("VK_KHR_dynamic_rendering", "VK_KHR_synchronization2");
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv version("BFR_ADAPTER_MINOR", "3");
    const ScopedEnv hidden("BFR_ADAPTER_HIDE", extension);

    const auto rhi = Instance::create(vulkanDesc(), {});

    INFO(extension << ' ' << failureOf(rhi));
    REQUIRE(rhi.has_value());
    CHECK_FALSE((*rhi)->adapters().empty());
}

TEST_CASE("Vulkan lists an adapter without the swapchain extension and its device can't present", "[vulkan][gpu]") {
    const AdapterOverrideLayer adapterOverride{"VK_LAYER_KHRONOS_validation"};
    const ScopedEnv hidden("BFR_ADAPTER_HIDE", "VK_KHR_swapchain");
    std::vector<std::string> errors;
    const auto rhi = Instance::create(recordingErrorsInto(errors), kValidation);
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());

    const auto device = (*rhi)->createDevice(cpuAdapter(**rhi), kNoWindow, kTestDevice);
    INFO(failureOf(device));
    REQUIRE(device.has_value());
    const auto swapchain = (*device)->createSwapchain(bfr::core::window::NativeWindow{bfr::core::window::Headless{}},
                                                      kTestSwapchain);

    REQUIRE_FALSE(swapchain.has_value());
    CHECK(swapchain.error().code == bfr::core::ErrorCode::RhiPresentUnsupported);
    CHECK_THAT(errors, Catch::Matchers::IsEmpty());
}

TEST_CASE("Vulkan lists an adapter with no graphics family and opens its compute queues", "[vulkan][gpu]") {
    const AdapterOverrideLayer adapterOverride{"VK_LAYER_KHRONOS_validation"};
    const ScopedEnv noGraphics("BFR_ADAPTER_NO_GRAPHICS", "1");
    std::vector<std::string> errors;
    const auto rhi = Instance::create(recordingErrorsInto(errors), kValidation);
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());
    const std::size_t adapter = cpuAdapter(**rhi);
    CHECK((*rhi)->adapters()[adapter].caps.queues.graphics == 0);

    DeviceDesc computeOnly = kTestDevice;
    computeOnly.graphics = {};
    computeOnly.compute = std::span{kTestQueues}.first(1);
    const auto device = (*rhi)->createDevice(adapter, kNoWindow, computeOnly);
    INFO(failureOf(device));
    CHECK(device.has_value());
    CHECK(((*rhi)->createDevice(adapter, kNoWindow, kTestDevice).error().code == bfr::core::ErrorCode::RhiUnsupported));
    CHECK_THAT(errors, Catch::Matchers::IsEmpty());
}

TEST_CASE("Vulkan fails with RhiUnsupported for one kind's queues at different levels", "[vulkan][gpu]") {
    std::vector<std::string> errors;
    const auto rhi = Instance::create(recordingErrorsInto(errors), kValidation);
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());
    const auto adapters = (*rhi)->adapters();
    const auto found = std::ranges::find_if(adapters,
                                            [](const Adapter& adapter) { return adapter.caps.queues.graphics >= 2; });
    if (found == adapters.end()) {
        SKIP("no adapter has two graphics queues");
    }
    constexpr std::array kMixed{QueueDesc{.rank = 0, .level = QueueLevel::Normal},
                                QueueDesc{.rank = 0, .level = QueueLevel::Low}};
    DeviceDesc mixed = kTestDevice;
    mixed.graphics = kMixed;

    const auto device = (*rhi)->createDevice(static_cast<std::size_t>(found - adapters.begin()), kNoWindow, mixed);

    REQUIRE_FALSE(device.has_value());
    CHECK(device.error().code == bfr::core::ErrorCode::RhiUnsupported);
    // Without VK_KHR_global_priority, the low level is refused before the mix is.
    const auto& listed = found->caps.queues.graphicsLevels;
    if (listed.normal || listed.low || listed.high || listed.realtime) {
        CHECK_THAT(device.error().message.text(), Catch::Matchers::ContainsSubstring("one level"));
    }
    CHECK_THAT(errors, Catch::Matchers::IsEmpty());
}

TEST_CASE("Vulkan fails with RhiPresentUnsupported when no queue family presents to the window", "[vulkan][gpu]") {
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv unsupported("BFR_PRESENT_UNSUPPORTED", "1");
    const auto rhi = Instance::create(vulkanDesc(), {});
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());

    const auto device = (*rhi)->createDevice(cpuAdapter(**rhi), bfr::core::window::Headless{}, kTestDevice);

    REQUIRE_FALSE(device.has_value());
    CHECK(device.error().code == bfr::core::ErrorCode::RhiPresentUnsupported);
    CHECK_THAT(device.error().message.text(), Catch::Matchers::ContainsSubstring("presents to the window"));
}

TEST_CASE("Vulkan fails with RhiWindowFailed when the API refuses a surface for the device's window", "[vulkan][gpu]") {
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv refused("BFR_FAIL", "vkCreateHeadlessSurfaceEXT");
    const auto rhi = Instance::create(vulkanDesc(), {});
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());

    const auto device = (*rhi)->createDevice(0, bfr::core::window::Headless{}, kTestDevice);

    REQUIRE_FALSE(device.has_value());
    CHECK(device.error().code == bfr::core::ErrorCode::RhiWindowFailed);
    CHECK_THAT(device.error().message.text(),
               Catch::Matchers::ContainsSubstring("vkCreateHeadlessSurfaceEXT: VK_ERROR_UNKNOWN"));
}

// In these two, lavapipe reports NVIDIA's driver ID and stands in for NVIDIA's
// driver, which has no headless surfaces and crashes when a loader before
// 1.4.319 asks it about one.
TEST_CASE("Vulkan doesn't ask NVIDIA's driver about a headless window on a loader that lets it crash",
          "[vulkan][gpu]") {
    const LoaderVersion loader{"1.4.318"};
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv nvidia("BFR_ADAPTER_DRIVER_ID", std::to_string(VK_DRIVER_ID_NVIDIA_PROPRIETARY));
    std::ostringstream log;
    const auto rhi = Instance::create(vulkanLoggingInto(log), {});
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());

    const auto device = (*rhi)->createDevice(cpuAdapter(**rhi), bfr::core::window::Headless{}, kTestDevice);

    REQUIRE_FALSE(device.has_value());
    CHECK(device.error().code == bfr::core::ErrorCode::RhiPresentUnsupported);
    if (bfr::core::kLoggingEnabled) {
        CHECK_THAT(log.str(), Catch::Matchers::ContainsSubstring(
                                  "does not present to headless surfaces; not asked, since loader 1.4.318"));
    }
}

TEST_CASE("Vulkan asks NVIDIA's driver about a headless window on a loader that answers for it", "[vulkan][gpu]") {
    const LoaderVersion loader{"1.4.319"};
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv nvidia("BFR_ADAPTER_DRIVER_ID", std::to_string(VK_DRIVER_ID_NVIDIA_PROPRIETARY));
    const auto rhi = Instance::create(vulkanDesc(), {});
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());

    const auto device = (*rhi)->createDevice(cpuAdapter(**rhi), bfr::core::window::Headless{}, kTestDevice);

    INFO(failureOf(device));
    CHECK(device.has_value());
}

TEST_CASE("Vulkan fails with RhiDeviceFailed when the API refuses the device and leaves nothing behind",
          "[vulkan][gpu]") {
    const AdapterOverrideLayer adapterOverride{"VK_LAYER_KHRONOS_validation"};
    const ScopedEnv failing("BFR_FAIL", "vkCreateDevice");
    std::vector<std::string> errors;
    {
        const auto rhi = Instance::create(recordingErrorsInto(errors), kValidation);
        INFO(failureOf(rhi));
        REQUIRE(rhi.has_value());

        const auto device = (*rhi)->createDevice(0, kNoWindow, kTestDevice);

        REQUIRE_FALSE(device.has_value());
        CHECK(device.error().code == bfr::core::ErrorCode::RhiDeviceFailed);
        CHECK_THAT(device.error().message.text(),
                   Catch::Matchers::ContainsSubstring("vkCreateDevice on " + (*rhi)->adapters()[0].name +
                                                      ": VK_ERROR_UNKNOWN"));
    }
    // The loader reports the refusal; anything more would be the RHI misusing
    // what the failed call left.
    CHECK_THAT(errors, Catch::Matchers::AllMatch(Catch::Matchers::ContainsSubstring("Failed to create device chain")));
}

TEST_CASE("Vulkan creates a valid device on an adapter that lists VK_KHR_portability_subset", "[vulkan][gpu]") {
    const AdapterOverrideLayer adapterOverride{"VK_LAYER_KHRONOS_validation"};
    const ScopedEnv added("BFR_ADAPTER_ADD", "VK_KHR_portability_subset");
    // Errors only: the loader also reports layer warnings.
    std::vector<std::string> errors;
    const auto rhi = Instance::create(recordingErrorsInto(errors), kValidation);
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());

    const auto device = (*rhi)->createDevice(0, kNoWindow, kTestDevice);

    INFO(failureOf(device));
    REQUIRE(device.has_value());
    CHECK_THAT(errors, Catch::Matchers::IsEmpty());
}

TEST_CASE("Vulkan logs the loader and each adapter it lists", "[vulkan][gpu]") {
    if (!bfr::core::kLoggingEnabled) {
        SKIP("logging is compiled out");
    }
    std::ostringstream log;

    const auto rhi = Instance::create(vulkanLoggingInto(log), {});

    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());
    const bfr::test::ApiVersion version = bfr::test::testApiVersion("vulkan");
    CHECK_THAT(log.str(), Catch::Matchers::ContainsSubstring("Vulkan loader 1.") &&
                              Catch::Matchers::ContainsSubstring(
                                  std::format("Vulkan instance {}.{}, validation off", version.major, version.minor)));
    for (const Adapter& adapter : (*rhi)->adapters()) {
        CHECK_THAT(lineStartingWith(log.str(), "adapter " + adapter.name + ", "),
                   Catch::Matchers::EndsWith(": listed"));
    }
}

TEST_CASE("Vulkan logs why an adapter is below BFR's minimum", "[vulkan][gpu]") {
    if (!bfr::core::kLoggingEnabled) {
        SKIP("logging is compiled out");
    }
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv version("BFR_ADAPTER_MINOR", "1");
    std::ostringstream log;

    const auto rhi = Instance::create(vulkanLoggingInto(log), {});

    REQUIRE_FALSE(rhi.has_value());
    CHECK_THAT(log.str(),
               Catch::Matchers::ContainsSubstring(", Vulkan 1.1: below BFR's minimum, Vulkan 1.1, needs 1.2"));
}

TEST_CASE("Vulkan logs the validation layer's version", "[vulkan][gpu]") {
    if (!bfr::core::kLoggingEnabled) {
        SKIP("logging is compiled out");
    }
    std::ostringstream log;

    const auto rhi = Instance::create(vulkanLoggingInto(log), kValidation);

    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());
    CHECK_THAT(log.str(), Catch::Matchers::ContainsSubstring("VK_LAYER_KHRONOS_validation 1."));
}

TEST_CASE("Vulkan at 1.2 enables each extension 1.3 made core that the adapter lists, and at 1.3 none",
          "[vulkan][gpu]") {
    if (!bfr::core::kLoggingEnabled) {
        SKIP("logging is compiled out");
    }
    // One with a features struct, one without.
    const std::string extension = GENERATE("VK_EXT_shader_demote_to_helper_invocation", "VK_EXT_tooling_info");
    const std::uint32_t minor = GENERATE(2U, 3U);
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv added("BFR_ADAPTER_ADD", extension);
    std::ostringstream log;
    InstanceDesc desc = vulkanLoggingInto(log);
    desc.apiMinor = minor;
    const auto rhi = Instance::create(desc, {});
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());

    const auto device = (*rhi)->createDevice(cpuAdapter(**rhi), kNoWindow, kTestDevice);

    INFO(extension << " at 1." << minor << ' ' << failureOf(device));
    REQUIRE(device.has_value());
    const std::string line = lineStartingWith(log.str(), "device on ");
    if (minor == 2) {
        CHECK_THAT(line, Catch::Matchers::ContainsSubstring(extension));
    }
    else {
        CHECK_THAT(line, !Catch::Matchers::ContainsSubstring(extension));
    }
}

TEST_CASE("Vulkan logs the device it creates, its queue families and its extensions", "[vulkan][gpu]") {
    if (!bfr::core::kLoggingEnabled) {
        SKIP("logging is compiled out");
    }
    const AdapterOverrideLayer adapterOverride;
    const ScopedEnv added("BFR_ADAPTER_ADD", "VK_KHR_portability_subset");
    // Outlives the Instance, which logs as it goes.
    std::ostringstream log;
    const auto rhi = Instance::create(vulkanLoggingInto(log), {});
    INFO(failureOf(rhi));
    REQUIRE(rhi.has_value());

    // lavapipe has one queue family.
    const auto device = (*rhi)->createDevice(cpuAdapter(**rhi), kNoWindow, kTestDevice);

    INFO(failureOf(device));
    REQUIRE(device.has_value());
    CHECK_THAT(log.str(), Catch::Matchers::ContainsSubstring("device on " + (*rhi)->adapters()[cpuAdapter(**rhi)].name +
                                                             ", queue families 0, extensions: VK_KHR_swapchain, ") &&
                              Catch::Matchers::ContainsSubstring("VK_KHR_portability_subset"));
}
