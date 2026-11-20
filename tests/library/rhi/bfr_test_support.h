#pragma once

// Helpers for the library tests that speak only to the public API, bfr.hpp,
// and run on every backend the build has. A backend's own helpers sit under
// backends/<name>.

#include "bfr.hpp"
#include "bfr/bfr.h"
#include "test_backends.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace bfr::test {

// Lists each test command pool holds, more than any test records between resets.
inline constexpr std::uint32_t kPoolLists = 8;

/// Validation on, for createInstance.
inline constexpr bool kValidation = true;
/// Validation off, for createInstance.
inline constexpr bool kValidationOff = false;

/// The adapter a test that needs no particular one opens its device on.
inline constexpr std::uint32_t kFirstAdapter = 0;

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
// The queue families a test's queues may be of, more than any adapter the
// tests run on has; a family past them fails the test.
inline constexpr std::uint32_t kTestFamilies = 16;
// The most queues of one family a test opens.
inline constexpr std::size_t kTestFamilyQueues = 4;
// For each family, queues of it at rank 0 and level normal, as many as a test
// opens: storage that outlives every DeviceDesc naming it.
inline constexpr auto kTestQueues = [] {
    std::array<std::array<QueueDesc, kTestFamilyQueues>, kTestFamilies> queues{};
    for (std::uint32_t family = 0; family < kTestFamilies; ++family) {
        for (QueueDesc& queue : queues.at(family)) {
            queue.family = family;
        }
    }
    return queues;
}();
// The longest wait there is, so a slow GPU never fails a test.
inline constexpr std::uint64_t kTestWait = std::numeric_limits<std::uint64_t>::max();

/// No window: testDeviceOn asks nothing of presenting.
inline constexpr Window kNoWindow{};
inline constexpr Window kHeadless{.system = WindowSystem::Headless};

/// What the tests' devices hold; testDeviceOn and testDeviceWith name the
/// adapter and the queue.
inline constexpr DeviceDesc kTestDevice{.maxCommandLists = kTestLists,
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

/// The code and message of a failed result, for INFO; empty on success.
template <class T>
std::string failureOf(const Expected<T>& result) {
    return result.has_value() ? std::string{}
                              : std::string{bfr::name(result.error().code)} + ": " + result.error().message.text();
}

/// count queues of family, at rank 0 and level normal.
inline std::span<const QueueDesc> testQueues(std::uint32_t family, std::size_t count = 1) {
    INFO("family " << family << " is past the tests' queue storage");
    REQUIRE(family < kTestFamilies);
    return std::span{kTestQueues.at(family)}.first(count);
}

/// The first family of adapter that runs graphics, compute and copies and,
/// unless window is kNoWindow, presents to window; the test fails when there
/// is none.
inline std::uint32_t testFamily(const Instance& instance, std::uint32_t adapter, const Window& window) {
    const bool none = window.window == nullptr && window.display == nullptr && window.system == WindowSystem{};
    const auto family = instance.findQueueFamily(
        adapter, {.graphics = true, .compute = true, .copy = true, .presents = none ? nullptr : &window});
    INFO(failureOf(family));
    REQUIRE(family.has_value());
    return *family;
}

/// The queue families adapter of instance lists; the test fails when it can't
/// list them.
inline std::vector<QueueFamily> queueFamiliesOf(const Instance& instance, std::uint32_t adapter) {
    auto families = instance.queueFamilies(adapter);
    INFO(failureOf(families));
    REQUIRE(families.has_value());
    return std::move(*families);
}

/// The first family of adapter for which wanted is true, if any.
template <class Wanted>
std::optional<std::uint32_t> familyWhere(const Instance& instance, std::uint32_t adapter, Wanted wanted) {
    const std::vector<QueueFamily> families = queueFamiliesOf(instance, adapter);
    const auto found = std::ranges::find_if(families, wanted);
    if (found == families.end()) {
        return std::nullopt;
    }
    return static_cast<std::uint32_t>(found - families.begin());
}

/// What testFamily's family on adapter is, presenting nowhere.
inline QueueFamily testQueueFamily(const Instance& instance, std::uint32_t adapter) {
    return queueFamiliesOf(instance, adapter).at(testFamily(instance, adapter, kNoWindow));
}

/// kTestDevice on adapter, with one queue of testFamily's, and swapchains
/// unless window is kNoWindow.
inline DeviceDesc testDeviceOn(const Instance& instance, std::uint32_t adapter, const Window& window) {
    DeviceDesc desc = kTestDevice;
    desc.adapter = adapter;
    desc.queues = testQueues(testFamily(instance, adapter, window));
    desc.features.swapchain = window.window != nullptr || window.display != nullptr || window.system != WindowSystem{};
    return desc;
}

/// testDeviceOn with change made to it.
template <class Change>
DeviceDesc testDeviceWith(const Instance& instance, std::uint32_t adapter, const Window& window, Change change) {
    DeviceDesc desc = testDeviceOn(instance, adapter, window);
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

/// The code of a failed bfr.h call and the calling thread's last failure
/// message, for INFO straight after that call; empty when result is no error.
inline std::string failureOf(Result result) {
    if (std::to_underlying(result) >= 0) {
        return {};
    }
    const bfr_string message = bfr_get_last_error();
    return std::string{bfr::name(result)} + ": " + std::string{std::string_view{message.data, message.length}};
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
        // Unsupported: the backend can't be asked, as on a Vulkan loader that
        // lets NVIDIA's driver crash on a headless window.
        REQUIRE(
            (caps.error().code == Result::ErrorPresentUnsupported || caps.error().code == Result::ErrorUnsupported));
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
    auto device = instance.createDevice(testDeviceOn(instance, presentingAdapter(instance, window), window));
    INFO(failureOf(device));
    REQUIRE(device.has_value());
    return std::move(*device);
}

/// An Instance as desc says, with validation on or off, and a device that
/// presents to a headless window. The device goes before its Instance.
struct HeadlessDevice {
    explicit HeadlessDevice(const InstanceDesc& desc, bool validation = kValidation) {
        auto created = createInstance(desc, validation);
        INFO(failureOf(created));
        REQUIRE(created.has_value());
        instance = std::move(*created);
        device = presentingDevice(instance, kHeadless);
    }

    Instance instance;
    Device device;
};

/// An Instance as desc says, with validation on, and the tests' device on the
/// first adapter, presenting nowhere, with change made to its desc; a change
/// that takes the adapter's caps too, to ask for the features the test uses
/// that the adapter has, is given them. The device goes before its Instance.
struct FirstAdapterDevice {
    explicit FirstAdapterDevice(const InstanceDesc& desc)
        : FirstAdapterDevice(desc, [](DeviceDesc& /*unchanged*/) {}) {}

    template <class Change>
    FirstAdapterDevice(const InstanceDesc& desc, Change change) {
        auto created = createInstance(desc, kValidation);
        INFO(failureOf(created));
        REQUIRE(created.has_value());
        instance = std::move(*created);
        auto made = instance.createDevice(testDeviceWith(instance, kFirstAdapter, kNoWindow, [&](DeviceDesc& wanted) {
            if constexpr (std::is_invocable_v<Change, DeviceDesc&, const Caps&>) {
                const auto caps = instance.caps(kFirstAdapter);
                INFO(failureOf(caps));
                REQUIRE(caps.has_value());
                change(wanted, *caps);
            }
            else {
                change(wanted);
            }
        }));
        INFO(failureOf(made));
        REQUIRE(made.has_value());
        device = std::move(*made);
    }

    Instance instance;
    Device device;
};

/// What adapter of instance reports; the test fails when the instance can't say.
inline Caps capsOf(const Instance& instance, std::uint32_t adapter) {
    auto caps = instance.caps(adapter);
    INFO(failureOf(caps));
    REQUIRE(caps.has_value());
    return std::move(*caps);
}

/// The ray tracing features, buffer addresses with them, each where caps
/// offer it: what a device for the ray tracing tests asks for.
inline Features rayTracingFeaturesOf(const Caps& caps) {
    const CapsFeatures& offered = caps.features;
    return {.rayQuery = offered.rayQuery,
            .rayQueryGraphics = offered.rayQueryGraphics,
            .rayTracingPipeline = offered.rayTracingPipeline,
            .traceRaysIndirect = offered.traceRaysIndirect,
            .rayPrimitiveCulling = offered.rayPrimitiveCulling,
            .accelerationStructure = offered.accelerationStructure,
            .bufferAddress = offered.bufferAddress,
            .rayPositionFetch = offered.rayPositionFetch,
            .accelerationStructureCurrentSize = offered.accelerationStructureCurrentSize,
            .serializedBottomCount = offered.serializedBottomCount,
            .rayInvocationReorder = offered.rayInvocationReorder,
            .rayTracingPipelineLibrary = offered.rayTracingPipelineLibrary,
            .pipelineLibraryIdentifiers = offered.pipelineLibraryIdentifiers,
            .rayTracingCaptureReplay = offered.rayTracingCaptureReplay,
            .rayTracingCaptureReplayMixed = offered.rayTracingCaptureReplayMixed,
            .accelerationStructureIndirectBuild = offered.accelerationStructureIndirectBuild,
            .opacityMicromap = offered.opacityMicromap,
            .rayTracingMotion = offered.rayTracingMotion,
            .rayTracingMotionIndirect = offered.rayTracingMotionIndirect,
            .partitionedTop = offered.partitionedTop,
            .clusterAccelerationStructure = offered.clusterAccelerationStructure,
            .accelerationStructureCaptureReplay = offered.accelerationStructureCaptureReplay};
}

/// The first adapter of instance whose caps wanted takes, if any.
template <class Wanted>
std::optional<std::uint32_t> adapterWhere(const Instance& instance, Wanted wanted) {
    const auto count = static_cast<std::uint32_t>(adaptersOf(instance).size());
    for (std::uint32_t adapter = 0; adapter < count; ++adapter) {
        if (wanted(capsOf(instance, adapter))) {
            return adapter;
        }
    }
    return std::nullopt;
}

/// What adapter does with format, from the instance's listing of every
/// format; the listing's failure when it fails. The test fails when the
/// listing leaves format out.
inline Expected<FormatCaps> formatCapsOf(const Instance& instance, std::uint32_t adapter, Format format) {
    auto formats = instance.formats(adapter);
    if (!formats.has_value()) {
        return std::unexpected(formats.error());
    }
    const auto found = std::ranges::find(*formats, format, &FormatCaps::format);
    REQUIRE(found != formats->end());
    return *found;
}

/// What a test swapchain asks for, but its image count: a headless window,
/// and a format and color space every headless surface the tests run on takes.
inline constexpr SwapchainDesc kTestSwapchain{.window = kHeadless,
                                              .format = Format::Bgra8Unorm,
                                              .colorSpace = ColorSpace::Srgb};

/// kTestSwapchain with the fewest images the headless surface of instance's
/// first presenting adapter reports it takes, as a caller must set.
inline SwapchainDesc swapchainDesc(const Instance& instance) {
    const auto caps = instance.swapchainCaps(headlessAdapter(instance), {.window = kHeadless});
    INFO(failureOf(caps));
    REQUIRE(caps.has_value());
    SwapchainDesc desc = kTestSwapchain;
    desc.imageCount = caps->minImageCount;
    return desc;
}

/// Whether backend takes the SPIR-V checked in: Vulkan does. A case that
/// needs code skips elsewhere.
inline bool takesSpirv(const TestBackend& backend) {
    return backend.backend == Backend::Vulkan;
}

/// A buffer on device as desc says; the test fails when it can't be made.
inline Buffer created(const Device& device, const BufferDesc& desc) {
    auto buffer = device.createBuffer(desc);
    INFO(failureOf(buffer));
    REQUIRE(buffer.has_value());
    return std::move(*buffer);
}

/// A texture on device as desc says; the test fails when it can't be made.
inline Texture created(const Device& device, const TextureDesc& desc) {
    auto texture = device.createTexture(desc);
    INFO(failureOf(texture));
    REQUIRE(texture.has_value());
    return std::move(*texture);
}

/// An Upload buffer on device of usage holding data; the test fails when it
/// can't be made or mapped.
template <class T, std::size_t N>
Buffer uploaded(const Device& device, const std::array<T, N>& data, const BufferUsage& usage) {
    Buffer buffer = created(device, {.size = sizeof(data), .memory = BufferMemory::Upload, .usage = usage});
    const auto written = buffer.map();
    REQUIRE(written.has_value());
    std::memcpy(*written, data.data(), sizeof(data));
    return buffer;
}

/// buffer's slot in the storage buffer table, as Buffer::bufferSlots reports it.
inline Expected<std::uint32_t> storageSlot(Buffer& buffer) {
    const auto slots = buffer.bufferSlots();
    if (!slots) {
        return std::unexpected(slots.error());
    }
    return slots->storage;
}

/// The entry point every shader checked in under tests/library/shaders has.
inline constexpr std::string_view kEntryPoint = "main";

/// bytes, a shader checked in under tests/library/shaders, at its entry point.
inline ShaderCode shaderCode(std::span<const std::uint8_t> bytes) {
    return {.code = bytes, .entryPoint = kEntryPoint};
}

/// The bytes of a buffer of size at where Buffer::map put it: map gives no size.
inline std::span<std::byte> mappedBytes(void* data, std::uint64_t size) {
    return {static_cast<std::byte*>(data), size};
}

/// value rounded up to a multiple of alignment, which is not 0.
inline std::uint64_t alignUp(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

/// Destroys what owner holds, as letting it go does, and gives back its C
/// handle, now stale. Destroy returns nothing, so a test sees it by that
/// handle's next use.
template <class Owner>
auto destroyed(Owner& owner) {
    const auto handle = owner.get();
    owner = {};
    return handle;
}

/// What a bfr.h call returned, as bfr.hpp names it. bfr.h stands in where a
/// handle may be stale or forged: bfr.hpp names an object only by a live one.
inline Result resultOf(bfr_result result) {
    return static_cast<Result>(result);
}

/// What a bfr.hpp call returned, as its failure's code; Result{} where it
/// succeeded.
template <class T>
Result resultOf(const Expected<T>& result) {
    return result.has_value() ? Result{} : result.error().code;
}

/// What bfr_get_buffer_slots says of buffer.
inline Result slotsResult(const Device& device, bfr_buffer buffer) {
    bfr_buffer_slots slots{};
    return resultOf(bfr_get_buffer_slots(device.get(), buffer, &slots));
}

/// What bfr_get_texture_slots says of texture.
inline Result slotsResult(const Device& device, bfr_texture texture) {
    bfr_texture_slots slots{};
    return resultOf(bfr_get_texture_slots(device.get(), texture, &slots));
}

/// What bfr_get_sampler_slot says of sampler.
inline Result slotResult(const Device& device, bfr_sampler sampler) {
    std::uint32_t slot = 0;
    return resultOf(bfr_get_sampler_slot(device.get(), sampler, &slot));
}

/// What bfr_capture_acceleration_structure says of structure.
inline Result captureResult(const Device& device, bfr_acceleration_structure structure) {
    bfr_acceleration_structure_capture capture{};
    return resultOf(bfr_capture_acceleration_structure(device.get(), structure, &capture));
}

/// What bfr_reset_command_pool says of pool.
inline Result resetResult(const Device& device, bfr_command_pool pool) {
    return resultOf(bfr_reset_command_pool(device.get(), pool));
}

/// What bfr_resize_swapchain says of swapchain, resized to width by height.
inline Result resizeResult(const Device& device, bfr_swapchain swapchain, std::uint32_t width, std::uint32_t height) {
    bfr_swapchain_images images{};
    return resultOf(bfr_resize_swapchain(device.get(), swapchain, width, height, &images));
}

/// What use returns from the bfr.h call it makes with device and a list
/// recording on queue, from a pool of its own that goes once use returns, or
/// why the pool or list could not be made. A call that takes a list checks the
/// handles beside it only once it finds the list, so a stale handle such a
/// call names is checked through this. It asserts nothing, so a destructor may
/// call it while a failed test unwinds.
template <class Use>
Result inList(const Device& device, std::uint32_t queue, Use use) {
    auto pool = device.createCommandPool({.queue = queue, .listCount = kPoolLists});
    if (!pool) {
        return pool.error().code;
    }
    const auto list = pool->beginCommands();
    if (!list) {
        return list.error().code;
    }
    return resultOf(use(device, *list));
}

/// What bfr_bind_pipeline says of pipeline, bound in a list recording on queue.
inline Result bindResult(const Device& device, std::uint32_t queue, bfr_pipeline pipeline) {
    return inList(device, queue, [pipeline](const Device& on, CommandList recording) {
        return bfr_bind_pipeline(on.get(), recording.get(), pipeline);
    });
}

/// The work the tests do with a texture in layout and the accesses it makes:
/// what a barrier into layout holds back, and what one out of it waits on.
/// Undefined is none.
inline BarrierScope scopeIn(TextureLayout layout) {
    switch (layout) {
        case TextureLayout::CopySrc:
            return {.stages = {.copy = true}, .accesses = {.copyRead = true}};
        case TextureLayout::CopyDst:
            return {.stages = {.copy = true}, .accesses = {.copyWrite = true}};
        case TextureLayout::Sampled:
            return {.stages = {.fragmentShader = true, .computeShader = true}, .accesses = {.sampledRead = true}};
        case TextureLayout::Storage:
            return {.stages = {.computeShader = true}, .accesses = {.storageRead = true, .storageWrite = true}};
        case TextureLayout::ColorTarget:
            return {.stages = {.colorOutput = true}, .accesses = {.colorTargetRead = true, .colorTargetWrite = true}};
        case TextureLayout::DepthStencilTarget:
            return {.stages = {.earlyFragmentTests = true, .lateFragmentTests = true},
                    .accesses = {.depthStencilRead = true, .depthStencilWrite = true}};
        case TextureLayout::DepthStencilRead:
            return {.stages = {.fragmentShader = true, .earlyFragmentTests = true, .lateFragmentTests = true},
                    .accesses = {.sampledRead = true, .depthStencilRead = true}};
        case TextureLayout::ResolveSrc:
            return {.stages = {.resolve = true}, .accesses = {.resolveRead = true}};
        case TextureLayout::ResolveDst:
            return {.stages = {.resolve = true}, .accesses = {.resolveWrite = true}};
        case TextureLayout::Undefined:
        default:
            return {};
    }
}

/// A barrier moving range of texture from before to after, ordering the work
/// scopeIn names for each.
inline TextureBarrier layoutBarrier(TextureRef texture, TextureLayout before, TextureLayout after,
                                    const TextureRange& range = {}) {
    return {.texture = texture,
            .before = scopeIn(before),
            .after = scopeIn(after),
            .layoutBefore = before,
            .layoutAfter = after,
            .range = range};
}

/// Records layoutBarrier's barrier in list.
inline Expected<void> textureBarrier(const Device& device, CommandList list, TextureRef texture, TextureLayout before,
                                     TextureLayout after, const TextureRange& range = {}) {
    const TextureBarrier barrier = layoutBarrier(texture, before, after, range);
    return device.barrier(list, {.textures = std::span{&barrier, 1}});
}

/// Records a global barrier in list ordering before's work and accesses before
/// after's.
inline Expected<void> globalBarrier(const Device& device, CommandList list, const BarrierScope& before,
                                    const BarrierScope& after) {
    const GlobalBarrier barrier{.before = before, .after = after};
    return device.barrier(list, {.globals = std::span{&barrier, 1}});
}

/// Copies writing, before reads of what they wrote: by copies, or by shaders
/// as storage.
inline constexpr BarrierScope kCopyWrites{.stages = {.copy = true}, .accesses = {.copyWrite = true}};
inline constexpr BarrierScope kCopyReads{.stages = {.copy = true}, .accesses = {.copyRead = true}};
inline constexpr BarrierScope kComputeWrites{.stages = {.computeShader = true}, .accesses = {.storageWrite = true}};
/// Acceleration structure builds, copies and compactions writing, before any
/// work reading what they wrote: a later build, a trace, a ray query, a copy
/// or a property write.
inline constexpr BarrierScope kStructureWrites{.stages = {.accelerationStructureBuild = true,
                                                          .accelerationStructureCopy = true},
                                               .accesses = {.accelerationStructureWrite = true}};
inline constexpr BarrierScope kStructureReads{.stages = {.all = true}, .accesses = {.accelerationStructureRead = true}};

/// What bfr_reset_timestamps says of pool, resetting its first timestamp in a
/// list recording on queue.
inline Result resetTimestampsResult(const Device& device, std::uint32_t queue, bfr_timestamp_pool pool) {
    constexpr std::uint32_t kFirstTimestamp = 0;
    constexpr std::uint32_t kTimestampCount = 1;
    return inList(device, queue, [pool](const Device& on, CommandList recording) {
        return bfr_reset_timestamps(on.get(), recording.get(), pool, kFirstTimestamp, kTimestampCount);
    });
}

} // namespace bfr::test
