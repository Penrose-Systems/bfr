#include "bfr.hpp"
#include "shaders/spirv.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_container_properties.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kWidth = 64;
constexpr std::uint32_t kHeight = 48;
// Lists the test command pool holds, more than a frame records.
constexpr std::uint32_t kPoolLists = 4;
constexpr bfr::Window kHeadless{.system = bfr::WindowSystem::Headless};
// The longest wait there is, so a slow GPU never fails a test.
constexpr std::uint64_t kTestWait = std::numeric_limits<std::uint64_t>::max();
// What the tests' devices hold, more than any test uses.
constexpr std::uint32_t kDeviceLists = 64;
constexpr std::uint32_t kDevicePools = 16;
constexpr std::uint32_t kDeviceSwapchains = 4;
constexpr std::uint32_t kDeviceTableSlots = 256;
constexpr std::uint32_t kDeviceSamplers = 16;
// One queue at rank 0 and level normal.
constexpr std::array<bfr::QueueDesc, 1> kOneQueue{};

bfr::DeviceDesc testDevice(std::uint32_t adapter) {
    return {.adapter = adapter,
            .window = kHeadless,
            .graphicsQueues = kOneQueue,
            .maxCommandLists = kDeviceLists,
            .maxCommandPools = kDevicePools,
            .maxSwapchains = kDeviceSwapchains,
            .maxBindlessTextures = kDeviceTableSlots,
            .maxBindlessStorageTextures = kDeviceTableSlots,
            .maxBindlessStorageBuffers = kDeviceTableSlots,
            .maxBindlessSamplers = kDeviceSamplers};
}

// Records every message an instance reports, from whichever thread reports it.
// It outlives the instance and devices it records for.
class Messages {
public:
    [[nodiscard]] bfr::MessageCallback callback() {
        return [this](bfr::MessageSeverity /*severity*/, std::string_view message) {
            const std::scoped_lock lock{mutex_};
            recorded_.emplace_back(message);
        };
    }

    [[nodiscard]] std::vector<std::string> recorded() {
        const std::scoped_lock lock{mutex_};
        return recorded_;
    }

private:
    std::mutex mutex_;
    std::vector<std::string> recorded_;
};

// The message of a failed result, for INFO; empty on success.
template <class T>
std::string failureOf(const bfr::Expected<T>& result) {
    return result.has_value() ? std::string{} : result.error().message.text();
}

// Each backend this build has, for a case that runs on each:
// `const bfr::Backend backend = GENERATE(eachBackend());`.
auto eachBackend() {
    const auto backends = bfr::backends();
    REQUIRE(backends.has_value());
    INFO("this build has no backend");
    REQUIRE_FALSE(backends->empty());
    return Catch::Generators::from_range(*backends);
}

bfr::Instance validatedInstance(bfr::Backend backend, Messages& messages) {
    auto instance = bfr::Instance::create({.backend = backend,
                                           .validation = true,
                                           .messageCallback = messages.callback(),
                                           .minSeverity = bfr::MessageSeverity::Warning});
    INFO(bfr::name(backend) << ' ' << failureOf(instance));
    REQUIRE(instance.has_value());
    return std::move(*instance);
}

// The first adapter that presents to a headless window. An adapter may not
// present to one at all: NVIDIA's driver has no headless surfaces.
std::uint32_t headlessAdapter(const bfr::Instance& instance) {
    const auto adapters = instance.adapters();
    REQUIRE(adapters.has_value());
    for (std::uint32_t adapter = 0; adapter < adapters->size(); ++adapter) {
        const auto caps = instance.swapchainCaps(adapter, {.window = kHeadless});
        if (caps.has_value()) {
            return adapter;
        }
        INFO((*adapters)[adapter].name << ' ' << caps.error().message.text());
        REQUIRE(caps.error().code == bfr::Result::ErrorPresentUnsupported);
    }
    FAIL("no adapter presents to a headless window");
    return 0;
}

// The fewest images a headless window takes on headlessAdapter, which a
// swapchain on it must ask for.
std::uint32_t fewestImages(const bfr::Instance& instance) {
    const auto caps = instance.swapchainCaps(headlessAdapter(instance), {.window = kHeadless});
    REQUIRE(caps.has_value());
    return caps->minImageCount;
}

// A device on the first adapter that presents headless.
bfr::Device headlessDevice(const bfr::Instance& instance) {
    const auto adapters = instance.adapters();
    REQUIRE(adapters.has_value());
    for (std::uint32_t adapter = 0; adapter < adapters->size(); ++adapter) {
        auto device = instance.createDevice(testDevice(adapter));
        if (device.has_value()) {
            return std::move(*device);
        }
        INFO(device.error().message.text());
        REQUIRE(device.error().code == bfr::Result::ErrorPresentUnsupported);
    }
    FAIL("no adapter presents headless");
    return {};
}

} // namespace

TEST_CASE("C++ API lists the build's backends, each with a name", "[cpp_api][unit]") {
    const auto backends = bfr::backends();

    REQUIRE(backends.has_value());
    for (const bfr::Backend backend : *backends) {
        CHECK(bfr::name(backend) != "unknown");
    }
}

TEST_CASE("C++ API reports a failed call's result and message", "[cpp_api][unit]") {
    const auto backends = bfr::backends();
    REQUIRE(backends.has_value());
    // One of the real backends is always missing: no build has all three.
    constexpr std::array kReal{bfr::Backend::Vulkan, bfr::Backend::D3d12, bfr::Backend::Metal};
    const auto* const missing = std::ranges::find_if_not(kReal, [&backends](bfr::Backend backend) {
        return std::ranges::contains(*backends, backend);
    });
    REQUIRE(missing != kReal.end());

    const auto instance = bfr::Instance::create({.backend = *missing});

    REQUIRE_FALSE(instance.has_value());
    CHECK(instance.error().code == bfr::Result::ErrorBackendUnavailable);
    CHECK_THAT(instance.error().message.text(), Catch::Matchers::ContainsSubstring("not in this build"));
}

TEST_CASE("C++ API objects that own nothing go quietly", "[cpp_api][unit]") {
    const bfr::Instance instance;
    const bfr::Device device;
    const bfr::Swapchain swapchain;

    CHECK_FALSE(instance);
    CHECK_FALSE(device);
    CHECK_FALSE(swapchain);
}

TEST_CASE("C++ API names enum values as bfr.yaml does", "[cpp_api][unit]") {
    STATIC_CHECK(bfr::name(bfr::Format::Astc4x4Unorm) == "astc_4x4_unorm");
    STATIC_CHECK(bfr::name(bfr::ColorSpace::Hdr10Pq) == "hdr10_pq");
    STATIC_CHECK(bfr::name(static_cast<bfr::Format>(-1)).empty());
}

TEST_CASE("C++ API lists adapters", "[cpp_api][gpu]") {
    const bfr::Backend backend = GENERATE(eachBackend());
    Messages messages;
    {
        const bfr::Instance instance = validatedInstance(backend, messages);

        const auto adapters = instance.adapters();

        REQUIRE(adapters.has_value());
        REQUIRE_FALSE(adapters->empty());
        CHECK_FALSE(adapters->front().name.empty());
    }
    CHECK_THAT(messages.recorded(), Catch::Matchers::IsEmpty());
}

TEST_CASE("C++ API builds, resizes and destroys a headless swapchain", "[cpp_api][gpu]") {
    const bfr::Backend backend = GENERATE(eachBackend());
    // More than the default and than a CPU adapter's minimum.
    constexpr std::uint32_t kImages = 5;
    Messages messages;
    {
        const bfr::Instance instance = validatedInstance(backend, messages);
        const bfr::Device device = headlessDevice(instance);
        auto swapchain = device.createSwapchain({.window = kHeadless,
                                                 .imageCount = kImages,
                                                 .format = bfr::Format::Bgra8Unorm,
                                                 .colorSpace = bfr::ColorSpace::Srgb});
        INFO(failureOf(swapchain));
        REQUIRE(swapchain.has_value());
        CHECK(*swapchain);

        const auto images = swapchain->resize(kWidth, kHeight);

        INFO(failureOf(images));
        REQUIRE(images.has_value());
        CHECK(images->width == kWidth);
        CHECK(images->height == kHeight);
        CHECK(images->count >= kImages);
    }
    CHECK_THAT(messages.recorded(), Catch::Matchers::IsEmpty());
}

TEST_CASE("C++ API clears and presents a headless swapchain", "[cpp_api][gpu]") {
    const bfr::Backend backend = GENERATE(eachBackend());
    Messages messages;
    {
        const bfr::Instance instance = validatedInstance(backend, messages);
        const bfr::Device device = headlessDevice(instance);
        auto swapchain = device.createSwapchain({.window = kHeadless,
                                                 .imageCount = fewestImages(instance),
                                                 .format = bfr::Format::Bgra8Unorm,
                                                 .colorSpace = bfr::ColorSpace::Srgb});
        REQUIRE(swapchain.has_value());
        REQUIRE(swapchain->resize(kWidth, kHeight).has_value());

        auto pool = device.createCommandPool({.queue = 0, .listCount = kPoolLists});
        INFO(failureOf(pool));
        REQUIRE(pool.has_value());
        bfr::Token last;
        for (int frame = 0; frame < 2; ++frame) {
            INFO("frame " << frame);
            REQUIRE(device.wait(last, kTestWait).has_value());
            REQUIRE(pool->reset().has_value());
            const auto acquired = swapchain->acquire(kTestWait);
            INFO(failureOf(acquired));
            REQUIRE(acquired.has_value());
            auto commands = pool->beginCommands();
            INFO(failureOf(commands));
            REQUIRE(commands.has_value());
            CHECK(device
                      .beginRendering(*commands, *swapchain,
                                      {.colorAttachmentCount = 1,
                                       .colorAttachments = {{{.loadOp = bfr::LoadOp::Clear,
                                                              .clearColor = {1, 0, 0, 1}}}}})
                      .has_value());
            CHECK(device.endRendering(*commands).has_value());
            const auto submitted = device.submit({.queue = 0, .commandLists = std::span{&*commands, 1}});
            INFO(failureOf(submitted));
            REQUIRE(submitted.has_value());
            last = *submitted;
            const auto presented = swapchain->present();
            INFO(failureOf(presented));
            CHECK(presented.has_value());
        }
        REQUIRE(device.wait(last, kTestWait).has_value());
        CHECK(device.poll(last).value_or(false));
    }
    CHECK_THAT(messages.recorded(), Catch::Matchers::IsEmpty());
}

TEST_CASE("C++ API passes on the C API's refusal of a swapchain without a window", "[cpp_api][gpu]") {
    const bfr::Backend backend = GENERATE(eachBackend());
    Messages messages;
    const bfr::Instance instance = validatedInstance(backend, messages);
    const bfr::Device device = headlessDevice(instance);

    const auto swapchain = device.createSwapchain({.imageCount = 0});

    REQUIRE_FALSE(swapchain.has_value());
    CHECK(swapchain.error().code == bfr::Result::ErrorInvalidArgument);
    CHECK_THAT(swapchain.error().message.text(), Catch::Matchers::ContainsSubstring("needs a window"));
}

TEST_CASE("C++ API moves objects without destroying what they own", "[cpp_api][gpu]") {
    const bfr::Backend backend = GENERATE(eachBackend());
    Messages messages;
    {
        const bfr::Instance instance = validatedInstance(backend, messages);
        bfr::Device device = headlessDevice(instance);
        auto created = device.createSwapchain({.window = kHeadless,
                                               .imageCount = fewestImages(instance),
                                               .format = bfr::Format::Bgra8Unorm,
                                               .colorSpace = bfr::ColorSpace::Srgb});
        REQUIRE(created.has_value());

        // A swapchain shares the device, so it can outlive the C++ Device it came from.
        const bfr::Device moved = std::move(device);
        bfr::Swapchain swapchain = std::move(*created);
        bfr::Swapchain assigned;
        assigned = std::move(swapchain);

        CHECK_FALSE(device); // NOLINT(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
        CHECK_FALSE(swapchain); // NOLINT(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
        CHECK(moved);
        CHECK(assigned.resize(kWidth, kHeight).has_value());
    }
    CHECK_THAT(messages.recorded(), Catch::Matchers::IsEmpty());
}

TEST_CASE("C++ API keeps a device alive for the swapchains made from it", "[cpp_api][gpu]") {
    const bfr::Backend backend = GENERATE(eachBackend());
    Messages messages;
    {
        const bfr::Instance instance = validatedInstance(backend, messages);
        bfr::Swapchain swapchain;
        {
            const bfr::Device device = headlessDevice(instance);
            auto created = device.createSwapchain({.window = kHeadless,
                                                   .imageCount = fewestImages(instance),
                                                   .format = bfr::Format::Bgra8Unorm,
                                                   .colorSpace = bfr::ColorSpace::Srgb});
            REQUIRE(created.has_value());
            swapchain = std::move(*created);
        }

        CHECK(swapchain.resize(kWidth, kHeight).has_value());
    }
    CHECK_THAT(messages.recorded(), Catch::Matchers::IsEmpty());
}

TEST_CASE("C++ API presents as the mode asked for", "[cpp_api][gpu]") {
    const bfr::Backend backend = GENERATE(eachBackend());
    Messages messages;
    {
        const bfr::Instance instance = validatedInstance(backend, messages);
        const bfr::Device device = headlessDevice(instance);
        auto fifo = device.createSwapchain({.window = kHeadless,
                                            .imageCount = fewestImages(instance),
                                            .presentMode = bfr::PresentMode::Fifo,
                                            .format = bfr::Format::Bgra8Unorm,
                                            .colorSpace = bfr::ColorSpace::Srgb});
        INFO(failureOf(fifo));
        REQUIRE(fifo.has_value());

        const auto images = fifo->resize(kWidth, kHeight);

        INFO(failureOf(images));
        REQUIRE(images.has_value());
        CHECK(images->presentMode == bfr::PresentMode::Fifo);
    }
    CHECK_THAT(messages.recorded(), Catch::Matchers::IsEmpty());
}

TEST_CASE("C++ API keeps the message callback alive while a device lives", "[cpp_api][gpu]") {
    const bfr::Backend backend = GENERATE(eachBackend());
    auto state = std::make_shared<int>(0);
    const std::weak_ptr<int> watched = state;
    auto instance = bfr::Instance::create(
        {.backend = backend,
         .messageCallback = [state](bfr::MessageSeverity /*severity*/, std::string_view /*message*/) {},
         .minSeverity = bfr::MessageSeverity::Warning});
    state.reset();
    REQUIRE(instance.has_value());
    bfr::Device device = headlessDevice(*instance);

    *instance = bfr::Instance{};
    CHECK_FALSE(watched.expired());

    device = bfr::Device{};
    CHECK(watched.expired());
}

TEST_CASE("C++ API reports when a device holds all the swapchains it can", "[cpp_api][gpu]") {
    const bfr::Backend backend = GENERATE(eachBackend());
    // Far past any limit a device sets, so the loop always ends.
    constexpr int kTries = 4096;
    Messages messages;
    const bfr::Instance instance = validatedInstance(backend, messages);
    const bfr::Device device = headlessDevice(instance);
    std::vector<bfr::Swapchain> made;
    bfr::Expected<bfr::Swapchain> swapchain;

    const std::uint32_t images = fewestImages(instance);
    for (int i = 0; i < kTries && swapchain.has_value(); ++i) {
        swapchain = device.createSwapchain({.window = kHeadless,
                                            .imageCount = images,
                                            .format = bfr::Format::Bgra8Unorm,
                                            .colorSpace = bfr::ColorSpace::Srgb});
        if (swapchain.has_value()) {
            made.push_back(std::move(*swapchain));
        }
    }

    REQUIRE_FALSE(swapchain.has_value());
    CHECK(swapchain.error().code == bfr::Result::ErrorLimitReached);
    CHECK_FALSE(made.empty());
}

TEST_CASE("C++ API creates and destroys swapchains on one device from many threads", "[cpp_api][gpu]") {
    const bfr::Backend backend = GENERATE(eachBackend());
    constexpr int kThreads = 4;
    constexpr int kEach = 8;
    Messages messages;
    {
        const bfr::Instance instance = validatedInstance(backend, messages);
        const bfr::Device device = headlessDevice(instance);
        std::atomic<int> failed{0};
        {
            std::vector<std::jthread> threads;
            threads.reserve(kThreads);
            const std::uint32_t images = fewestImages(instance);
            for (int thread = 0; thread < kThreads; ++thread) {
                threads.emplace_back([&device, &failed, images] {
                    for (int i = 0; i < kEach; ++i) {
                        auto swapchain = device.createSwapchain({.window = kHeadless,
                                                                 .imageCount = images,
                                                                 .format = bfr::Format::Bgra8Unorm,
                                                                 .colorSpace = bfr::ColorSpace::Srgb});
                        if (!swapchain.has_value() || !swapchain->resize(kWidth, kHeight).has_value()) {
                            ++failed;
                        }
                    }
                });
            }
        }
        CHECK(failed == 0);
    }
    CHECK_THAT(messages.recorded(), Catch::Matchers::IsEmpty());
}

TEST_CASE("C++ API gives an adapter's caps, formats and surface caps", "[cpp_api][gpu]") {
    const bfr::Backend backend = GENERATE(eachBackend());
    Messages messages;
    {
        const bfr::Instance instance = validatedInstance(backend, messages);
        const auto adapters = instance.adapters();
        REQUIRE(adapters.has_value());
        const auto adapter = headlessAdapter(instance);
        const bfr::AdapterInfo& listed = (*adapters)[adapter];

        const auto caps = instance.caps(adapter);
        const auto formats = instance.formats(adapter);
        const auto surface = instance.swapchainCaps(adapter, {.window = kHeadless});
        const auto swapchainFormats = instance.swapchainFormats(adapter, {.window = kHeadless});

        INFO(failureOf(caps) << failureOf(formats) << failureOf(surface) << failureOf(swapchainFormats));
        REQUIRE(caps.has_value());
        CHECK(caps->info.name == listed.name);
        CHECK(caps->info.type == listed.type);
        REQUIRE(formats.has_value());
        CHECK(formats->front().format == bfr::Format::R8Unorm);
        REQUIRE(surface.has_value());
        CHECK(surface->fifo);
        REQUIRE(swapchainFormats.has_value());
        CHECK_FALSE(swapchainFormats->empty());
    }
    CHECK_THAT(messages.recorded(), Catch::Matchers::IsEmpty());
}

TEST_CASE("C++ API creates and destroys pipelines", "[cpp_api][gpu]") {
    const bfr::Backend backend = GENERATE(eachBackend());
    // The shaders checked in are SPIR-V, which only Vulkan takes.
    if (backend != bfr::Backend::Vulkan) {
        SKIP("no bytecode is checked in for this backend");
    }
    constexpr std::uint32_t kWorkgroup = 64;
    Messages messages;
    {
        const bfr::Instance instance = validatedInstance(backend, messages);
        const bfr::Device device = headlessDevice(instance);
        const auto code = [](const auto& bytes) {
            return bfr::ShaderCode{.code = bytes, .entryPoint = "main"};
        };
        auto compute = device.createComputePipeline(
            {.compute = code(bfr::test::spirv::kBindlessComp), .workgroupSize = {kWorkgroup, 1, 1}});
        INFO(failureOf(compute));
        REQUIRE(compute.has_value());
        bfr::GraphicsPipelineDesc desc{.vertex = code(bfr::test::spirv::kBindlessVert),
                                       .fragment = code(bfr::test::spirv::kBindlessFrag),
                                       .colorTargetCount = 1,
                                       .samples = 1};
        desc.colorTargets[0] = {.format = bfr::Format::Rgba8Unorm,
                                .blend = {.enable = true,
                                          .srcColor = bfr::BlendFactor::SrcAlpha,
                                          .dstColor = bfr::BlendFactor::OneMinusSrcAlpha,
                                          .writeMask = BFR_MASK_ALL}};
        auto graphics = device.createGraphicsPipeline(desc);
        INFO(failureOf(graphics));
        REQUIRE(graphics.has_value());
    }
    CHECK_THAT(messages.recorded(), Catch::Matchers::IsEmpty());
}

TEST_CASE("C++ API creates a sampler with a slot, destroyed with its object", "[cpp_api][gpu]") {
    const bfr::Backend backend = GENERATE(eachBackend());
    Messages messages;
    {
        const bfr::Instance instance = validatedInstance(backend, messages);
        const bfr::Device device = headlessDevice(instance);
        auto sampler = device.createSampler({.magFilter = bfr::Filter::Linear,
                                             .minFilter = bfr::Filter::Linear,
                                             .addressU = bfr::AddressMode::ClampToEdge,
                                             .addressV = bfr::AddressMode::ClampToEdge,
                                             .addressW = bfr::AddressMode::ClampToEdge});
        INFO(failureOf(sampler));
        REQUIRE(sampler.has_value());
        const auto slot = sampler->samplerSlot();
        INFO(failureOf(slot));
        REQUIRE(slot.has_value());
        CHECK(*slot < kDeviceSamplers);
    }
    CHECK_THAT(messages.recorded(), Catch::Matchers::IsEmpty());
}

TEST_CASE("C++ API dispatches a compute pipeline through a buffer's slot", "[cpp_api][gpu]") {
    const bfr::Backend backend = GENERATE(eachBackend());
    // The shaders checked in are SPIR-V, which only Vulkan takes.
    if (backend != bfr::Backend::Vulkan) {
        SKIP("no bytecode is checked in for this backend");
    }
    constexpr std::uint32_t kWorkgroup = 64;
    constexpr std::uint64_t kBytes = std::uint64_t{kWorkgroup} * sizeof(std::uint32_t);
    Messages messages;
    {
        const bfr::Instance instance = validatedInstance(backend, messages);
        const bfr::Device device = headlessDevice(instance);
        auto pipeline = device.createComputePipeline(
            {.compute = {.code = bfr::test::spirv::kBindlessComp, .entryPoint = "main"},
             .workgroupSize = {kWorkgroup, 1, 1}});
        INFO(failureOf(pipeline));
        REQUIRE(pipeline.has_value());
        auto target = device.createBuffer({.size = kBytes, .usage = {.copySrc = true, .storage = true}});
        REQUIRE(target.has_value());
        auto readback = device.createBuffer(
            {.size = kBytes, .memory = bfr::BufferMemory::Readback, .usage = {.copyDst = true}});
        REQUIRE(readback.has_value());
        const auto slot = target->bufferSlot();
        REQUIRE(slot.has_value());
        CHECK(readback->bufferSlot().error().code == bfr::Result::ErrorInvalidArgument);

        auto pool = device.createCommandPool({.queue = 0, .listCount = 1, .pushConstantUpdates = 1});
        REQUIRE(pool.has_value());
        auto commands = pool->beginCommands();
        REQUIRE(commands.has_value());
        CHECK(device.bindPipeline(*commands, *pipeline).has_value());
        // The shader reads the slot from the first push constant.
        const auto bytes = std::bit_cast<std::array<std::uint8_t, sizeof(std::uint32_t)>>(*slot);
        CHECK(device.pushConstants(*commands, bytes).has_value());
        CHECK(device.dispatch(*commands, 1, 1, 1).has_value());
        CHECK(device.barrier(*commands).has_value());
        const bfr::BufferCopy whole{.srcOffset = 0, .dstOffset = 0, .size = kBytes};
        CHECK(device.copyBuffer(*commands, *target, *readback, std::span{&whole, 1}).has_value());
        const auto submitted = device.submit({.queue = 0, .commandLists = std::span{&*commands, 1}});
        INFO(failureOf(submitted));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted, kTestWait).has_value());

        const auto data = readback->map();
        REQUIRE(data.has_value());
        std::array<std::uint32_t, kWorkgroup> values{};
        std::memcpy(values.data(), *data, kBytes);
        for (std::uint32_t i = 0; i < kWorkgroup; ++i) {
            CHECK(values.at(i) == i);
        }
    }
    CHECK_THAT(messages.recorded(), Catch::Matchers::IsEmpty());
}

TEST_CASE("C++ API renders to a texture it names by a borrowed pointer", "[cpp_api][gpu]") {
    constexpr std::uint32_t kSide = 8;
    const bfr::Backend backend = GENERATE(eachBackend());
    Messages messages;
    {
        const bfr::Instance instance = validatedInstance(backend, messages);
        const bfr::Device device = headlessDevice(instance);
        auto target = device.createTexture({.format = bfr::Format::Rgba8Unorm,
                                            .width = kSide,
                                            .height = kSide,
                                            .depth = 1,
                                            .layers = 1,
                                            .mipLevels = 1,
                                            .samples = 1,
                                            .usage = {.colorAttachment = true}});
        INFO(failureOf(target));
        REQUIRE(target.has_value());
        auto pool = device.createCommandPool({.queue = 0, .listCount = 1});
        REQUIRE(pool.has_value());
        auto commands = pool->beginCommands();
        REQUIRE(commands.has_value());
        REQUIRE(
            device.textureBarrier(*commands, *target, bfr::TextureLayout::Undefined, bfr::TextureLayout::ColorTarget)
                .has_value());
        bfr::RenderingDesc rendering{.colorAttachmentCount = 1};
        rendering.colorAttachments[0] = {.texture = &*target, .loadOp = bfr::LoadOp::Clear, .clearColor = {0, 0, 0, 1}};
        const auto begun = device.beginRendering(*commands, bfr::Swapchain{}, rendering);
        INFO(failureOf(begun));
        REQUIRE(begun.has_value());
        CHECK(device.endRendering(*commands).has_value());
        const auto submitted = device.submit({.queue = 0, .commandLists = std::span{&*commands, 1}});
        INFO(failureOf(submitted));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted, kTestWait).has_value());
    }
    CHECK_THAT(messages.recorded(), Catch::Matchers::IsEmpty());
}

TEST_CASE("C++ API gives a device's memory budget or says the driver can't", "[cpp_api][gpu]") {
    const bfr::Backend backend = GENERATE(eachBackend());
    Messages messages;
    {
        const bfr::Instance instance = validatedInstance(backend, messages);
        const auto adapter = headlessAdapter(instance);
        const auto caps = instance.caps(adapter);
        REQUIRE(caps.has_value());
        auto device = instance.createDevice(testDevice(adapter));
        INFO(failureOf(device));
        REQUIRE(device.has_value());

        const auto budget = device->budget();

        if (budget.has_value()) {
            CHECK(budget->heapCount == caps->memory.heapCount);
        }
        else {
            CHECK(budget.error().code == bfr::Result::ErrorUnsupported);
        }
    }
    CHECK_THAT(messages.recorded(), Catch::Matchers::IsEmpty());
}
