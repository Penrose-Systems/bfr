#include "allocation/allocation_counter.h"
#include "bfr/bfr.h"
#include "bfr/core/log.h"
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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

std::string lastError() {
    const bfr_string error = bfr_get_last_error();
    return {error.data, error.length};
}

// Records every message the instance reports, from whichever thread reports it.
void record(bfr_message_severity /*severity*/, bfr_string message, void* userData) {
    static std::mutex mutex;
    const std::scoped_lock lock{mutex};
    static_cast<std::vector<std::string>*>(userData)->emplace_back(message.data, message.length);
}

constexpr bfr_window kHeadless{.window = nullptr, .display = nullptr, .system = BFR_WINDOW_SYSTEM_HEADLESS};
// The longest wait there is, so a slow GPU never fails a test.
constexpr std::uint64_t kTestWait = std::numeric_limits<std::uint64_t>::max();
// A chain entry of a type no desc takes.
constexpr bfr_chain kStray{.next = nullptr, .type = BFR_STRUCT_TYPE_FORCE32};

constexpr std::uint32_t kWidth = 64;
constexpr std::uint32_t kHeight = 48;

// A value no BFR enum has.
constexpr int kNotAValue = 99;
// What the tests' devices hold, more than any test uses.
constexpr std::uint32_t kDeviceLists = 64;
constexpr std::uint32_t kDevicePools = 16;
constexpr std::uint32_t kDeviceSwapchains = 4;
constexpr std::uint32_t kDeviceTableSlots = 256;
constexpr std::uint32_t kDeviceSamplers = 16;
// One queue at rank 0 and level normal.
constexpr bfr_queue_desc kOneQueue{.rank = 0, .level = BFR_QUEUE_LEVEL_NORMAL};

// The backends this build has.
std::vector<bfr_backend> backendsInBuild() {
    std::uint32_t count = 0;
    REQUIRE(bfr_get_backends(&count, nullptr) == BFR_RESULT_SUCCESS);
    std::vector<bfr_backend> backends(count);
    REQUIRE(bfr_get_backends(&count, backends.data()) == BFR_RESULT_SUCCESS);
    backends.resize(count);
    return backends;
}

// Each backend this build has, for a case that runs on each:
// `const bfr_backend backend = GENERATE(eachBackend());`.
auto eachBackend() {
    const std::vector<bfr_backend> backends = backendsInBuild();
    INFO("this build has no backend");
    REQUIRE_FALSE(backends.empty());
    return Catch::Generators::from_range(backends);
}

// An instance on backend, with validation on unless told otherwise, recording
// its messages, destroyed at scope end.
struct Instance {
    Instance(bfr_backend backend, std::vector<std::string>& messages, bfr_bool validation = BFR_TRUE,
             bfr_bool callTimings = BFR_FALSE) {
        const bfr_instance_desc desc{.next = nullptr,
                                     .backend = backend,
                                     .validation = validation,
                                     .message_callback = record,
                                     .min_severity = BFR_MESSAGE_SEVERITY_WARNING,
                                     .message_user_data = &messages,
                                     .application_name = {},
                                     .application_version = 0,
                                     .engine_name = {},
                                     .engine_version = 0,
                                     .call_timings = callTimings};
        const bfr_result result = bfr_create_instance(&desc, &instance);
        INFO(lastError());
        REQUIRE(result == BFR_RESULT_SUCCESS);
    }
    ~Instance() { bfr_destroy_instance(instance); }
    Instance(const Instance&) = delete;
    Instance& operator=(const Instance&) = delete;
    Instance(Instance&&) = delete;
    Instance& operator=(Instance&&) = delete;

    bfr_instance* instance = nullptr;
};

// A device on the first adapter that presents headless, destroyed at scope end.
struct HeadlessDevice {
    explicit HeadlessDevice(bfr_instance* instance) {
        std::uint32_t count = 0;
        REQUIRE(bfr_get_adapters(instance, &count, nullptr) == BFR_RESULT_SUCCESS);
        for (std::uint32_t adapter = 0; adapter < count && device == nullptr; ++adapter) {
            const bfr_device_desc desc{.next = nullptr,
                                       .adapter = adapter,
                                       .window = kHeadless,
                                       .graphics_queues = &kOneQueue,
                                       .graphics_queue_count = 1,
                                       .compute_queues = nullptr,
                                       .compute_queue_count = 0,
                                       .copy_queues = nullptr,
                                       .copy_queue_count = 0,
                                       .max_command_lists = kDeviceLists,
                                       .max_command_pools = kDevicePools,
                                       .max_swapchains = kDeviceSwapchains,
                                       .max_bindless_textures = kDeviceTableSlots,
                                       .max_bindless_storage_textures = kDeviceTableSlots,
                                       .max_bindless_storage_buffers = kDeviceTableSlots,
                                       .max_bindless_samplers = kDeviceSamplers,
                                       .max_timestamp_pools = 1,
                                       .max_sparse_memories = 0,
                                       .max_tile_mappings = 0,
                                       .indirect_strides = nullptr,
                                       .indirect_stride_count = 0};
            const bfr_result result = bfr_create_device(instance, &desc, &device);
            INFO(lastError());
            REQUIRE((result == BFR_RESULT_SUCCESS || result == BFR_RESULT_ERROR_PRESENT_UNSUPPORTED));
            if (device != nullptr) {
                this->adapter = adapter;
            }
        }
        REQUIRE(device != nullptr);
        const bfr_swapchain_query surface{.next = nullptr, .window = kHeadless};
        bfr_swapchain_caps caps{};
        REQUIRE(bfr_get_swapchain_caps(instance, adapter, &surface, &caps) == BFR_RESULT_SUCCESS);
        images = caps.min_image_count;
    }
    ~HeadlessDevice() { bfr_destroy_device(device); }
    HeadlessDevice(const HeadlessDevice&) = delete;
    HeadlessDevice& operator=(const HeadlessDevice&) = delete;
    HeadlessDevice(HeadlessDevice&&) = delete;
    HeadlessDevice& operator=(HeadlessDevice&&) = delete;

    bfr_device* device = nullptr;
    // The adapter the device is on.
    std::uint32_t adapter = 0;
    // The fewest images the window takes, which a swapchain on the device asks for.
    std::uint32_t images = 0;
};

} // namespace

TEST_CASE("C API rejects null arguments and says which", "[c_api][unit]") {
    bfr_instance* instance = nullptr;

    CHECK(bfr_create_instance(nullptr, &instance) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
    CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("must not be null"));
    CHECK(instance == nullptr);
}

TEST_CASE("C API rejects a backend value BFR does not have", "[c_api][unit]") {
    const bfr_instance_desc desc{.next = nullptr,
                                 .backend = static_cast<bfr_backend>(kNotAValue),
                                 .validation = BFR_FALSE,
                                 .message_callback = nullptr,
                                 .min_severity = BFR_MESSAGE_SEVERITY_WARNING,
                                 .message_user_data = nullptr,
                                 .application_name = {},
                                 .application_version = 0,
                                 .engine_name = {},
                                 .engine_version = 0,
                                 .call_timings = BFR_FALSE};
    bfr_instance* instance = nullptr;

    CHECK(bfr_create_instance(&desc, &instance) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
    CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("not one BFR has"));
}

TEST_CASE("C API lists the build's backends in two calls", "[c_api][unit]") {
    std::uint32_t count = 0;

    REQUIRE(bfr_get_backends(&count, nullptr) == BFR_RESULT_SUCCESS);
    std::vector<bfr_backend> backends(count);
    REQUIRE(bfr_get_backends(&count, backends.data()) == BFR_RESULT_SUCCESS);

    CHECK(count == backends.size());
    CHECK_FALSE(std::ranges::contains(backends, BFR_BACKEND_FORCE32));
    CHECK(bfr_get_backends(nullptr, nullptr) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_CASE("C API says when the caller's backend array is too small", "[c_api][unit]") {
    std::uint32_t count = 1;
    std::array<bfr_backend, 1> one{};

    const bfr_result result = bfr_get_backends(&count, one.data());

    CHECK(count == 1);
    CHECK(result == (backendsInBuild().size() > 1 ? BFR_RESULT_INCOMPLETE : BFR_RESULT_SUCCESS));
}

TEST_CASE("C API takes the application's and engine's names, and refuses a null name with a length", "[c_api][gpu]") {
    constexpr std::string_view kApplication = "Test App";
    constexpr std::string_view kEngine = "Test Engine";
    constexpr std::size_t kSomeBytes = 4;
    const bfr_backend backend = GENERATE(eachBackend());
    bfr_instance_desc desc{.next = nullptr,
                           .backend = backend,
                           .validation = BFR_FALSE,
                           .message_callback = nullptr,
                           .min_severity = BFR_MESSAGE_SEVERITY_WARNING,
                           .message_user_data = nullptr,
                           .application_name = {.data = kApplication.data(), .length = kApplication.size()},
                           .application_version = 1,
                           .engine_name = {.data = kEngine.data(), .length = kEngine.size()},
                           .engine_version = 1,
                           .call_timings = BFR_FALSE};
    bfr_instance* instance = nullptr;

    REQUIRE(bfr_create_instance(&desc, &instance) == BFR_RESULT_SUCCESS);
    bfr_destroy_instance(instance);
    instance = nullptr;
    desc.engine_name = {.data = nullptr, .length = kSomeBytes};
    CHECK(bfr_create_instance(&desc, &instance) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
    CHECK(instance == nullptr);
}

TEST_CASE("C API reports a backend this build lacks", "[c_api][unit]") {
    const std::vector<bfr_backend> built = backendsInBuild();
    for (const bfr_backend backend : {BFR_BACKEND_VULKAN, BFR_BACKEND_D3D12, BFR_BACKEND_METAL}) {
        if (std::ranges::contains(built, backend)) {
            continue;
        }
        const bfr_instance_desc desc{.next = nullptr,
                                     .backend = backend,
                                     .validation = BFR_FALSE,
                                     .message_callback = nullptr,
                                     .min_severity = BFR_MESSAGE_SEVERITY_WARNING,
                                     .message_user_data = nullptr,
                                     .application_name = {},
                                     .application_version = 0,
                                     .engine_name = {},
                                     .engine_version = 0,
                                     .call_timings = BFR_FALSE};
        bfr_instance* instance = nullptr;

        INFO(static_cast<int>(backend));
        CHECK(bfr_create_instance(&desc, &instance) == BFR_RESULT_ERROR_BACKEND_UNAVAILABLE);
        CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("not in this build"));
        CHECK(instance == nullptr);
    }
}

TEST_CASE("C API rejects a struct chained where it does not belong", "[c_api][unit]") {
    const bfr_instance_desc desc{.next = &kStray,
                                 .backend = BFR_BACKEND_VULKAN,
                                 .validation = BFR_FALSE,
                                 .message_callback = nullptr,
                                 .min_severity = BFR_MESSAGE_SEVERITY_WARNING,
                                 .message_user_data = nullptr,
                                 .application_name = {},
                                 .application_version = 0,
                                 .engine_name = {},
                                 .engine_version = 0,
                                 .call_timings = BFR_FALSE};
    bfr_instance* instance = nullptr;

    CHECK(bfr_create_instance(&desc, &instance) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_CASE("C API rejects a struct chained to a device or swapchain desc", "[c_api][gpu]") {
    const bfr_backend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const Instance instance{backend, messages};
        const bfr_device_desc deviceDesc{.next = &kStray,
                                         .adapter = 0,
                                         .window = kHeadless,
                                         .graphics_queues = &kOneQueue,
                                         .graphics_queue_count = 1,
                                         .compute_queues = nullptr,
                                         .compute_queue_count = 0,
                                         .copy_queues = nullptr,
                                         .copy_queue_count = 0,
                                         .max_command_lists = kDeviceLists,
                                         .max_command_pools = kDevicePools,
                                         .max_swapchains = kDeviceSwapchains,
                                         .max_bindless_textures = kDeviceTableSlots,
                                         .max_bindless_storage_textures = kDeviceTableSlots,
                                         .max_bindless_storage_buffers = kDeviceTableSlots,
                                         .max_bindless_samplers = kDeviceSamplers,
                                         .max_timestamp_pools = 0,
                                         .max_sparse_memories = 0,
                                         .max_tile_mappings = 0,
                                         .indirect_strides = nullptr,
                                         .indirect_stride_count = 0};
        bfr_device* stray = nullptr;
        CHECK(bfr_create_device(instance.instance, &deviceDesc, &stray) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
        CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("takes no chained structs"));
        CHECK(stray == nullptr);

        const HeadlessDevice device{instance.instance};
        const bfr_swapchain_desc swapchainDesc{.next = &kStray,
                                               .window = kHeadless,
                                               .queue = 0,
                                               .image_count = device.images,
                                               .present_mode = BFR_PRESENT_MODE_FIFO,
                                               .format = BFR_FORMAT_BGRA8_UNORM,
                                               .color_space = BFR_COLOR_SPACE_SRGB,
                                               .alpha_mode = BFR_ALPHA_MODE_OPAQUE};
        bfr_swapchain swapchain{};
        CHECK(bfr_create_swapchain(device.device, &swapchainDesc, &swapchain) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
        CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("takes no chained structs"));
        CHECK(swapchain.bits == 0);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("C API rejects a device desc's null queues with a count, and a queue level BFR doesn't have",
          "[c_api][gpu]") {
    const bfr_backend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const Instance instance{backend, messages};
        const bfr_queue_desc unknownLevel{.rank = 0, .level = static_cast<bfr_queue_level>(kNotAValue)};
        const bfr_device_desc nullQueues{.next = nullptr,
                                         .adapter = 0,
                                         .window = {},
                                         .graphics_queues = nullptr,
                                         .graphics_queue_count = 1,
                                         .compute_queues = nullptr,
                                         .compute_queue_count = 0,
                                         .copy_queues = nullptr,
                                         .copy_queue_count = 0,
                                         .max_command_lists = 0,
                                         .max_command_pools = 0,
                                         .max_swapchains = 0,
                                         .max_bindless_textures = 0,
                                         .max_bindless_storage_textures = 0,
                                         .max_bindless_storage_buffers = 0,
                                         .max_bindless_samplers = 0,
                                         .max_timestamp_pools = 0,
                                         .max_sparse_memories = 0,
                                         .max_tile_mappings = 0,
                                         .indirect_strides = nullptr,
                                         .indirect_stride_count = 0};
        bfr_device* device = nullptr;
        CHECK(bfr_create_device(instance.instance, &nullQueues, &device) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
        CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("graphics_queues is null"));

        bfr_device_desc unknown = nullQueues;
        unknown.graphics_queues = &unknownLevel;
        CHECK(bfr_create_device(instance.instance, &unknown, &device) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
        CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("not one BFR has"));
        CHECK(device == nullptr);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("C API destroy calls take null and stale handles", "[c_api][unit]") {
    bfr_destroy_instance(nullptr);
    bfr_destroy_device(nullptr);
    bfr_destroy_swapchain(nullptr, bfr_swapchain{.bits = 0});
    bfr_destroy_buffer(nullptr, bfr_buffer{.bits = 0});
    bfr_destroy_sampler(nullptr, bfr_sampler{.bits = 0});
}

TEST_CASE("C API lists adapters in two calls", "[c_api][gpu]") {
    const bfr_backend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    const Instance instance{backend, messages};
    std::uint32_t count = 0;
    REQUIRE(bfr_get_adapters(instance.instance, &count, nullptr) == BFR_RESULT_SUCCESS);
    REQUIRE(count > 0);
    std::vector<bfr_adapter_info> adapters(count);
    std::uint32_t written = count;

    CHECK(bfr_get_adapters(instance.instance, &written, adapters.data()) == BFR_RESULT_SUCCESS);

    CHECK(written == count);
    CHECK(adapters.front().name.length > 0);
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("C API says when the caller's adapter array is too small", "[c_api][gpu]") {
    const bfr_backend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    const Instance instance{backend, messages};
    bfr_adapter_info adapter{};
    std::uint32_t written = 0;

    CHECK(bfr_get_adapters(instance.instance, &written, &adapter) == BFR_RESULT_INCOMPLETE);
    CHECK(written == 0);
}

TEST_CASE("C API builds, resizes and destroys a headless swapchain", "[c_api][gpu]") {
    const bfr_backend backend = GENERATE(eachBackend());
    // More than the default and than lavapipe's minimum.
    constexpr std::uint32_t kImages = 5;
    std::vector<std::string> messages;
    {
        const Instance instance{backend, messages};
        const HeadlessDevice device{instance.instance};
        const bfr_swapchain_desc desc{.next = nullptr,
                                      .window = kHeadless,
                                      .queue = 0,
                                      .image_count = kImages,
                                      .present_mode = BFR_PRESENT_MODE_FIFO,
                                      .format = BFR_FORMAT_BGRA8_UNORM,
                                      .color_space = BFR_COLOR_SPACE_SRGB,
                                      .alpha_mode = {}};
        bfr_swapchain swapchain{};
        REQUIRE(bfr_create_swapchain(device.device, &desc, &swapchain) == BFR_RESULT_SUCCESS);
        CHECK(swapchain.bits != 0);

        bfr_swapchain_images images{};
        const bfr_result resized = bfr_resize_swapchain(device.device, swapchain, kWidth, kHeight, &images);

        INFO(lastError());
        REQUIRE(resized == BFR_RESULT_SUCCESS);
        CHECK(images.width == kWidth);
        CHECK(images.height == kHeight);
        CHECK(images.count >= kImages);

        bfr_destroy_swapchain(device.device, swapchain);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

namespace {

struct Mode {
    bfr_present_mode mode;
    bfr_bool bfr_swapchain_caps::* allowed;
    const char* name;
};

constexpr std::array kModes{
    Mode{.mode = BFR_PRESENT_MODE_FIFO, .allowed = &bfr_swapchain_caps::fifo, .name = "fifo"},
    Mode{.mode = BFR_PRESENT_MODE_FIFO_RELAXED, .allowed = &bfr_swapchain_caps::fifo_relaxed, .name = "fifo_relaxed"},
    Mode{.mode = BFR_PRESENT_MODE_MAILBOX, .allowed = &bfr_swapchain_caps::mailbox, .name = "mailbox"},
    Mode{.mode = BFR_PRESENT_MODE_IMMEDIATE, .allowed = &bfr_swapchain_caps::immediate, .name = "immediate"},
};

// Makes a swapchain on device in mode, which caps allows, and checks it is made.
void checkPresentMode(bfr_device* device, const bfr_swapchain_caps& caps, const Mode& mode) {
    const bfr_swapchain_desc desc{.next = nullptr,
                                  .window = kHeadless,
                                  .queue = 0,
                                  .image_count = caps.min_image_count,
                                  .present_mode = mode.mode,
                                  .format = BFR_FORMAT_BGRA8_UNORM,
                                  .color_space = BFR_COLOR_SPACE_SRGB,
                                  .alpha_mode = {}};
    bfr_swapchain swapchain{};
    bfr_swapchain_images images{};

    const bfr_result created = bfr_create_swapchain(device, &desc, &swapchain);
    const bfr_result resized = bfr_resize_swapchain(device, swapchain, kWidth, kHeight, &images);

    INFO(mode.name << ": " << lastError());
    CHECK(created == BFR_RESULT_SUCCESS);
    CHECK(resized == BFR_RESULT_SUCCESS);
    CHECK(images.present_mode == mode.mode);
    bfr_destroy_swapchain(device, swapchain);
}

} // namespace

TEST_CASE("C API presents as each mode the window allows", "[c_api][gpu]") {
    const bfr_backend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const Instance instance{backend, messages};
        const HeadlessDevice device{instance.instance};
        const bfr_swapchain_query surface{.next = nullptr, .window = kHeadless};
        bfr_swapchain_caps caps{};
        REQUIRE(bfr_get_swapchain_caps(instance.instance, device.adapter, &surface, &caps) == BFR_RESULT_SUCCESS);

        for (const Mode& mode : kModes) {
            if (caps.*mode.allowed != BFR_FALSE) {
                checkPresentMode(device.device, caps, mode);
            }
        }
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("C API builds a swapchain as asked and reports what it is", "[c_api][gpu]") {
    const bfr_backend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    {
        const Instance instance{backend, messages};
        const HeadlessDevice device{instance.instance};
        const bfr_swapchain_desc desc{.next = nullptr,
                                      .window = kHeadless,
                                      .queue = 0,
                                      .image_count = device.images,
                                      .present_mode = {},
                                      .format = BFR_FORMAT_BGRA8_UNORM,
                                      .color_space = BFR_COLOR_SPACE_SRGB,
                                      .alpha_mode = {}};
        bfr_swapchain swapchain{};
        REQUIRE(bfr_create_swapchain(device.device, &desc, &swapchain) == BFR_RESULT_SUCCESS);
        bfr_swapchain_images images{};

        const bfr_result resized = bfr_resize_swapchain(device.device, swapchain, kWidth, kHeight, &images);

        INFO(lastError());
        REQUIRE(resized == BFR_RESULT_SUCCESS);
        CHECK(images.format == BFR_FORMAT_BGRA8_UNORM);
        CHECK(images.color_space == BFR_COLOR_SPACE_SRGB);
        CHECK(images.present_mode == BFR_PRESENT_MODE_FIFO);
        CHECK(images.alpha_mode == BFR_ALPHA_MODE_OPAQUE);
        bfr_destroy_swapchain(device.device, swapchain);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("C API rejects a swapchain setting that is not one BFR has", "[c_api][gpu]") {
    const bfr_backend backend = GENERATE(eachBackend());
    // A value BFR doesn't have is the caller's mistake, not the window's.
    const bfr_swapchain_desc desc = GENERATE(bfr_swapchain_desc{.next = nullptr,
                                                                .window = kHeadless,
                                                                .queue = 0,
                                                                .image_count = 0,
                                                                .present_mode = static_cast<bfr_present_mode>(
                                                                    kNotAValue),
                                                                .format = BFR_FORMAT_BGRA8_UNORM,
                                                                .color_space = BFR_COLOR_SPACE_SRGB,
                                                                .alpha_mode = {}},
                                             bfr_swapchain_desc{.next = nullptr,
                                                                .window = kHeadless,
                                                                .queue = 0,
                                                                .image_count = 0,
                                                                .present_mode = {},
                                                                .format = static_cast<bfr_format>(kNotAValue),
                                                                .color_space = BFR_COLOR_SPACE_SRGB,
                                                                .alpha_mode = {}},
                                             bfr_swapchain_desc{.next = nullptr,
                                                                .window = kHeadless,
                                                                .queue = 0,
                                                                .image_count = 0,
                                                                .present_mode = {},
                                                                .format = BFR_FORMAT_BGRA8_UNORM,
                                                                .color_space = static_cast<bfr_color_space>(kNotAValue),
                                                                .alpha_mode = {}},
                                             bfr_swapchain_desc{.next = nullptr,
                                                                .window = kHeadless,
                                                                .queue = 0,
                                                                .image_count = 0,
                                                                .present_mode = {},
                                                                .format = BFR_FORMAT_BGRA8_UNORM,
                                                                .color_space = BFR_COLOR_SPACE_SRGB,
                                                                .alpha_mode = static_cast<bfr_alpha_mode>(kNotAValue)},
                                             // 0 is no format and no color space; it stands for no other.
                                             bfr_swapchain_desc{.next = nullptr,
                                                                .window = kHeadless,
                                                                .queue = 0,
                                                                .image_count = 0,
                                                                .present_mode = {},
                                                                .format = {},
                                                                .color_space = BFR_COLOR_SPACE_SRGB,
                                                                .alpha_mode = {}},
                                             bfr_swapchain_desc{.next = nullptr,
                                                                .window = kHeadless,
                                                                .queue = 0,
                                                                .image_count = 0,
                                                                .present_mode = {},
                                                                .format = BFR_FORMAT_BGRA8_UNORM,
                                                                .color_space = {},
                                                                .alpha_mode = {}});
    std::vector<std::string> messages;
    {
        const Instance instance{backend, messages};
        const HeadlessDevice device{instance.instance};
        bfr_swapchain swapchain{};

        CHECK(bfr_create_swapchain(device.device, &desc, &swapchain) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
        CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("not one BFR has"));
        CHECK(swapchain.bits == 0);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("C API reports a destroyed swapchain as invalid and destroys it again quietly", "[c_api][gpu]") {
    const bfr_backend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    const Instance instance{backend, messages};
    const HeadlessDevice device{instance.instance};
    const bfr_swapchain_desc desc{.next = nullptr,
                                  .window = kHeadless,
                                  .queue = 0,
                                  .image_count = device.images,
                                  .present_mode = BFR_PRESENT_MODE_FIFO,
                                  .format = BFR_FORMAT_BGRA8_UNORM,
                                  .color_space = BFR_COLOR_SPACE_SRGB,
                                  .alpha_mode = {}};
    bfr_swapchain swapchain{};
    REQUIRE(bfr_create_swapchain(device.device, &desc, &swapchain) == BFR_RESULT_SUCCESS);
    bfr_destroy_swapchain(device.device, swapchain);
    bfr_swapchain_images images{};

    CHECK(bfr_resize_swapchain(device.device, swapchain, kWidth, kHeight, &images) == BFR_RESULT_ERROR_INVALID_HANDLE);
    bfr_destroy_swapchain(device.device, swapchain);
    CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("stale"));
}

TEST_CASE("C API passes sparse calls through, and refuses what is not sparse or is past the device's room",
          "[c_api][gpu]") {
    constexpr std::uint64_t kSize = std::uint64_t{1} << 20U;
    const bfr_backend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    const Instance instance{backend, messages};
    const HeadlessDevice device{instance.instance};
    bfr_buffer_desc desc{.next = nullptr,
                         .size = kSize,
                         .memory = BFR_BUFFER_MEMORY_DEVICE,
                         .usage = {.copy_src = BFR_FALSE,
                                   .copy_dst = BFR_TRUE,
                                   .uniform = BFR_FALSE,
                                   .storage = BFR_FALSE,
                                   .index = BFR_FALSE,
                                   .indirect = BFR_FALSE},
                         .sparse = BFR_FALSE,
                         .sparse_aliased = BFR_FALSE,
                         .tile_bytes = 0};
    bfr_buffer plain{};
    REQUIRE(bfr_create_buffer(device.device, &desc, &plain) == BFR_RESULT_SUCCESS);
    bfr_buffer_tiling tiling{};
    CHECK(bfr_get_buffer_tiling(device.device, plain, &tiling) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
    const bfr_sparse_memory_desc memoryDesc{.size = kSize, .tile_bytes = 0, .texture = {}, .buffer = plain};
    bfr_sparse_memory memory{};
    CHECK(bfr_create_sparse_memory(device.device, &memoryDesc, &memory) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
    CHECK(memory.bits == 0);
    const bfr_tile_mapping_desc nullRegions{.next = nullptr,
                                            .queue = 0,
                                            .waits = nullptr,
                                            .wait_count = 0,
                                            .memory = {},
                                            .texture = {},
                                            .texture_regions = nullptr,
                                            .texture_region_count = 0,
                                            .buffer = plain,
                                            .buffer_regions = nullptr,
                                            .buffer_region_count = 1};
    CHECK(bfr_update_tile_mappings(device.device, &nullRegions, nullptr) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
    bfr_destroy_buffer(device.device, plain);

    bfr_caps caps{};
    REQUIRE(bfr_get_caps(instance.instance, 0, &caps) == BFR_RESULT_SUCCESS);
    if (caps.sparse.buffers == BFR_FALSE) {
        SKIP("the adapter makes no sparse buffers");
    }
    desc.sparse = BFR_TRUE;
    desc.tile_bytes = caps.sparse.tile_byte_count != 0 ? caps.sparse.tile_bytes[0] : 0;
    bfr_buffer sparse{};
    REQUIRE(bfr_create_buffer(device.device, &desc, &sparse) == BFR_RESULT_SUCCESS);
    CHECK(bfr_get_buffer_tiling(device.device, sparse, &tiling) == BFR_RESULT_SUCCESS);
    CHECK(tiling.tile_bytes != 0);
    // The test device holds no sparse memories.
    const bfr_sparse_memory_desc held{.size = kSize, .tile_bytes = desc.tile_bytes, .texture = {}, .buffer = sparse};
    CHECK(bfr_create_sparse_memory(device.device, &held, &memory) == BFR_RESULT_ERROR_LIMIT_REACHED);
    bfr_destroy_buffer(device.device, sparse);
}

TEST_CASE("C API creates, maps and destroys a buffer, and refuses what it can't map", "[c_api][gpu]") {
    constexpr std::uint64_t kSize = 256;
    const bfr_backend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    const Instance instance{backend, messages};
    const HeadlessDevice device{instance.instance};
    bfr_buffer_desc desc{.next = nullptr,
                         .size = kSize,
                         .memory = BFR_BUFFER_MEMORY_UPLOAD,
                         .usage = {.copy_src = BFR_TRUE,
                                   .copy_dst = BFR_FALSE,
                                   .uniform = BFR_FALSE,
                                   .storage = BFR_FALSE,
                                   .index = BFR_FALSE,
                                   .indirect = BFR_FALSE},
                         .sparse = BFR_FALSE,
                         .sparse_aliased = BFR_FALSE,
                         .tile_bytes = 0};
    bfr_buffer upload{};
    REQUIRE(bfr_create_buffer(device.device, &desc, &upload) == BFR_RESULT_SUCCESS);
    desc.memory = BFR_BUFFER_MEMORY_DEVICE;
    bfr_buffer local{};
    REQUIRE(bfr_create_buffer(device.device, &desc, &local) == BFR_RESULT_SUCCESS);
    desc.memory = static_cast<bfr_buffer_memory>(kNotAValue);
    bfr_buffer stray{};
    void* data = nullptr;

    CHECK(bfr_create_buffer(device.device, &desc, &stray) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
    CHECK(stray.bits == 0);
    CHECK(bfr_map_buffer(device.device, upload, &data) == BFR_RESULT_SUCCESS);
    CHECK(data != nullptr);
    CHECK(bfr_map_buffer(device.device, local, &data) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
    CHECK(data == nullptr);
    bfr_destroy_buffer(device.device, upload);
    bfr_destroy_buffer(device.device, local);
    CHECK(bfr_map_buffer(device.device, upload, &data) == BFR_RESULT_ERROR_INVALID_HANDLE);
    bfr_destroy_buffer(device.device, upload);
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("C API creates a sampler with a slot, and refuses a value it does not have", "[c_api][gpu]") {
    const bfr_backend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    const Instance instance{backend, messages};
    const HeadlessDevice device{instance.instance};
    bfr_sampler_desc desc{.next = nullptr,
                          .mag_filter = BFR_FILTER_LINEAR,
                          .min_filter = BFR_FILTER_LINEAR,
                          .mip_filter = BFR_FILTER_NEAREST,
                          .address_u = BFR_ADDRESS_MODE_CLAMP_TO_EDGE,
                          .address_v = BFR_ADDRESS_MODE_CLAMP_TO_EDGE,
                          .address_w = BFR_ADDRESS_MODE_CLAMP_TO_EDGE,
                          .lod_bias = 0,
                          .min_lod = 0,
                          .max_lod = 0,
                          .max_anisotropy = 0,
                          .reduction = BFR_SAMPLER_REDUCTION_WEIGHTED_AVERAGE,
                          .compare_op = BFR_COMPARE_OP_ALWAYS,
                          .border_color = {},
                          .border_integer = BFR_FALSE,
                          .border_color_integer = {},
                          .unnormalized_coordinates = BFR_FALSE,
                          .lod_average = BFR_FALSE};
    bfr_sampler sampler{};
    REQUIRE(bfr_create_sampler(device.device, &desc, &sampler) == BFR_RESULT_SUCCESS);
    // Past the table, so the call is seen to write it.
    std::uint32_t slot = kDeviceSamplers;
    CHECK(bfr_get_sampler_slot(device.device, sampler, &slot) == BFR_RESULT_SUCCESS);
    CHECK(slot < kDeviceSamplers);

    desc.address_v = static_cast<bfr_address_mode>(kNotAValue);
    bfr_sampler stray{};
    CHECK(bfr_create_sampler(device.device, &desc, &stray) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
    CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("address mode v"));
    CHECK(stray.bits == 0);
    bfr_destroy_sampler(device.device, sampler);
    CHECK(bfr_get_sampler_slot(device.device, sampler, &slot) == BFR_RESULT_ERROR_INVALID_HANDLE);
    CHECK(slot == 0);
    bfr_destroy_sampler(device.device, sampler);
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("C API creates and destroys pipelines, and refuses a state value it does not have", "[c_api][gpu]") {
    const bfr_backend backend = GENERATE(eachBackend());
    // The shaders checked in are SPIR-V, which only Vulkan takes.
    if (backend != BFR_BACKEND_VULKAN) {
        SKIP("no bytecode is checked in for this backend");
    }
    constexpr std::uint32_t kWorkgroup = 64;
    std::vector<std::string> messages;
    const Instance instance{backend, messages};
    const HeadlessDevice device{instance.instance};
    const auto code = [](const auto& bytes) {
        return bfr_shader_code{.code = bytes.data(),
                               .code_size = bytes.size(),
                               .entry_point = {.data = "main", .length = 4}};
    };
    const bfr_compute_pipeline_desc computeDesc{.next = nullptr,
                                                .compute = code(bfr::test::spirv::kBindlessComp),
                                                .workgroup_size = {kWorkgroup, 1, 1},
                                                .cache = {}};
    bfr_pipeline compute{};
    REQUIRE(bfr_create_compute_pipeline(device.device, &computeDesc, &compute) == BFR_RESULT_SUCCESS);

    // Every state 0, which is every test off and every target written whole,
    // but the samples written.
    bfr_graphics_pipeline_desc graphicsDesc{};
    graphicsDesc.vertex = code(bfr::test::spirv::kBindlessVert);
    graphicsDesc.fragment = code(bfr::test::spirv::kBindlessFrag);
    graphicsDesc.color_target_count = 1;
    graphicsDesc.color_targets[0].format = BFR_FORMAT_RGBA8_UNORM;
    graphicsDesc.samples = 1;
    graphicsDesc.sample_mask = BFR_MASK_ALL;
    bfr_pipeline graphics{};
    REQUIRE(bfr_create_graphics_pipeline(device.device, &graphicsDesc, &graphics) == BFR_RESULT_SUCCESS);
    CHECK(graphics.bits != compute.bits);

    bfr_graphics_pipeline_desc stray = graphicsDesc;
    stray.raster.cull = static_cast<bfr_cull_mode>(kNotAValue);
    bfr_pipeline none{};
    CHECK(bfr_create_graphics_pipeline(device.device, &stray, &none) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
    CHECK(none.bits == 0);
    CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("cull mode"));
    stray = graphicsDesc;
    stray.color_targets[0].blend.src_color = static_cast<bfr_blend_factor>(kNotAValue);
    CHECK(bfr_create_graphics_pipeline(device.device, &stray, &none) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
    stray = graphicsDesc;
    stray.depth_stencil_format = static_cast<bfr_format>(kNotAValue);
    CHECK(bfr_create_graphics_pipeline(device.device, &stray, &none) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
    stray = graphicsDesc;
    stray.vertex.code = nullptr;
    CHECK(bfr_create_graphics_pipeline(device.device, &stray, &none) == BFR_RESULT_ERROR_INVALID_ARGUMENT);

    bfr_destroy_pipeline(device.device, compute);
    bfr_destroy_pipeline(device.device, graphics);
    bfr_destroy_pipeline(device.device, compute);
    CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("stale"));
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("C API writes a pipeline cache's bytes, saying when it wrote only part, and makes a cache of them",
          "[c_api][gpu]") {
    const bfr_backend backend = GENERATE(eachBackend());
    // The shaders checked in are SPIR-V, which only Vulkan takes.
    if (backend != BFR_BACKEND_VULKAN) {
        SKIP("no bytecode is checked in for this backend");
    }
    constexpr std::uint32_t kWorkgroup = 64;
    std::vector<std::string> messages;
    const Instance instance{backend, messages};
    const HeadlessDevice device{instance.instance};
    const bfr_pipeline_cache_desc empty{.data = nullptr, .data_size = 0};
    bfr_pipeline_cache cache{};
    REQUIRE(bfr_create_pipeline_cache(device.device, &empty, &cache) == BFR_RESULT_SUCCESS);
    const bfr_compute_pipeline_desc computeDesc{.next = nullptr,
                                                .compute = {.code = bfr::test::spirv::kBindlessComp.data(),
                                                            .code_size = bfr::test::spirv::kBindlessComp.size(),
                                                            .entry_point = {.data = "main", .length = 4}},
                                                .workgroup_size = {kWorkgroup, 1, 1},
                                                .cache = cache};
    bfr_pipeline pipeline{};
    REQUIRE(bfr_create_compute_pipeline(device.device, &computeDesc, &pipeline) == BFR_RESULT_SUCCESS);

    std::size_t size = 0;
    REQUIRE(bfr_get_pipeline_cache_data(device.device, cache, &size, nullptr) == BFR_RESULT_SUCCESS);
    REQUIRE(size > 0);
    std::vector<std::uint8_t> bytes(size);
    std::size_t part = size - 1;
    // Short of room, the backend writes part and says so, or refuses.
    const bfr_result shortResult = bfr_get_pipeline_cache_data(device.device, cache, &part, bytes.data());
    CHECK((shortResult == BFR_RESULT_INCOMPLETE || shortResult == BFR_RESULT_ERROR_DEVICE_FAILED));
    std::size_t whole = size;
    REQUIRE(bfr_get_pipeline_cache_data(device.device, cache, &whole, bytes.data()) == BFR_RESULT_SUCCESS);
    CHECK(whole == size);

    const bfr_pipeline_cache_desc saved{.data = bytes.data(), .data_size = whole};
    bfr_pipeline_cache again{};
    REQUIRE(bfr_create_pipeline_cache(device.device, &saved, &again) == BFR_RESULT_SUCCESS);
    const bfr_pipeline_cache_desc stray{.data = nullptr, .data_size = 1};
    bfr_pipeline_cache none{};
    CHECK(bfr_create_pipeline_cache(device.device, &stray, &none) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
    CHECK(none.bits == 0);

    bfr_destroy_pipeline_cache(device.device, cache);
    CHECK(bfr_get_pipeline_cache_data(device.device, cache, &size, nullptr) == BFR_RESULT_ERROR_INVALID_HANDLE);
    bfr_destroy_pipeline_cache(device.device, again);
    bfr_destroy_pipeline(device.device, pipeline);
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("C API dispatches a compute pipeline through a buffer's slot, directly or indirectly, and refuses a value it "
          "does not have",
          "[c_api][gpu]") {
    const bfr_backend backend = GENERATE(eachBackend());
    // The shaders checked in are SPIR-V, which only Vulkan takes.
    if (backend != BFR_BACKEND_VULKAN) {
        SKIP("no bytecode is checked in for this backend");
    }
    constexpr std::uint32_t kWorkgroup = 64;
    constexpr std::uint64_t kBytes = std::uint64_t{kWorkgroup} * sizeof(std::uint32_t);
    std::vector<std::string> messages;
    const Instance instance{backend, messages};
    const HeadlessDevice device{instance.instance};
    const bfr_compute_pipeline_desc computeDesc{.next = nullptr,
                                                .compute = {.code = bfr::test::spirv::kBindlessComp.data(),
                                                            .code_size = bfr::test::spirv::kBindlessComp.size(),
                                                            .entry_point = {.data = "main", .length = 4}},
                                                .workgroup_size = {kWorkgroup, 1, 1},
                                                .cache = {}};
    bfr_pipeline pipeline{};
    REQUIRE(bfr_create_compute_pipeline(device.device, &computeDesc, &pipeline) == BFR_RESULT_SUCCESS);
    bfr_buffer_desc bufferDesc{.next = nullptr,
                               .size = kBytes,
                               .memory = BFR_BUFFER_MEMORY_DEVICE,
                               .usage = {.copy_src = BFR_TRUE,
                                         .copy_dst = BFR_FALSE,
                                         .uniform = BFR_FALSE,
                                         .storage = BFR_TRUE,
                                         .index = BFR_FALSE,
                                         .indirect = BFR_FALSE},
                               .sparse = BFR_FALSE,
                               .sparse_aliased = BFR_FALSE,
                               .tile_bytes = 0};
    bfr_buffer target{};
    REQUIRE(bfr_create_buffer(device.device, &bufferDesc, &target) == BFR_RESULT_SUCCESS);
    bufferDesc.memory = BFR_BUFFER_MEMORY_READBACK;
    bufferDesc.usage = {.copy_src = BFR_FALSE,
                        .copy_dst = BFR_TRUE,
                        .uniform = BFR_FALSE,
                        .storage = BFR_FALSE,
                        .index = BFR_FALSE,
                        .indirect = BFR_FALSE};
    bfr_buffer readback{};
    REQUIRE(bfr_create_buffer(device.device, &bufferDesc, &readback) == BFR_RESULT_SUCCESS);
    std::uint32_t slot = 0;
    REQUIRE(bfr_get_buffer_slot(device.device, target, &slot) == BFR_RESULT_SUCCESS);
    CHECK(bfr_get_buffer_slot(device.device, readback, &slot) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
    REQUIRE(bfr_get_buffer_slot(device.device, target, &slot) == BFR_RESULT_SUCCESS);
    // One workgroup, read by bfr_dispatch_indirect.
    bufferDesc.memory = BFR_BUFFER_MEMORY_UPLOAD;
    bufferDesc.size = sizeof(bfr_dispatch_indirect_args);
    bufferDesc.usage = {.copy_src = BFR_FALSE,
                        .copy_dst = BFR_FALSE,
                        .uniform = BFR_FALSE,
                        .storage = BFR_FALSE,
                        .index = BFR_FALSE,
                        .indirect = BFR_TRUE};
    bfr_buffer args{};
    REQUIRE(bfr_create_buffer(device.device, &bufferDesc, &args) == BFR_RESULT_SUCCESS);
    void* argsData = nullptr;
    REQUIRE(bfr_map_buffer(device.device, args, &argsData) == BFR_RESULT_SUCCESS);
    const bfr_dispatch_indirect_args oneGroup{.x = 1, .y = 1, .z = 1};
    std::memcpy(argsData, &oneGroup, sizeof(oneGroup));
    const bool indirect = GENERATE(false, true);

    const bfr_command_pool_desc poolDesc{.next = nullptr, .queue = 0, .list_count = 1, .push_constant_updates = 1};
    bfr_command_pool pool{};
    REQUIRE(bfr_create_command_pool(device.device, &poolDesc, &pool) == BFR_RESULT_SUCCESS);
    bfr_command_list list{};
    REQUIRE(bfr_begin_commands(device.device, pool, &list) == BFR_RESULT_SUCCESS);
    CHECK(bfr_bind_pipeline(device.device, list, pipeline) == BFR_RESULT_SUCCESS);
    // The shader reads the slot from the first push constant.
    const auto slotBytes = std::bit_cast<std::array<std::uint8_t, sizeof(slot)>>(slot);
    CHECK(bfr_push_constants(device.device, list, slotBytes.data(), sizeof(slot)) == BFR_RESULT_SUCCESS);
    CHECK(bfr_push_constants(device.device, list, nullptr, sizeof(slot)) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
    CHECK(bfr_bind_index_buffer(device.device, list, target, 0, static_cast<bfr_index_type>(kNotAValue)) ==
          BFR_RESULT_ERROR_INVALID_ARGUMENT);
    CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("index type"));
    CHECK(bfr_set_topology(device.device, list, static_cast<bfr_topology>(kNotAValue)) ==
          BFR_RESULT_ERROR_INVALID_ARGUMENT);
    CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("topology"));
    if (indirect) {
        CHECK(bfr_dispatch_indirect(device.device, list, args, 0) == BFR_RESULT_SUCCESS);
    }
    else {
        CHECK(bfr_dispatch(device.device, list, 1, 1, 1) == BFR_RESULT_SUCCESS);
    }
    CHECK(bfr_barrier(device.device, list) == BFR_RESULT_SUCCESS);
    const bfr_buffer_copy whole{.src_offset = 0, .dst_offset = 0, .size = kBytes};
    CHECK(bfr_copy_buffer(device.device, list, target, readback, &whole, 1) == BFR_RESULT_SUCCESS);
    const bfr_submit_desc submit{.next = nullptr,
                                 .queue = 0,
                                 .command_lists = &list,
                                 .command_list_count = 1,
                                 .waits = nullptr,
                                 .wait_count = 0};
    bfr_token token{};
    REQUIRE(bfr_submit(device.device, &submit, &token) == BFR_RESULT_SUCCESS);
    REQUIRE(bfr_wait(device.device, &token, kTestWait) == BFR_RESULT_SUCCESS);

    void* data = nullptr;
    REQUIRE(bfr_map_buffer(device.device, readback, &data) == BFR_RESULT_SUCCESS);
    std::array<std::uint32_t, kWorkgroup> values{};
    std::memcpy(values.data(), data, kBytes);
    for (std::uint32_t i = 0; i < kWorkgroup; ++i) {
        CHECK(values.at(i) == i);
    }
    bfr_destroy_command_pool(device.device, pool);
    bfr_destroy_buffer(device.device, target);
    bfr_destroy_buffer(device.device, readback);
    bfr_destroy_buffer(device.device, args);
    bfr_destroy_pipeline(device.device, pipeline);
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("C API renders to a texture with the viewport and scissor set, and refuses a load op it does not have",
          "[c_api][gpu]") {
    constexpr std::uint32_t kSide = 8;
    const bfr_backend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    const Instance instance{backend, messages};
    const HeadlessDevice device{instance.instance};
    const bfr_texture_desc textureDesc{.next = nullptr,
                                       .dimension = BFR_TEXTURE_DIMENSION_D2,
                                       .format = BFR_FORMAT_RGBA8_UNORM,
                                       .width = kSide,
                                       .height = kSide,
                                       .depth = 1,
                                       .layers = 1,
                                       .mip_levels = 1,
                                       .samples = 1,
                                       .usage = {.copy_src = BFR_FALSE,
                                                 .copy_dst = BFR_FALSE,
                                                 .sampled = BFR_FALSE,
                                                 .storage = BFR_FALSE,
                                                 .color_attachment = BFR_TRUE,
                                                 .depth_stencil = BFR_FALSE},
                                       .sparse = BFR_FALSE,
                                       .sparse_aliased = BFR_FALSE,
                                       .tile_bytes = 0};
    bfr_texture target{};
    REQUIRE(bfr_create_texture(device.device, &textureDesc, &target) == BFR_RESULT_SUCCESS);
    const bfr_command_pool_desc poolDesc{.next = nullptr, .queue = 0, .list_count = 1, .push_constant_updates = 0};
    bfr_command_pool pool{};
    REQUIRE(bfr_create_command_pool(device.device, &poolDesc, &pool) == BFR_RESULT_SUCCESS);
    bfr_command_list list{};
    REQUIRE(bfr_begin_commands(device.device, pool, &list) == BFR_RESULT_SUCCESS);
    REQUIRE(bfr_texture_barrier(device.device, list, target, BFR_TEXTURE_LAYOUT_UNDEFINED,
                                BFR_TEXTURE_LAYOUT_COLOR_TARGET) == BFR_RESULT_SUCCESS);

    bfr_rendering_desc rendering{.next = nullptr,
                                 .color_attachment_count = 1,
                                 .color_attachments = {},
                                 .depth_stencil = {}};
    rendering.color_attachments[0] = {.texture = target,
                                      .load_op = static_cast<bfr_load_op>(kNotAValue),
                                      .store_op = BFR_STORE_OP_STORE,
                                      .clear_color = {0, 0, 0, 1},
                                      .clear_integer = BFR_FALSE,
                                      .clear_color_integer = {},
                                      .resolve_texture = {}};
    const bfr_swapchain none{};
    CHECK(bfr_begin_rendering(device.device, list, none, &rendering) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
    CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("load op"));
    rendering.color_attachments[0].load_op = BFR_LOAD_OP_CLEAR;
    rendering.depth_stencil.depth_resolve = static_cast<bfr_resolve_mode>(kNotAValue);
    CHECK(bfr_begin_rendering(device.device, list, none, &rendering) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
    CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("resolve mode"));
    rendering.depth_stencil.depth_resolve = BFR_RESOLVE_MODE_NONE;
    // A texture that is gone, resolved from.
    CHECK(bfr_resolve_texture(device.device, list, bfr_texture{}, 0, 0, target, 0, 0) ==
          BFR_RESULT_ERROR_INVALID_HANDLE);
    REQUIRE(bfr_begin_rendering(device.device, list, none, &rendering) == BFR_RESULT_SUCCESS);
    const bfr_viewport viewport{.x = 0, .y = 0, .width = kSide, .height = kSide, .min_depth = 0, .max_depth = 1};
    CHECK(bfr_set_viewport(device.device, list, &viewport) == BFR_RESULT_SUCCESS);
    const bfr_rect scissor{.x = 0, .y = 0, .width = kSide, .height = kSide};
    CHECK(bfr_set_scissor(device.device, list, &scissor) == BFR_RESULT_SUCCESS);
    CHECK(bfr_end_rendering(device.device, list) == BFR_RESULT_SUCCESS);
    const bfr_submit_desc submit{.next = nullptr,
                                 .queue = 0,
                                 .command_lists = &list,
                                 .command_list_count = 1,
                                 .waits = nullptr,
                                 .wait_count = 0};
    bfr_token token{};
    REQUIRE(bfr_submit(device.device, &submit, &token) == BFR_RESULT_SUCCESS);
    REQUIRE(bfr_wait(device.device, &token, kTestWait) == BFR_RESULT_SUCCESS);
    bfr_destroy_command_pool(device.device, pool);
    bfr_destroy_texture(device.device, target);
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("C API keeps an instance alive until its last device is destroyed", "[c_api][gpu]") {
    const bfr_backend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    auto instance = std::make_unique<Instance>(backend, messages);
    const HeadlessDevice device{instance->instance};
    instance.reset();
    const bfr_swapchain_desc desc{.next = nullptr,
                                  .window = kHeadless,
                                  .queue = 0,
                                  .image_count = device.images,
                                  .present_mode = BFR_PRESENT_MODE_FIFO,
                                  .format = BFR_FORMAT_BGRA8_UNORM,
                                  .color_space = BFR_COLOR_SPACE_SRGB,
                                  .alpha_mode = {}};
    bfr_swapchain swapchain{};

    CHECK(bfr_create_swapchain(device.device, &desc, &swapchain) == BFR_RESULT_SUCCESS);
    bfr_destroy_swapchain(device.device, swapchain);
}

TEST_CASE("C API says when a wait runs past its timeout", "[c_api][gpu]") {
    constexpr std::uint64_t kNoWait = 0;
    const bfr_backend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    const Instance instance{backend, messages};
    const HeadlessDevice device{instance.instance};
    const bfr_token ahead{.queue = 0, .value = 1};

    CHECK(bfr_wait(device.device, &ahead, kNoWait) == BFR_RESULT_ERROR_TIMEOUT);
}

TEST_CASE("C API makes a timestamp pool, gives the tick rate, and lists call timings in two calls", "[c_api][gpu]") {
    const bfr_backend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    const Instance instance{backend, messages, BFR_TRUE, BFR_TRUE};
    const HeadlessDevice device{instance.instance};
    const bfr_timestamp_pool_desc desc{.queue = 0, .count = 2};
    bfr_timestamp_pool pool{};

    REQUIRE(bfr_create_timestamp_pool(device.device, &desc, &pool) == BFR_RESULT_SUCCESS);
    CHECK(bfr_create_timestamp_pool(device.device, &desc, &pool) == BFR_RESULT_ERROR_LIMIT_REACHED);
    bfr_timestamp_rate rate{};
    CHECK(bfr_get_timestamp_rate(device.device, 0, &rate) == BFR_RESULT_SUCCESS);
    CHECK((rate.period_ns > 0 || rate.frequency != 0));
    bfr_calibrated_timestamps taken{};
    CHECK(bfr_calibrate_timestamps(device.device, 0, static_cast<bfr_host_clock>(99), &taken) ==
          BFR_RESULT_ERROR_INVALID_ARGUMENT);

    std::uint32_t count = 0;
    REQUIRE(bfr_get_call_timings(device.device, &count, nullptr) == BFR_RESULT_SUCCESS);
    CHECK(count == BFR_CALL_DESTROY_PIPELINE_CACHE + 1);
    std::vector<bfr_call_timing> timings(count);
    REQUIRE(bfr_get_call_timings(device.device, &count, timings.data()) == BFR_RESULT_SUCCESS);
    CHECK(timings.at(BFR_CALL_CREATE_TIMESTAMP_POOL).count == 2);
    std::uint32_t fewer = 1;
    CHECK(bfr_get_instance_call_timings(instance.instance, &fewer, timings.data()) == BFR_RESULT_INCOMPLETE);
    bfr_destroy_timestamp_pool(device.device, pool);
}

TEST_CASE("C API creates and destroys swapchains on one device from many threads", "[c_api][gpu]") {
    const bfr_backend backend = GENERATE(eachBackend());
    constexpr int kThreads = 4;
    constexpr int kEach = 8;
    std::vector<std::string> messages;
    const Instance instance{backend, messages};
    const HeadlessDevice device{instance.instance};
    std::atomic<int> failed{0};
    {
        std::vector<std::jthread> threads;
        threads.reserve(kThreads);
        for (int thread = 0; thread < kThreads; ++thread) {
            threads.emplace_back([&device, &failed] {
                const bfr_swapchain_desc desc{.next = nullptr,
                                              .window = kHeadless,
                                              .queue = 0,
                                              .image_count = device.images,
                                              .present_mode = BFR_PRESENT_MODE_FIFO,
                                              .format = BFR_FORMAT_BGRA8_UNORM,
                                              .color_space = BFR_COLOR_SPACE_SRGB,
                                              .alpha_mode = {}};
                for (int i = 0; i < kEach; ++i) {
                    bfr_swapchain swapchain{};
                    bfr_swapchain_images images{};
                    if (bfr_create_swapchain(device.device, &desc, &swapchain) != BFR_RESULT_SUCCESS ||
                        bfr_resize_swapchain(device.device, swapchain, kWidth, kHeight, &images) !=
                            BFR_RESULT_SUCCESS) {
                        ++failed;
                    }
                    bfr_destroy_swapchain(device.device, swapchain);
                }
            });
        }
    }

    CHECK(failed == 0);
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("C API reports when a device holds all the swapchains it can", "[c_api][gpu]") {
    const bfr_backend backend = GENERATE(eachBackend());
    // Far past any limit a device sets, so the loop always ends.
    constexpr int kTries = 4096;
    std::vector<std::string> messages;
    const Instance instance{backend, messages};
    const HeadlessDevice device{instance.instance};
    const bfr_swapchain_desc desc{.next = nullptr,
                                  .window = kHeadless,
                                  .queue = 0,
                                  .image_count = device.images,
                                  .present_mode = BFR_PRESENT_MODE_FIFO,
                                  .format = BFR_FORMAT_BGRA8_UNORM,
                                  .color_space = BFR_COLOR_SPACE_SRGB,
                                  .alpha_mode = {}};
    std::vector<bfr_swapchain> made;
    bfr_result result = BFR_RESULT_SUCCESS;

    for (int i = 0; i < kTries && result == BFR_RESULT_SUCCESS; ++i) {
        bfr_swapchain swapchain{};
        result = bfr_create_swapchain(device.device, &desc, &swapchain);
        if (result == BFR_RESULT_SUCCESS) {
            made.push_back(swapchain);
        }
    }

    CHECK(result == BFR_RESULT_ERROR_LIMIT_REACHED);
    CHECK_FALSE(made.empty());
    for (const bfr_swapchain swapchain : made) {
        bfr_destroy_swapchain(device.device, swapchain);
    }
}

TEST_CASE("C API needs a usable window on a swapchain", "[c_api][gpu]") {
    const bfr_backend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    const Instance instance{backend, messages};
    const HeadlessDevice device{instance.instance};
    bfr_swapchain swapchain{};

    const bfr_swapchain_desc none{.next = nullptr,
                                  .window = {},
                                  .queue = 0,
                                  .image_count = 0,
                                  .present_mode = BFR_PRESENT_MODE_FIFO,
                                  .format = BFR_FORMAT_BGRA8_UNORM,
                                  .color_space = BFR_COLOR_SPACE_SRGB,
                                  .alpha_mode = {}};
    CHECK(bfr_create_swapchain(device.device, &none, &swapchain) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
    CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("needs a window"));

    int display = 0;
    const bfr_swapchain_desc noWindow{.next = nullptr,
                                      .window = {.window = nullptr,
                                                 .display = &display,
                                                 .system = BFR_WINDOW_SYSTEM_DEFAULT},
                                      .queue = 0,
                                      .image_count = 0,
                                      .present_mode = BFR_PRESENT_MODE_FIFO,
                                      .format = BFR_FORMAT_BGRA8_UNORM,
                                      .color_space = BFR_COLOR_SPACE_SRGB,
                                      .alpha_mode = {}};
    CHECK(bfr_create_swapchain(device.device, &noWindow, &swapchain) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
    CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("window value"));

    const bfr_swapchain_desc unknown{.next = nullptr,
                                     .window = {.window = nullptr,
                                                .display = nullptr,
                                                .system = static_cast<bfr_window_system>(99)},
                                     .queue = 0,
                                     .image_count = 0,
                                     .present_mode = BFR_PRESENT_MODE_FIFO,
                                     .format = BFR_FORMAT_BGRA8_UNORM,
                                     .color_space = BFR_COLOR_SPACE_SRGB,
                                     .alpha_mode = {}};
    CHECK(bfr_create_swapchain(device.device, &unknown, &swapchain) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
    CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("not one BFR has"));
    CHECK(swapchain.bits == 0);
}

TEST_CASE("C API rejects a min_severity that is not a severity, 0 included", "[c_api][unit]") {
    constexpr auto kNotASeverity = static_cast<bfr_message_severity>(9);
    const bfr_message_severity severity = GENERATE_COPY(kNotASeverity, bfr_message_severity{});
    // The severity filters the callback's messages, so it is read with one.
    const bfr_instance_desc desc{.next = nullptr,
                                 .backend = BFR_BACKEND_VULKAN,
                                 .validation = BFR_FALSE,
                                 .message_callback = [](bfr_message_severity, bfr_string, void*) {},
                                 .min_severity = severity,
                                 .message_user_data = nullptr,
                                 .application_name = {},
                                 .application_version = 0,
                                 .engine_name = {},
                                 .engine_version = 0,
                                 .call_timings = BFR_FALSE};
    bfr_instance* instance = nullptr;

    CHECK(bfr_create_instance(&desc, &instance) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
    CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring(
                                std::format("min_severity {} is not a severity", static_cast<int>(severity))));
    CHECK(instance == nullptr);
}

TEST_CASE("C API sends BFR's messages from min_severity up", "[c_api][gpu]") {
    if (!bfr::core::kLoggingEnabled) {
        SKIP("logging is compiled out");
    }
    const bfr_backend backend = GENERATE(eachBackend());
    const auto created = [backend](bfr_message_severity least) {
        std::vector<std::string> messages;
        const bfr_instance_desc desc{.next = nullptr,
                                     .backend = backend,
                                     .validation = BFR_FALSE,
                                     .message_callback = record,
                                     .min_severity = least,
                                     .message_user_data = &messages,
                                     .application_name = {},
                                     .application_version = 0,
                                     .engine_name = {},
                                     .engine_version = 0,
                                     .call_timings = BFR_FALSE};
        bfr_instance* instance = nullptr;
        REQUIRE(bfr_create_instance(&desc, &instance) == BFR_RESULT_SUCCESS);
        bfr_destroy_instance(instance);
        return messages;
    };

    // Every backend says what it found at info; with nothing wrong, nothing at warning.
    CHECK_FALSE(created(BFR_MESSAGE_SEVERITY_INFO).empty());
    CHECK_THAT(created(BFR_MESSAGE_SEVERITY_WARNING), Catch::Matchers::IsEmpty());
}

namespace {

std::string_view text(bfr_string string) {
    return {string.data, string.length};
}

// What is wrong with the caps an adapter of backend listed as listed wrote:
// each name whose check fails. What the numbers may be is the RHI tests'
// question; here the C structs carry what was listed.
std::vector<std::string> capsProblems(const bfr_caps& caps, const bfr_adapter_info& listed, bfr_backend backend) {
    const std::vector<std::pair<bool, std::string>> checks{
        {text(caps.info.name) == text(listed.name), "name"},
        {caps.info.type == listed.type, "type"},
        {caps.info.backend == backend, "backend"},
    };
    std::vector<std::string> problems;
    for (const auto& [held, name] : checks) {
        if (!held) {
            problems.push_back(name);
        }
    }
    return problems;
}

// The first adapter that presents to a headless window. An adapter may not
// present to one at all: NVIDIA's driver has no headless surfaces.
std::uint32_t headlessAdapter(bfr_instance* instance) {
    std::uint32_t count = 0;
    REQUIRE(bfr_get_adapters(instance, &count, nullptr) == BFR_RESULT_SUCCESS);
    const bfr_swapchain_query headless{.next = nullptr, .window = kHeadless};
    for (std::uint32_t adapter = 0; adapter < count; ++adapter) {
        bfr_swapchain_caps caps{};
        const bfr_result result = bfr_get_swapchain_caps(instance, adapter, &headless, &caps);
        if (result == BFR_RESULT_SUCCESS) {
            return adapter;
        }
        INFO(lastError());
        REQUIRE(result == BFR_RESULT_ERROR_PRESENT_UNSUPPORTED);
    }
    FAIL("no adapter presents to a headless window");
    return 0;
}

} // namespace

TEST_CASE("C API writes each adapter's caps", "[c_api][gpu]") {
    const bfr_backend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    const Instance instance{backend, messages};
    std::uint32_t count = 0;
    REQUIRE(bfr_get_adapters(instance.instance, &count, nullptr) == BFR_RESULT_SUCCESS);
    std::vector<bfr_adapter_info> adapters(count);
    REQUIRE(bfr_get_adapters(instance.instance, &count, adapters.data()) == BFR_RESULT_SUCCESS);

    for (std::uint32_t adapter = 0; adapter < count; ++adapter) {
        bfr_caps caps{};
        const bfr_result result = bfr_get_caps(instance.instance, adapter, &caps);

        INFO(lastError());
        CHECK(result == BFR_RESULT_SUCCESS);
        CHECK_THAT(capsProblems(caps, adapters[adapter], backend), Catch::Matchers::IsEmpty());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("C API refuses caps queries with null arguments", "[c_api][gpu]") {
    const bfr_backend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    const Instance instance{backend, messages};
    bfr_caps caps{};
    bfr_swapchain_caps swapchainCaps{};

    const std::vector<bfr_result> results{
        bfr_get_caps(nullptr, 0, &caps),
        bfr_get_caps(instance.instance, 0, nullptr),
        bfr_get_formats(instance.instance, 0, nullptr, nullptr),
        bfr_get_swapchain_caps(instance.instance, 0, nullptr, &swapchainCaps),
        bfr_get_swapchain_formats(instance.instance, 0, nullptr, nullptr, nullptr),
        bfr_get_budget(nullptr, nullptr),
    };

    CHECK(std::ranges::all_of(results, [](bfr_result result) { return result == BFR_RESULT_ERROR_INVALID_ARGUMENT; }));
    CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("must not be null"));
}

TEST_CASE("C API refuses caps queries about an adapter it does not list", "[c_api][gpu]") {
    const bfr_backend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    const Instance instance{backend, messages};
    std::uint32_t count = 0;
    REQUIRE(bfr_get_adapters(instance.instance, &count, nullptr) == BFR_RESULT_SUCCESS);
    bfr_caps caps{};
    std::uint32_t formats = 0;
    const bfr_swapchain_query surface{.next = nullptr, .window = kHeadless};
    bfr_swapchain_caps swapchainCaps{};

    const std::vector<bfr_result> results{
        bfr_get_formats(instance.instance, count, &formats, nullptr),
        bfr_get_swapchain_caps(instance.instance, count, &surface, &swapchainCaps),
        bfr_get_swapchain_formats(instance.instance, count, &surface, &formats, nullptr),
        bfr_get_caps(instance.instance, count, &caps),
    };

    CHECK(std::ranges::all_of(results, [](bfr_result result) { return result == BFR_RESULT_ERROR_INVALID_ARGUMENT; }));
    CHECK_THAT(lastError(), Catch::Matchers::Equals(std::format("adapter {} of {}", count, count)));
}

TEST_CASE("C API lists every format in two calls, and says when the array is too small", "[c_api][gpu]") {
    const bfr_backend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    const Instance instance{backend, messages};
    std::uint32_t count = 0;
    REQUIRE(bfr_get_formats(instance.instance, 0, &count, nullptr) == BFR_RESULT_SUCCESS);
    REQUIRE(count > 100);
    std::vector<bfr_format_caps> formats(count);

    std::uint32_t written = count;
    CHECK(bfr_get_formats(instance.instance, 0, &written, formats.data()) == BFR_RESULT_SUCCESS);
    CHECK(written == count);
    CHECK(formats.front().format == BFR_FORMAT_R8_UNORM);
    CHECK(formats.back().format == BFR_FORMAT_ASTC_12X12_FLOAT);

    std::uint32_t few = 2;
    CHECK(bfr_get_formats(instance.instance, 0, &few, formats.data()) == BFR_RESULT_INCOMPLETE);
    CHECK(few == 2);
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("C API reports what a swapchain for a headless window may be", "[c_api][gpu]") {
    const bfr_backend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    const Instance instance{backend, messages};
    const std::uint32_t adapter = headlessAdapter(instance.instance);
    const bfr_swapchain_query desc{.next = nullptr, .window = kHeadless};
    bfr_swapchain_caps caps{};
    std::uint32_t count = 0;

    CHECK(bfr_get_swapchain_caps(instance.instance, adapter, &desc, &caps) == BFR_RESULT_SUCCESS);
    CHECK(bfr_get_swapchain_formats(instance.instance, adapter, &desc, &count, nullptr) == BFR_RESULT_SUCCESS);

    INFO(lastError());
    CHECK(caps.min_image_count >= 1);
    CHECK(caps.fifo == BFR_TRUE);
    REQUIRE(count > 0);
    std::vector<bfr_swapchain_format> formats(count);
    CHECK(bfr_get_swapchain_formats(instance.instance, adapter, &desc, &count, formats.data()) == BFR_RESULT_SUCCESS);
    CHECK(std::ranges::any_of(formats, [](const bfr_swapchain_format& format) {
        return format.color_space == BFR_COLOR_SPACE_SRGB;
    }));
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("C API needs a usable window to ask about a surface", "[c_api][gpu]") {
    const bfr_backend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    const Instance instance{backend, messages};
    bfr_swapchain_caps caps{};

    const bfr_swapchain_query none{.next = nullptr, .window = {}};
    CHECK(bfr_get_swapchain_caps(instance.instance, 0, &none, &caps) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
    CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("needs a window"));

    const bfr_swapchain_query unknown{.next = nullptr,
                                      .window = {.window = nullptr,
                                                 .display = nullptr,
                                                 .system = static_cast<bfr_window_system>(99)}};
    CHECK(bfr_get_swapchain_caps(instance.instance, 0, &unknown, &caps) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
    CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("not one BFR has"));
}

TEST_CASE("C API gives a device's memory budget or says the driver can't", "[c_api][gpu]") {
    const bfr_backend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    const Instance instance{backend, messages};
    const HeadlessDevice device{instance.instance};
    bfr_caps caps{};
    REQUIRE(bfr_get_caps(instance.instance, device.adapter, &caps) == BFR_RESULT_SUCCESS);
    bfr_memory_budget budget{};

    const bfr_result result = bfr_get_budget(device.device, &budget);

    INFO(lastError());
    const bool unsupported = result == BFR_RESULT_ERROR_UNSUPPORTED;
    CHECK((result == BFR_RESULT_SUCCESS || unsupported));
    CHECK((unsupported || budget.heap_count == caps.memory.heap_count));
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

// Validation stays off: its layer allocates through the operator new this
// program counts.
TEST_CASE("C API allocates and frees nothing on the heap frame after frame", "[c_api][gpu]") {
    if (!bfr::test::AllocationCounter::counts()) {
        SKIP("a sanitizer replaces operator new, so allocations are not counted");
    }
    const bfr_backend backend = GENERATE(eachBackend());
    // More than the 16 a submit once held in place before it went to the heap;
    // the first two render to the swapchain.
    constexpr std::uint32_t kLists = 20;
    constexpr std::uint32_t kRendering = 2;
    constexpr int kFrames = 8;
    // The swapchain's image, cleared.
    constexpr bfr_rendering_desc kClear{.next = nullptr,
                                        .color_attachment_count = 1,
                                        .color_attachments = {{.texture = {},
                                                               .load_op = BFR_LOAD_OP_CLEAR,
                                                               .store_op = BFR_STORE_OP_STORE,
                                                               .clear_color = {0, 0, 0, 1},
                                                               .clear_integer = BFR_FALSE,
                                                               .clear_color_integer = {},
                                                               .resolve_texture = {}}},
                                        .depth_stencil = {}};
    std::vector<std::string> messages;
    {
        const Instance instance{backend, messages, BFR_FALSE};
        const HeadlessDevice device{instance.instance};
        const bfr_swapchain_desc swapchainDesc{.next = nullptr,
                                               .window = kHeadless,
                                               .queue = 0,
                                               .image_count = device.images,
                                               .present_mode = BFR_PRESENT_MODE_FIFO,
                                               .format = BFR_FORMAT_BGRA8_UNORM,
                                               .color_space = BFR_COLOR_SPACE_SRGB,
                                               .alpha_mode = {}};
        bfr_swapchain swapchain{};
        REQUIRE(bfr_create_swapchain(device.device, &swapchainDesc, &swapchain) == BFR_RESULT_SUCCESS);
        bfr_swapchain_images images{};
        REQUIRE(bfr_resize_swapchain(device.device, swapchain, kWidth, kHeight, &images) == BFR_RESULT_SUCCESS);
        const bfr_command_pool_desc poolDesc{.next = nullptr,
                                             .queue = 0,
                                             .list_count = kLists,
                                             .push_constant_updates = 0};
        bfr_command_pool pool{};
        REQUIRE(bfr_create_command_pool(device.device, &poolDesc, &pool) == BFR_RESULT_SUCCESS);

        // The first call that failed, kept without a string so keeping it allocates nothing.
        const char* failed = nullptr;
        std::size_t allocations = 0;
        std::size_t frees = 0;
        {
            const bfr::test::AllocationCounter counter;
            const auto check = [&failed](const char* call, bfr_result result) {
                if (result < 0 && failed == nullptr) {
                    failed = call;
                }
                return result >= 0;
            };
            bfr_token last{};
            bfr_swapchain_status status{};
            for (int frame = 0; frame < kFrames && failed == nullptr; ++frame) {
                check("bfr_wait", bfr_wait(device.device, &last, kTestWait));
                check("bfr_reset_command_pool", bfr_reset_command_pool(device.device, pool));
                check("bfr_acquire", bfr_acquire(device.device, swapchain, kTestWait, &status));
                std::array<bfr_command_list, kLists> lists{};
                for (std::uint32_t index = 0; index < kLists; ++index) {
                    check("bfr_begin_commands", bfr_begin_commands(device.device, pool, &lists.at(index)));
                    if (index < kRendering) {
                        check("bfr_begin_rendering",
                              bfr_begin_rendering(device.device, lists.at(index), swapchain, &kClear));
                        check("bfr_end_rendering", bfr_end_rendering(device.device, lists.at(index)));
                    }
                }
                const bfr_submit_desc submit{.next = nullptr,
                                             .queue = 0,
                                             .command_lists = lists.data(),
                                             .command_list_count = kLists,
                                             .waits = &last,
                                             .wait_count = frame == 0 ? 0U : 1U};
                bfr_token token{};
                if (check("bfr_submit", bfr_submit(device.device, &submit, &token))) {
                    last = token;
                }
                check("bfr_present", bfr_present(device.device, swapchain, &status));
                bfr_bool complete = BFR_FALSE;
                check("bfr_poll", bfr_poll(device.device, &last, &complete));
            }
            check("bfr_wait", bfr_wait(device.device, &last, kTestWait));
            allocations = counter.allocations();
            frees = counter.frees();
        }
        INFO("first failure: " << (failed == nullptr ? "none" : failed) << " " << lastError());
        CHECK(failed == nullptr);
        CHECK(allocations == 0);
        CHECK(frees == 0);
        bfr_destroy_command_pool(device.device, pool);
        bfr_destroy_swapchain(device.device, swapchain);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

// bfr_acquire on a swapchain with no images, as on a minimised window, fails
// every frame; recording the failure as the last error allocates nothing.
TEST_CASE("C API allocates nothing when a frame-path call fails frame after frame", "[c_api][gpu]") {
    if (!bfr::test::AllocationCounter::counts()) {
        SKIP("a sanitizer replaces operator new, so allocations are not counted");
    }
    const bfr_backend backend = GENERATE(eachBackend());
    constexpr int kFrames = 8;
    std::vector<std::string> messages;
    {
        const Instance instance{backend, messages, BFR_FALSE};
        const HeadlessDevice device{instance.instance};
        const bfr_swapchain_desc desc{.next = nullptr,
                                      .window = kHeadless,
                                      .queue = 0,
                                      .image_count = device.images,
                                      .present_mode = BFR_PRESENT_MODE_FIFO,
                                      .format = BFR_FORMAT_BGRA8_UNORM,
                                      .color_space = BFR_COLOR_SPACE_SRGB,
                                      .alpha_mode = {}};
        bfr_swapchain swapchain{};
        REQUIRE(bfr_create_swapchain(device.device, &desc, &swapchain) == BFR_RESULT_SUCCESS);

        int refused = 0;
        std::size_t allocations = 0;
        std::size_t frees = 0;
        {
            const bfr::test::AllocationCounter counter;
            bfr_swapchain_status status{};
            for (int frame = 0; frame < kFrames; ++frame) {
                refused += bfr_acquire(device.device, swapchain, kTestWait, &status) == BFR_RESULT_ERROR_OUT_OF_DATE
                               ? 1
                               : 0;
                refused -= bfr_get_last_error().length == 0 ? 1 : 0;
            }
            allocations = counter.allocations();
            frees = counter.frees();
        }
        CHECK(refused == kFrames);
        CHECK(allocations == 0);
        CHECK(frees == 0);
        bfr_destroy_swapchain(device.device, swapchain);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}
