#include "bfr/bfr.h"
#include "bfr/core/log.h"

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
#include <cstdint>
#include <format>
#include <iterator>
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
// A chain entry of a type no desc takes.
constexpr bfr_chain kStray{.next = nullptr, .type = BFR_STRUCT_TYPE_FORCE32};

constexpr std::uint32_t kWidth = 64;
constexpr std::uint32_t kHeight = 48;

// A value no BFR enum has.
constexpr int kNotAValue = 99;

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

// An instance on backend with validation on, recording its messages, destroyed
// at scope end.
struct Instance {
    Instance(bfr_backend backend, std::vector<std::string>& messages) {
        const bfr_instance_desc desc{.next = nullptr,
                                     .backend = backend,
                                     .validation = BFR_TRUE,
                                     .message_callback = record,
                                     .min_severity = BFR_MESSAGE_SEVERITY_WARNING,
                                     .message_user_data = &messages};
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
            const bfr_device_desc desc{.next = nullptr, .adapter = adapter, .window = kHeadless};
            const bfr_result result = bfr_create_device(instance, &desc, &device);
            INFO(lastError());
            REQUIRE((result == BFR_RESULT_SUCCESS || result == BFR_RESULT_ERROR_PRESENT_UNSUPPORTED));
            if (device != nullptr) {
                this->adapter = adapter;
            }
        }
        REQUIRE(device != nullptr);
        const bfr_surface_desc surface{.next = nullptr, .window = kHeadless};
        bfr_surface_caps caps{};
        REQUIRE(bfr_get_surface_caps(instance, adapter, &surface, &caps) == BFR_RESULT_SUCCESS);
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
                                 .message_user_data = nullptr};
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
                                     .message_user_data = nullptr};
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
                                 .message_user_data = nullptr};
    bfr_instance* instance = nullptr;

    CHECK(bfr_create_instance(&desc, &instance) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_CASE("C API destroy calls take null and stale handles", "[c_api][unit]") {
    bfr_destroy_instance(nullptr);
    bfr_destroy_device(nullptr);
    bfr_destroy_swapchain(nullptr, bfr_swapchain{.bits = 0});
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
                                      .image_count = kImages,
                                      .present_mode = BFR_PRESENT_MODE_FIFO,
                                      .format = {},
                                      .color_space = {},
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
    bfr_bool bfr_surface_caps::* allowed;
    const char* name;
};

constexpr std::array kModes{
    Mode{.mode = BFR_PRESENT_MODE_FIFO, .allowed = &bfr_surface_caps::fifo, .name = "fifo"},
    Mode{.mode = BFR_PRESENT_MODE_FIFO_RELAXED, .allowed = &bfr_surface_caps::fifo_relaxed, .name = "fifo_relaxed"},
    Mode{.mode = BFR_PRESENT_MODE_MAILBOX, .allowed = &bfr_surface_caps::mailbox, .name = "mailbox"},
    Mode{.mode = BFR_PRESENT_MODE_IMMEDIATE, .allowed = &bfr_surface_caps::immediate, .name = "immediate"},
};

// Makes a swapchain on device in mode, which caps allows, and checks it is made.
void checkPresentMode(bfr_device* device, const bfr_surface_caps& caps, const Mode& mode) {
    const bfr_swapchain_desc desc{.next = nullptr,
                                  .window = kHeadless,
                                  .image_count = caps.min_image_count,
                                  .present_mode = mode.mode,
                                  .format = {},
                                  .color_space = {},
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
        const bfr_surface_desc surface{.next = nullptr, .window = kHeadless};
        bfr_surface_caps caps{};
        REQUIRE(bfr_get_surface_caps(instance.instance, device.adapter, &surface, &caps) == BFR_RESULT_SUCCESS);

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
        const bfr_swapchain_desc defaults{.next = nullptr,
                                          .window = kHeadless,
                                          .image_count = device.images,
                                          .present_mode = {},
                                          .format = {},
                                          .color_space = {},
                                          .alpha_mode = {}};
        bfr_swapchain swapchain{};
        REQUIRE(bfr_create_swapchain(device.device, &defaults, &swapchain) == BFR_RESULT_SUCCESS);
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
                                                                .image_count = 0,
                                                                .present_mode = static_cast<bfr_present_mode>(
                                                                    kNotAValue),
                                                                .format = {},
                                                                .color_space = {},
                                                                .alpha_mode = {}},
                                             bfr_swapchain_desc{.next = nullptr,
                                                                .window = kHeadless,
                                                                .image_count = 0,
                                                                .present_mode = {},
                                                                .format = static_cast<bfr_format>(kNotAValue),
                                                                .color_space = {},
                                                                .alpha_mode = {}},
                                             bfr_swapchain_desc{.next = nullptr,
                                                                .window = kHeadless,
                                                                .image_count = 0,
                                                                .present_mode = {},
                                                                .format = {},
                                                                .color_space = static_cast<bfr_color_space>(kNotAValue),
                                                                .alpha_mode = {}},
                                             bfr_swapchain_desc{.next = nullptr,
                                                                .window = kHeadless,
                                                                .image_count = 0,
                                                                .present_mode = {},
                                                                .format = {},
                                                                .color_space = {},
                                                                .alpha_mode = static_cast<bfr_alpha_mode>(kNotAValue)});
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
                                  .image_count = device.images,
                                  .present_mode = BFR_PRESENT_MODE_FIFO,
                                  .format = {},
                                  .color_space = {},
                                  .alpha_mode = {}};
    bfr_swapchain swapchain{};
    REQUIRE(bfr_create_swapchain(device.device, &desc, &swapchain) == BFR_RESULT_SUCCESS);
    bfr_destroy_swapchain(device.device, swapchain);
    bfr_swapchain_images images{};

    CHECK(bfr_resize_swapchain(device.device, swapchain, kWidth, kHeight, &images) == BFR_RESULT_ERROR_INVALID_HANDLE);
    bfr_destroy_swapchain(device.device, swapchain);
    CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("stale"));
}

TEST_CASE("C API keeps an instance alive until its last device is destroyed", "[c_api][gpu]") {
    const bfr_backend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    auto instance = std::make_unique<Instance>(backend, messages);
    const HeadlessDevice device{instance->instance};
    instance.reset();
    const bfr_swapchain_desc desc{.next = nullptr,
                                  .window = kHeadless,
                                  .image_count = device.images,
                                  .present_mode = BFR_PRESENT_MODE_FIFO,
                                  .format = {},
                                  .color_space = {},
                                  .alpha_mode = {}};
    bfr_swapchain swapchain{};

    CHECK(bfr_create_swapchain(device.device, &desc, &swapchain) == BFR_RESULT_SUCCESS);
    bfr_destroy_swapchain(device.device, swapchain);
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
                                              .image_count = device.images,
                                              .present_mode = BFR_PRESENT_MODE_FIFO,
                                              .format = {},
                                              .color_space = {},
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
                                  .image_count = device.images,
                                  .present_mode = BFR_PRESENT_MODE_FIFO,
                                  .format = {},
                                  .color_space = {},
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
                                  .image_count = 0,
                                  .present_mode = BFR_PRESENT_MODE_FIFO,
                                  .format = {},
                                  .color_space = {},
                                  .alpha_mode = {}};
    CHECK(bfr_create_swapchain(device.device, &none, &swapchain) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
    CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("needs a window"));

    int display = 0;
    const bfr_swapchain_desc noWindow{.next = nullptr,
                                      .window = {.window = nullptr,
                                                 .display = &display,
                                                 .system = BFR_WINDOW_SYSTEM_DEFAULT},
                                      .image_count = 0,
                                      .present_mode = BFR_PRESENT_MODE_FIFO,
                                      .format = {},
                                      .color_space = {},
                                      .alpha_mode = {}};
    CHECK(bfr_create_swapchain(device.device, &noWindow, &swapchain) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
    CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("window value"));

    const bfr_swapchain_desc unknown{.next = nullptr,
                                     .window = {.window = nullptr,
                                                .display = nullptr,
                                                .system = static_cast<bfr_window_system>(99)},
                                     .image_count = 0,
                                     .present_mode = BFR_PRESENT_MODE_FIFO,
                                     .format = {},
                                     .color_space = {},
                                     .alpha_mode = {}};
    CHECK(bfr_create_swapchain(device.device, &unknown, &swapchain) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
    CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("not one BFR has"));
    CHECK(swapchain.bits == 0);
}

TEST_CASE("C API rejects a min_severity that is not a severity", "[c_api][unit]") {
    constexpr auto kNotASeverity = static_cast<bfr_message_severity>(9);
    const bfr_instance_desc desc{.next = nullptr,
                                 .backend = BFR_BACKEND_VULKAN,
                                 .validation = BFR_FALSE,
                                 .message_callback = nullptr,
                                 .min_severity = kNotASeverity,
                                 .message_user_data = nullptr};
    bfr_instance* instance = nullptr;

    CHECK(bfr_create_instance(&desc, &instance) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
    CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("min_severity 9"));
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
                                     .message_user_data = &messages};
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
    const bfr_surface_desc headless{.next = nullptr, .window = kHeadless};
    for (std::uint32_t adapter = 0; adapter < count; ++adapter) {
        bfr_surface_caps caps{};
        const bfr_result result = bfr_get_surface_caps(instance, adapter, &headless, &caps);
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
    bfr_surface_caps surfaceCaps{};

    const std::vector<bfr_result> results{
        bfr_get_caps(nullptr, 0, &caps),
        bfr_get_caps(instance.instance, 0, nullptr),
        bfr_get_formats(instance.instance, 0, nullptr, nullptr),
        bfr_get_surface_caps(instance.instance, 0, nullptr, &surfaceCaps),
        bfr_get_surface_formats(instance.instance, 0, nullptr, nullptr, nullptr),
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
    const bfr_surface_desc surface{.next = nullptr, .window = kHeadless};
    bfr_surface_caps surfaceCaps{};

    const std::vector<bfr_result> results{
        bfr_get_formats(instance.instance, count, &formats, nullptr),
        bfr_get_surface_caps(instance.instance, count, &surface, &surfaceCaps),
        bfr_get_surface_formats(instance.instance, count, &surface, &formats, nullptr),
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
    const bfr_surface_desc desc{.next = nullptr, .window = kHeadless};
    bfr_surface_caps caps{};
    std::uint32_t count = 0;

    CHECK(bfr_get_surface_caps(instance.instance, adapter, &desc, &caps) == BFR_RESULT_SUCCESS);
    CHECK(bfr_get_surface_formats(instance.instance, adapter, &desc, &count, nullptr) == BFR_RESULT_SUCCESS);

    INFO(lastError());
    CHECK(caps.min_image_count >= 1);
    CHECK(caps.fifo == BFR_TRUE);
    REQUIRE(count > 0);
    std::vector<bfr_surface_format> formats(count);
    CHECK(bfr_get_surface_formats(instance.instance, adapter, &desc, &count, formats.data()) == BFR_RESULT_SUCCESS);
    CHECK(std::ranges::any_of(formats, [](const bfr_surface_format& format) {
        return format.color_space == BFR_COLOR_SPACE_SRGB;
    }));
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("C API needs a usable window to ask about a surface", "[c_api][gpu]") {
    const bfr_backend backend = GENERATE(eachBackend());
    std::vector<std::string> messages;
    const Instance instance{backend, messages};
    bfr_surface_caps caps{};

    const bfr_surface_desc none{.next = nullptr, .window = {}};
    CHECK(bfr_get_surface_caps(instance.instance, 0, &none, &caps) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
    CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("needs a window"));

    const bfr_surface_desc unknown{.next = nullptr,
                                   .window = {.window = nullptr,
                                              .display = nullptr,
                                              .system = static_cast<bfr_window_system>(99)}};
    CHECK(bfr_get_surface_caps(instance.instance, 0, &unknown, &caps) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
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
