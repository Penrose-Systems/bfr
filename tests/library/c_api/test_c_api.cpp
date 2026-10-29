#include "bfr.h"
#include "log.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_container_properties.hpp>
#include <catch2/matchers/catch_matchers_quantifiers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <format>
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

// An instance with validation on, recording its messages, destroyed at scope end.
struct Instance {
    explicit Instance(std::vector<std::string>& messages) {
        const bfr_instance_desc desc{.next = nullptr,
                                     .backend = BFR_BACKEND_VULKAN,
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
        }
        REQUIRE(device != nullptr);
    }
    ~HeadlessDevice() { bfr_destroy_device(device); }
    HeadlessDevice(const HeadlessDevice&) = delete;
    HeadlessDevice& operator=(const HeadlessDevice&) = delete;
    HeadlessDevice(HeadlessDevice&&) = delete;
    HeadlessDevice& operator=(HeadlessDevice&&) = delete;

    bfr_device* device = nullptr;
};

} // namespace

TEST_CASE("C API rejects null arguments and says which", "[c_api][unit]") {
    bfr_instance* instance = nullptr;

    CHECK(bfr_create_instance(nullptr, &instance) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
    CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("must not be null"));
    CHECK(instance == nullptr);
}

TEST_CASE("C API reports a backend this build lacks", "[c_api][unit]") {
    const bfr_instance_desc desc{.next = nullptr,
                                 .backend = BFR_BACKEND_D3D12,
                                 .validation = BFR_FALSE,
                                 .message_callback = nullptr,
                                 .min_severity = BFR_MESSAGE_SEVERITY_WARNING,
                                 .message_user_data = nullptr};
    bfr_instance* instance = nullptr;

    CHECK(bfr_create_instance(&desc, &instance) == BFR_RESULT_ERROR_BACKEND_UNAVAILABLE);
    CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("not in this build"));
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
    std::vector<std::string> messages;
    const Instance instance{messages};
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
    std::vector<std::string> messages;
    const Instance instance{messages};
    bfr_adapter_info adapter{};
    std::uint32_t written = 0;

    CHECK(bfr_get_adapters(instance.instance, &written, &adapter) == BFR_RESULT_INCOMPLETE);
    CHECK(written == 0);
}

TEST_CASE("C API builds, resizes and destroys a headless swapchain", "[c_api][gpu]") {
    // More than the default and than lavapipe's minimum.
    constexpr std::uint32_t kImages = 5;
    std::vector<std::string> messages;
    {
        const Instance instance{messages};
        const HeadlessDevice device{instance.instance};
        const bfr_swapchain_desc desc{.next = nullptr, .window = kHeadless, .image_count = kImages};
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

TEST_CASE("C API reports a destroyed swapchain as invalid and destroys it again quietly", "[c_api][gpu]") {
    std::vector<std::string> messages;
    const Instance instance{messages};
    const HeadlessDevice device{instance.instance};
    const bfr_swapchain_desc desc{.next = nullptr, .window = kHeadless, .image_count = 0};
    bfr_swapchain swapchain{};
    REQUIRE(bfr_create_swapchain(device.device, &desc, &swapchain) == BFR_RESULT_SUCCESS);
    bfr_destroy_swapchain(device.device, swapchain);
    bfr_swapchain_images images{};

    CHECK(bfr_resize_swapchain(device.device, swapchain, kWidth, kHeight, &images) == BFR_RESULT_ERROR_INVALID_HANDLE);
    bfr_destroy_swapchain(device.device, swapchain);
    CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("stale"));
}

TEST_CASE("C API keeps an instance alive until its last device is destroyed", "[c_api][gpu]") {
    std::vector<std::string> messages;
    auto instance = std::make_unique<Instance>(messages);
    const HeadlessDevice device{instance->instance};
    instance.reset();
    const bfr_swapchain_desc desc{.next = nullptr, .window = kHeadless, .image_count = 0};
    bfr_swapchain swapchain{};

    CHECK(bfr_create_swapchain(device.device, &desc, &swapchain) == BFR_RESULT_SUCCESS);
    bfr_destroy_swapchain(device.device, swapchain);
}

TEST_CASE("C API creates and destroys swapchains on one device from many threads", "[c_api][gpu]") {
    constexpr int kThreads = 4;
    constexpr int kEach = 8;
    std::vector<std::string> messages;
    const Instance instance{messages};
    const HeadlessDevice device{instance.instance};
    std::atomic<int> failed{0};
    {
        std::vector<std::jthread> threads;
        threads.reserve(kThreads);
        for (int thread = 0; thread < kThreads; ++thread) {
            threads.emplace_back([&device, &failed] {
                const bfr_swapchain_desc desc{.next = nullptr, .window = kHeadless, .image_count = 0};
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
    // Far past any limit a device sets, so the loop always ends.
    constexpr int kTries = 4096;
    std::vector<std::string> messages;
    const Instance instance{messages};
    const HeadlessDevice device{instance.instance};
    const bfr_swapchain_desc desc{.next = nullptr, .window = kHeadless, .image_count = 0};
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
    std::vector<std::string> messages;
    const Instance instance{messages};
    const HeadlessDevice device{instance.instance};
    bfr_swapchain swapchain{};

    const bfr_swapchain_desc none{.next = nullptr, .window = {}, .image_count = 0};
    CHECK(bfr_create_swapchain(device.device, &none, &swapchain) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
    CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("needs a window"));

    int display = 0;
    const bfr_swapchain_desc noWindow{.next = nullptr,
                                      .window = {.window = nullptr,
                                                 .display = &display,
                                                 .system = BFR_WINDOW_SYSTEM_DEFAULT},
                                      .image_count = 0};
    CHECK(bfr_create_swapchain(device.device, &noWindow, &swapchain) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
    CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("window value"));

    const bfr_swapchain_desc unknown{.next = nullptr,
                                     .window = {.window = nullptr,
                                                .display = nullptr,
                                                .system = static_cast<bfr_window_system>(99)},
                                     .image_count = 0};
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
    const auto created = [](bfr_message_severity least) {
        std::vector<std::string> messages;
        const bfr_instance_desc desc{.next = nullptr,
                                     .backend = BFR_BACKEND_VULKAN,
                                     .validation = BFR_FALSE,
                                     .message_callback = record,
                                     .min_severity = least,
                                     .message_user_data = &messages};
        bfr_instance* instance = nullptr;
        REQUIRE(bfr_create_instance(&desc, &instance) == BFR_RESULT_SUCCESS);
        bfr_destroy_instance(instance);
        return messages;
    };

    CHECK_THAT(created(BFR_MESSAGE_SEVERITY_INFO),
               Catch::Matchers::AnyMatch(Catch::Matchers::StartsWith("Vulkan loader ")));
    CHECK_THAT(created(BFR_MESSAGE_SEVERITY_WARNING),
               Catch::Matchers::NoneMatch(Catch::Matchers::StartsWith("Vulkan loader ")));
}

namespace {

std::string_view text(bfr_string string) {
    return {string.data, string.length};
}

// What is wrong with the caps a Vulkan adapter listed as listed wrote: each
// name whose check fails.
std::vector<std::string> capsProblems(const bfr_caps& caps, const bfr_adapter_info& listed) {
    constexpr std::uint32_t kMinTexture2d = 4096;
    constexpr std::uint32_t kMinWorkgroupSize = 128;
    const std::vector<std::pair<bool, std::string>> checks{
        {text(caps.info.name) == text(listed.name), "name"},
        {caps.info.type == listed.type, "type"},
        {caps.info.backend == BFR_BACKEND_VULKAN, "backend"},
        {caps.info.api_major == 1, "api_major"},
        {caps.info.driver_name.length > 0, "driver_name"},
        {caps.limits.max_texture_2d >= kMinTexture2d, "max_texture_2d"},
        {caps.compute.max_workgroup_size[0] >= kMinWorkgroupSize, "max_workgroup_size"},
        {caps.memory.heap_count > 0 && caps.memory.heaps[0].size > 0, "heaps"},
        {caps.memory.type_count > 0, "types"},
        {caps.queues.graphics >= 1, "graphics queues"},
    };
    std::vector<std::string> problems;
    for (const auto& [held, name] : checks) {
        if (!held) {
            problems.push_back(name);
        }
    }
    return problems;
}

// The CPU adapter, lavapipe, which presents headless on every test machine.
std::uint32_t cpuAdapter(bfr_instance* instance) {
    std::uint32_t count = 0;
    REQUIRE(bfr_get_adapters(instance, &count, nullptr) == BFR_RESULT_SUCCESS);
    std::vector<bfr_adapter_info> adapters(count);
    REQUIRE(bfr_get_adapters(instance, &count, adapters.data()) == BFR_RESULT_SUCCESS);
    for (std::uint32_t adapter = 0; adapter < count; ++adapter) {
        if (adapters[adapter].type == BFR_ADAPTER_TYPE_CPU) {
            return adapter;
        }
    }
    FAIL("no CPU adapter");
    return 0;
}

} // namespace

TEST_CASE("C API writes each adapter's caps", "[c_api][gpu]") {
    std::vector<std::string> messages;
    const Instance instance{messages};
    std::uint32_t count = 0;
    REQUIRE(bfr_get_adapters(instance.instance, &count, nullptr) == BFR_RESULT_SUCCESS);
    std::vector<bfr_adapter_info> adapters(count);
    REQUIRE(bfr_get_adapters(instance.instance, &count, adapters.data()) == BFR_RESULT_SUCCESS);

    for (std::uint32_t adapter = 0; adapter < count; ++adapter) {
        bfr_caps caps{};
        const bfr_result result = bfr_get_caps(instance.instance, adapter, &caps);

        INFO(lastError());
        CHECK(result == BFR_RESULT_SUCCESS);
        CHECK_THAT(capsProblems(caps, adapters[adapter]), Catch::Matchers::IsEmpty());
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("C API refuses caps queries with null arguments", "[c_api][gpu]") {
    std::vector<std::string> messages;
    const Instance instance{messages};
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
    std::vector<std::string> messages;
    const Instance instance{messages};
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
    std::vector<std::string> messages;
    const Instance instance{messages};
    std::uint32_t count = 0;
    REQUIRE(bfr_get_formats(instance.instance, 0, &count, nullptr) == BFR_RESULT_SUCCESS);
    REQUIRE(count > 100);
    std::vector<bfr_format_caps> formats(count);

    std::uint32_t written = count;
    CHECK(bfr_get_formats(instance.instance, 0, &written, formats.data()) == BFR_RESULT_SUCCESS);
    CHECK(written == count);
    CHECK(formats.front().format == BFR_FORMAT_R8_UNORM);
    CHECK(formats.back().format == BFR_FORMAT_ASTC_12X12_FLOAT);
    const auto rgba8 = std::ranges::find(formats, BFR_FORMAT_RGBA8_UNORM, &bfr_format_caps::format);
    REQUIRE(rgba8 != formats.end());
    CHECK(rgba8->color_attachment == BFR_TRUE);
    CHECK(rgba8->sampled == BFR_TRUE);

    std::uint32_t few = 2;
    CHECK(bfr_get_formats(instance.instance, 0, &few, formats.data()) == BFR_RESULT_INCOMPLETE);
    CHECK(few == 2);
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("C API reports what a swapchain for a headless window may be", "[c_api][gpu]") {
    std::vector<std::string> messages;
    const Instance instance{messages};
    const std::uint32_t adapter = cpuAdapter(instance.instance);
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
    std::vector<std::string> messages;
    const Instance instance{messages};
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
    std::vector<std::string> messages;
    const Instance instance{messages};
    const HeadlessDevice device{instance.instance};
    bfr_memory_budget budget{};

    const bfr_result result = bfr_get_budget(device.device, &budget);

    INFO(lastError());
    const bool unsupported = result == BFR_RESULT_ERROR_UNSUPPORTED;
    CHECK((result == BFR_RESULT_SUCCESS || unsupported));
    CHECK((unsupported || (budget.heap_count > 0 && budget.heaps[0].budget > 0)));
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}
