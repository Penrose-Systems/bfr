#include "bfr.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_container_properties.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace {

std::string lastError() {
    const bfr_string error = bfr_get_last_error();
    return {error.data, error.length};
}

// Records every message the instance reports.
void record(bfr_message_severity /*severity*/, bfr_string message, void* userData) {
    static_cast<std::vector<std::string>*>(userData)->emplace_back(message.data, message.length);
}

constexpr bfr_window_headless kHeadless{.chain = {.next = nullptr, .type = BFR_STRUCT_TYPE_WINDOW_HEADLESS}};

constexpr std::uint32_t kWidth = 64;
constexpr std::uint32_t kHeight = 48;

// An instance with validation on, recording its messages, destroyed at scope end.
struct Instance {
    explicit Instance(std::vector<std::string>& messages) {
        const bfr_instance_desc desc{.next = nullptr,
                                     .backend = BFR_BACKEND_VULKAN,
                                     .validation = BFR_TRUE,
                                     .message_callback = record,
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
        REQUIRE(bfr_instance_get_adapters(instance, &count, nullptr) == BFR_RESULT_SUCCESS);
        for (std::uint32_t adapter = 0; adapter < count && device == nullptr; ++adapter) {
            const bfr_device_desc desc{.next = &kHeadless.chain, .adapter = adapter};
            const bfr_result result = bfr_instance_create_device(instance, &desc, &device);
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
                                 .message_user_data = nullptr};
    bfr_instance* instance = nullptr;

    CHECK(bfr_create_instance(&desc, &instance) == BFR_RESULT_ERROR_BACKEND_UNAVAILABLE);
    CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("not in this build"));
}

TEST_CASE("C API rejects a struct chained where it does not belong", "[c_api][unit]") {
    const bfr_instance_desc desc{.next = &kHeadless.chain,
                                 .backend = BFR_BACKEND_VULKAN,
                                 .validation = BFR_FALSE,
                                 .message_callback = nullptr,
                                 .message_user_data = nullptr};
    bfr_instance* instance = nullptr;

    CHECK(bfr_create_instance(&desc, &instance) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_CASE("C API destroy calls take null and stale handles", "[c_api][unit]") {
    bfr_destroy_instance(nullptr);
    bfr_destroy_device(nullptr);
    bfr_device_destroy_swapchain(nullptr, bfr_swapchain{.bits = 0});
}

TEST_CASE("C API lists adapters in two calls", "[c_api][gpu]") {
    std::vector<std::string> messages;
    const Instance instance{messages};
    std::uint32_t count = 0;
    REQUIRE(bfr_instance_get_adapters(instance.instance, &count, nullptr) == BFR_RESULT_SUCCESS);
    REQUIRE(count > 0);
    std::vector<bfr_adapter_info> adapters(count);
    std::uint32_t written = count;

    CHECK(bfr_instance_get_adapters(instance.instance, &written, adapters.data()) == BFR_RESULT_SUCCESS);

    CHECK(written == count);
    CHECK(adapters.front().name.length > 0);
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("C API says when the caller's adapter array is too small", "[c_api][gpu]") {
    std::vector<std::string> messages;
    const Instance instance{messages};
    bfr_adapter_info adapter{};
    std::uint32_t written = 0;

    CHECK(bfr_instance_get_adapters(instance.instance, &written, &adapter) == BFR_RESULT_INCOMPLETE);
    CHECK(written == 0);
}

TEST_CASE("C API builds, resizes and destroys a headless swapchain", "[c_api][gpu]") {
    // More than the default and than lavapipe's minimum.
    constexpr std::uint32_t kImages = 5;
    std::vector<std::string> messages;
    {
        const Instance instance{messages};
        const HeadlessDevice device{instance.instance};
        const bfr_swapchain_desc desc{.next = &kHeadless.chain, .image_count = kImages};
        bfr_swapchain swapchain{};
        REQUIRE(bfr_device_create_swapchain(device.device, &desc, &swapchain) == BFR_RESULT_SUCCESS);
        CHECK(swapchain.bits != 0);

        bfr_swapchain_images images{};
        const bfr_result resized = bfr_device_resize_swapchain(device.device, swapchain, kWidth, kHeight, &images);

        INFO(lastError());
        REQUIRE(resized == BFR_RESULT_SUCCESS);
        CHECK(images.width == kWidth);
        CHECK(images.height == kHeight);
        CHECK(images.count >= kImages);

        bfr_device_destroy_swapchain(device.device, swapchain);
    }
    CHECK_THAT(messages, Catch::Matchers::IsEmpty());
}

TEST_CASE("C API reports a destroyed swapchain as invalid and destroys it again quietly", "[c_api][gpu]") {
    std::vector<std::string> messages;
    const Instance instance{messages};
    const HeadlessDevice device{instance.instance};
    const bfr_swapchain_desc desc{.next = &kHeadless.chain, .image_count = 0};
    bfr_swapchain swapchain{};
    REQUIRE(bfr_device_create_swapchain(device.device, &desc, &swapchain) == BFR_RESULT_SUCCESS);
    bfr_device_destroy_swapchain(device.device, swapchain);
    bfr_swapchain_images images{};

    CHECK(bfr_device_resize_swapchain(device.device, swapchain, kWidth, kHeight, &images) ==
          BFR_RESULT_ERROR_INVALID_HANDLE);
    bfr_device_destroy_swapchain(device.device, swapchain);
    CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("stale"));
}

TEST_CASE("C API needs exactly one window on a swapchain", "[c_api][gpu]") {
    std::vector<std::string> messages;
    const Instance instance{messages};
    const HeadlessDevice device{instance.instance};
    bfr_swapchain swapchain{};

    const bfr_swapchain_desc none{.next = nullptr, .image_count = 0};
    CHECK(bfr_device_create_swapchain(device.device, &none, &swapchain) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
    CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("needs a window"));

    const bfr_window_headless second{.chain = {.next = &kHeadless.chain, .type = BFR_STRUCT_TYPE_WINDOW_HEADLESS}};
    const bfr_swapchain_desc two{.next = &second.chain, .image_count = 0};
    CHECK(bfr_device_create_swapchain(device.device, &two, &swapchain) == BFR_RESULT_ERROR_INVALID_ARGUMENT);
    CHECK_THAT(lastError(), Catch::Matchers::ContainsSubstring("more than one window"));
    CHECK(swapchain.bits == 0);
}
