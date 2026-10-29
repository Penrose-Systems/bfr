#include "bfr.hpp"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_container_properties.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kWidth = 64;
constexpr std::uint32_t kHeight = 48;
constexpr bfr::Window kHeadless{.system = bfr::WindowSystem::Headless};

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
    return result.has_value() ? std::string{} : result.error().message;
}

bfr::Instance validatedInstance(Messages& messages) {
    auto instance = bfr::Instance::create(
        {.backend = bfr::Backend::Vulkan, .validation = true, .messageCallback = messages.callback()});
    INFO(failureOf(instance));
    REQUIRE(instance.has_value());
    return std::move(*instance);
}

// A device on the first adapter that presents headless.
bfr::Device headlessDevice(const bfr::Instance& instance) {
    const auto adapters = instance.adapters();
    REQUIRE(adapters.has_value());
    for (std::uint32_t adapter = 0; adapter < adapters->size(); ++adapter) {
        auto device = instance.createDevice({.adapter = adapter, .window = kHeadless});
        if (device.has_value()) {
            return std::move(*device);
        }
        INFO(device.error().message);
        REQUIRE(device.error().code == bfr::Result::ErrorPresentUnsupported);
    }
    FAIL("no adapter presents headless");
    return {};
}

} // namespace

TEST_CASE("C++ API reports a failed call's result and message", "[cpp_api][unit]") {
    const auto instance = bfr::Instance::create({.backend = bfr::Backend::D3d12});

    REQUIRE_FALSE(instance.has_value());
    CHECK(instance.error().code == bfr::Result::ErrorBackendUnavailable);
    CHECK_THAT(instance.error().message, Catch::Matchers::ContainsSubstring("not in this build"));
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
    Messages messages;
    {
        const bfr::Instance instance = validatedInstance(messages);

        const auto adapters = instance.adapters();

        REQUIRE(adapters.has_value());
        REQUIRE_FALSE(adapters->empty());
        CHECK_FALSE(adapters->front().name.empty());
    }
    CHECK_THAT(messages.recorded(), Catch::Matchers::IsEmpty());
}

TEST_CASE("C++ API builds, resizes and destroys a headless swapchain", "[cpp_api][gpu]") {
    // More than the default and than lavapipe's minimum.
    constexpr std::uint32_t kImages = 5;
    Messages messages;
    {
        const bfr::Instance instance = validatedInstance(messages);
        const bfr::Device device = headlessDevice(instance);
        auto swapchain = device.createSwapchain({.window = kHeadless, .imageCount = kImages});
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

TEST_CASE("C++ API passes on the C API's refusal of a swapchain without a window", "[cpp_api][gpu]") {
    Messages messages;
    const bfr::Instance instance = validatedInstance(messages);
    const bfr::Device device = headlessDevice(instance);

    const auto swapchain = device.createSwapchain({.imageCount = 0});

    REQUIRE_FALSE(swapchain.has_value());
    CHECK(swapchain.error().code == bfr::Result::ErrorInvalidArgument);
    CHECK_THAT(swapchain.error().message, Catch::Matchers::ContainsSubstring("needs a window"));
}

TEST_CASE("C++ API moves objects without destroying what they own", "[cpp_api][gpu]") {
    Messages messages;
    {
        const bfr::Instance instance = validatedInstance(messages);
        bfr::Device device = headlessDevice(instance);
        auto created = device.createSwapchain({.window = kHeadless, .imageCount = 0});
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
    Messages messages;
    {
        const bfr::Instance instance = validatedInstance(messages);
        bfr::Swapchain swapchain;
        {
            const bfr::Device device = headlessDevice(instance);
            auto created = device.createSwapchain({.window = kHeadless, .imageCount = 0});
            REQUIRE(created.has_value());
            swapchain = std::move(*created);
        }

        CHECK(swapchain.resize(kWidth, kHeight).has_value());
    }
    CHECK_THAT(messages.recorded(), Catch::Matchers::IsEmpty());
}

TEST_CASE("C++ API presents as the mode asked for or says the window doesn't allow it", "[cpp_api][gpu]") {
    Messages messages;
    {
        const bfr::Instance instance = validatedInstance(messages);
        const bfr::Device device = headlessDevice(instance);
        const auto fifo = device.createSwapchain({.window = kHeadless, .presentMode = bfr::PresentMode::Fifo});
        const auto immediate = device.createSwapchain(
            {.window = kHeadless, .presentMode = bfr::PresentMode::Immediate});

        INFO(failureOf(fifo));
        CHECK(fifo.has_value());
        if (!immediate.has_value()) {
            CHECK(immediate.error().code == bfr::Result::ErrorUnsupported);
            CHECK_THAT(immediate.error().message, Catch::Matchers::ContainsSubstring("immediate"));
        }
    }
    CHECK_THAT(messages.recorded(), Catch::Matchers::IsEmpty());
}

TEST_CASE("C++ API keeps the message callback alive while a device lives", "[cpp_api][gpu]") {
    auto state = std::make_shared<int>(0);
    const std::weak_ptr<int> watched = state;
    auto instance = bfr::Instance::create(
        {.backend = bfr::Backend::Vulkan,
         .messageCallback = [state](bfr::MessageSeverity /*severity*/, std::string_view /*message*/) {
         }});
    state.reset();
    REQUIRE(instance.has_value());
    bfr::Device device = headlessDevice(*instance);

    *instance = bfr::Instance{};
    CHECK_FALSE(watched.expired());

    device = bfr::Device{};
    CHECK(watched.expired());
}

TEST_CASE("C++ API reports when a device holds all the swapchains it can", "[cpp_api][gpu]") {
    // Far past any limit a device sets, so the loop always ends.
    constexpr int kTries = 4096;
    Messages messages;
    const bfr::Instance instance = validatedInstance(messages);
    const bfr::Device device = headlessDevice(instance);
    std::vector<bfr::Swapchain> made;
    bfr::Expected<bfr::Swapchain> swapchain;

    for (int i = 0; i < kTries && swapchain.has_value(); ++i) {
        swapchain = device.createSwapchain({.window = kHeadless, .imageCount = 0});
        if (swapchain.has_value()) {
            made.push_back(std::move(*swapchain));
        }
    }

    REQUIRE_FALSE(swapchain.has_value());
    CHECK(swapchain.error().code == bfr::Result::ErrorLimitReached);
    CHECK_FALSE(made.empty());
}

TEST_CASE("C++ API creates and destroys swapchains on one device from many threads", "[cpp_api][gpu]") {
    constexpr int kThreads = 4;
    constexpr int kEach = 8;
    Messages messages;
    {
        const bfr::Instance instance = validatedInstance(messages);
        const bfr::Device device = headlessDevice(instance);
        std::atomic<int> failed{0};
        {
            std::vector<std::jthread> threads;
            threads.reserve(kThreads);
            for (int thread = 0; thread < kThreads; ++thread) {
                threads.emplace_back([&device, &failed] {
                    for (int i = 0; i < kEach; ++i) {
                        auto swapchain = device.createSwapchain({.window = kHeadless, .imageCount = 0});
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
    Messages messages;
    {
        const bfr::Instance instance = validatedInstance(messages);
        const auto adapters = instance.adapters();
        REQUIRE(adapters.has_value());
        const auto cpu = std::ranges::find(*adapters, bfr::AdapterType::Cpu, &bfr::AdapterInfo::type);
        REQUIRE(cpu != adapters->end());
        const auto adapter = static_cast<std::uint32_t>(cpu - adapters->begin());

        const auto caps = instance.caps(adapter);
        const auto formats = instance.formats(adapter);
        const auto surface = instance.surfaceCaps(adapter, {.window = kHeadless});
        const auto surfaceFormats = instance.surfaceFormats(adapter, {.window = kHeadless});

        INFO(failureOf(caps) << failureOf(formats) << failureOf(surface) << failureOf(surfaceFormats));
        REQUIRE(caps.has_value());
        CHECK(caps->info.name == cpu->name);
        CHECK(caps->info.type == bfr::AdapterType::Cpu);
        CHECK(caps->limits.maxTexture2d >= 4096);
        CHECK(caps->compute.maxWorkgroupSize[0] >= 128);
        CHECK(caps->memory.heapCount > 0);
        CHECK(caps->memory.heaps[0].size > 0);
        REQUIRE(formats.has_value());
        CHECK(formats->front().format == bfr::Format::R8Unorm);
        REQUIRE(surface.has_value());
        CHECK(surface->fifo);
        REQUIRE(surfaceFormats.has_value());
        CHECK_FALSE(surfaceFormats->empty());
    }
    CHECK_THAT(messages.recorded(), Catch::Matchers::IsEmpty());
}

TEST_CASE("C++ API gives a device's memory budget or says the driver can't", "[cpp_api][gpu]") {
    Messages messages;
    {
        const bfr::Instance instance = validatedInstance(messages);
        const bfr::Device device = headlessDevice(instance);

        const auto budget = device.budget();

        if (budget.has_value()) {
            CHECK(budget->heapCount > 0);
        }
        else {
            CHECK(budget.error().code == bfr::Result::ErrorUnsupported);
        }
    }
    CHECK_THAT(messages.recorded(), Catch::Matchers::IsEmpty());
}
