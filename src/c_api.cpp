// The C API in bfr.h, over the C++ implementation. No exception crosses it:
// each entry point turns one into a result.

#include "bfr.h"
#include "device.h"
#include "error.h"
#include "rhi.h"
#include "rhi_config.h"
#include "swapchain.h"
#include "window.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <format>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

// Shares the Rhi with its devices, which keep it alive: garbage-collected
// wrappers destroy them in no fixed order.
struct bfr_instance {
    std::shared_ptr<bfr::rhi::Rhi> rhi;
};

// The Rhi comes first so that it outlives the device it made.
struct bfr_device {
    std::shared_ptr<bfr::rhi::Rhi> rhi;
    std::unique_ptr<bfr::rhi::Device> device;
};

namespace {

using bfr::core::ErrorCode;
using bfr::core::window::NativeWindow;

// What bfr_swapchain_desc.image_count = 0 asks for.
constexpr std::uint32_t kDefaultSwapchainImages = 3;

// The calling thread's last error, which bfr_get_last_error returns.
std::string& lastError() {
    thread_local std::string error;
    return error;
}

// Records message as the calling thread's last error and returns result.
bfr_result fail(bfr_result result, std::string_view message) noexcept {
    try {
        lastError() = message;
    } catch (...) {
        lastError().clear();
    }
    return result;
}

bfr_result resultFor(ErrorCode code) {
    switch (code) {
        case ErrorCode::RhiInstanceFailed:
            return BFR_RESULT_ERROR_INSTANCE_FAILED;
        case ErrorCode::RhiLoaderFailed:
            return BFR_RESULT_ERROR_LOADER_FAILED;
        case ErrorCode::RhiValidationUnavailable:
            return BFR_RESULT_ERROR_VALIDATION_UNAVAILABLE;
        case ErrorCode::RhiAdapterOutOfRange:
            return BFR_RESULT_ERROR_INVALID_ARGUMENT;
        case ErrorCode::RhiDeviceFailed:
            return BFR_RESULT_ERROR_DEVICE_FAILED;
        case ErrorCode::RhiNoAdapter:
            return BFR_RESULT_ERROR_NO_ADAPTER;
        case ErrorCode::RhiSurfaceFailed:
            return BFR_RESULT_ERROR_SURFACE_FAILED;
        case ErrorCode::RhiInvalidHandle:
            return BFR_RESULT_ERROR_INVALID_HANDLE;
        case ErrorCode::RhiPresentUnsupported:
            return BFR_RESULT_ERROR_PRESENT_UNSUPPORTED;
        case ErrorCode::RhiSwapchainFailed:
            return BFR_RESULT_ERROR_SWAPCHAIN_FAILED;
        case ErrorCode::RhiLimitReached:
            return BFR_RESULT_ERROR_LIMIT_REACHED;
        // The display and config are not behind the C API.
        case ErrorCode::DisplayAlreadyInitialised:
        case ErrorCode::DisplayNotInitialised:
        case ErrorCode::DisplayBackendFailed:
        case ErrorCode::DisplayWindowFailed:
        case ErrorCode::DisplayFullscreenFailed:
        case ErrorCode::ConfigLoadFailed:
            break;
    }
    return BFR_RESULT_ERROR_INTERNAL;
}

bfr_result fail(const bfr::core::Error& error) noexcept {
    return fail(resultFor(error.code), error.message);
}

// Runs body, which returns a bfr_result, turning an exception into one.
template <class Body>
bfr_result guarded(Body&& body) noexcept {
    try {
        return std::forward<Body>(body)();
    } catch (const std::bad_alloc&) {
        return fail(BFR_RESULT_ERROR_OUT_OF_MEMORY, "out of memory");
    } catch (const std::exception& exception) {
        return fail(BFR_RESULT_ERROR_INTERNAL, exception.what());
    } catch (...) {
        return fail(BFR_RESULT_ERROR_INTERNAL, "an unknown exception");
    }
}

// Runs body, which returns nothing. The destroy calls have no result to
// report an exception through, so it is recorded as the last error only.
template <class Body>
void quietly(Body&& body) noexcept {
    try {
        std::forward<Body>(body)();
    } catch (const std::bad_alloc&) {
        fail(BFR_RESULT_ERROR_OUT_OF_MEMORY, "out of memory");
    } catch (const std::exception& exception) {
        fail(BFR_RESULT_ERROR_INTERNAL, exception.what());
    } catch (...) {
        fail(BFR_RESULT_ERROR_INTERNAL, "an unknown exception");
    }
}

// A C handle is a Handle's bits, so the two convert by copy.
bfr_swapchain toC(bfr::rhi::SwapchainHandle handle) {
    return {.bits = handle.bits};
}

bfr::rhi::SwapchainHandle fromC(bfr_swapchain handle) {
    return bfr::rhi::SwapchainHandle::fromBits(handle.bits);
}

// The extension struct that chain heads. Every one starts with its bfr_chain,
// so the two share an address; chain->type has said which struct it is.
template <class Extension>
const Extension* extension(const bfr_chain* chain) {
    static_assert(offsetof(Extension, chain) == 0);
    return reinterpret_cast<const Extension*>(chain); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
}

// The window chained to next: none, one, or an error for a second window or a
// struct this desc does not take.
std::optional<NativeWindow> windowIn(const bfr_chain* next, std::string& error) {
    NativeWindow window;
    for (; next != nullptr; next = next->next) {
        const bool first = std::holds_alternative<std::monostate>(window);
        switch (next->type) {
            case BFR_STRUCT_TYPE_WINDOW_XLIB: {
                const auto* xlib = extension<bfr_window_xlib>(next);
                window = bfr::core::window::Xlib{.display = xlib->display, .window = xlib->window};
                break;
            }
            case BFR_STRUCT_TYPE_WINDOW_WAYLAND: {
                const auto* wayland = extension<bfr_window_wayland>(next);
                window = bfr::core::window::Wayland{.display = wayland->display, .surface = wayland->surface};
                break;
            }
            case BFR_STRUCT_TYPE_WINDOW_WIN32: {
                const auto* win32 = extension<bfr_window_win32>(next);
                window = bfr::core::window::Win32{.instance = win32->instance, .window = win32->window};
                break;
            }
            case BFR_STRUCT_TYPE_WINDOW_METAL: {
                const auto* metal = extension<bfr_window_metal>(next);
                window = bfr::core::window::Metal{.layer = metal->layer};
                break;
            }
            case BFR_STRUCT_TYPE_WINDOW_HEADLESS:
                window = bfr::core::window::Headless{};
                break;
            case BFR_STRUCT_TYPE_FORCE32:
            default:
                error = std::format("struct type {} cannot be chained here", static_cast<std::int64_t>(next->type));
                return std::nullopt;
        }
        if (!first) {
            error = "more than one window is chained";
            return std::nullopt;
        }
    }
    return window;
}

bfr_adapter_type toC(bfr::rhi::AdapterType type) {
    switch (type) {
        case bfr::rhi::AdapterType::Discrete:
            return BFR_ADAPTER_TYPE_DISCRETE;
        case bfr::rhi::AdapterType::Integrated:
            return BFR_ADAPTER_TYPE_INTEGRATED;
        case bfr::rhi::AdapterType::Virtual:
            return BFR_ADAPTER_TYPE_VIRTUAL;
        case bfr::rhi::AdapterType::Cpu:
            return BFR_ADAPTER_TYPE_CPU;
        case bfr::rhi::AdapterType::Other:
            break;
    }
    return BFR_ADAPTER_TYPE_OTHER;
}

} // namespace

extern "C" {

bfr_string bfr_get_last_error(void) {
    const std::string& error = lastError();
    return {.data = error.data(), .length = error.size()};
}

bfr_result bfr_create_instance(const bfr_instance_desc* desc, bfr_instance** instance) {
    return guarded([&] {
        if (desc == nullptr || instance == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "desc and instance must not be null");
        }
        *instance = nullptr;
        if (desc->next != nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "bfr_instance_desc takes no chained structs");
        }
        if (desc->backend != BFR_BACKEND_VULKAN) {
            return fail(BFR_RESULT_ERROR_BACKEND_UNAVAILABLE,
                        std::format("backend {} is not in this build", static_cast<std::int64_t>(desc->backend)));
        }
        bfr::rhi::RhiDesc rhiDesc;
        if (desc->message_callback != nullptr) {
            rhiDesc.onValidationMessage = [callback = desc->message_callback,
                                           userData = desc->message_user_data](bool isError, std::string_view message) {
                callback(isError ? BFR_MESSAGE_SEVERITY_ERROR : BFR_MESSAGE_SEVERITY_WARNING,
                         {.data = message.data(), .length = message.size()}, userData);
            };
        }
        auto rhi = bfr::rhi::Rhi::create(rhiDesc, bfr::rhi::RhiConfig{.validation = desc->validation != BFR_FALSE});
        if (!rhi) {
            return fail(rhi.error());
        }
        // The caller owns it until bfr_destroy_instance.
        *instance = std::make_unique<bfr_instance>(bfr_instance{.rhi = std::move(*rhi)}).release();
        return BFR_RESULT_SUCCESS;
    });
}

void bfr_destroy_instance(bfr_instance* instance) {
    quietly([&] { std::unique_ptr<bfr_instance>{instance}.reset(); });
}

bfr_result bfr_instance_get_adapters(bfr_instance* instance, uint32_t* count, bfr_adapter_info* adapters) {
    return guarded([&] {
        if (instance == nullptr || count == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "instance and count must not be null");
        }
        const std::span<const bfr::rhi::Adapter> listed = instance->rhi->adapters();
        const auto total = static_cast<std::uint32_t>(listed.size());
        if (adapters == nullptr) {
            *count = total;
            return BFR_RESULT_SUCCESS;
        }
        const std::uint32_t written = std::min(*count, total);
        for (std::uint32_t i = 0; i < written; ++i) {
            const bfr::rhi::Adapter& adapter = listed[i];
            adapters[i] = {.name = {.data = adapter.name.data(), .length = adapter.name.size()},
                           .type = toC(adapter.type)};
        }
        *count = written;
        return written < total ? BFR_RESULT_INCOMPLETE : BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_instance_create_device(bfr_instance* instance, const bfr_device_desc* desc, bfr_device** device) {
    return guarded([&] {
        if (instance == nullptr || desc == nullptr || device == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "instance, desc and device must not be null");
        }
        *device = nullptr;
        std::string error;
        const auto window = windowIn(desc->next, error);
        if (!window) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, error);
        }
        auto created = instance->rhi->createDevice(desc->adapter, *window);
        if (!created) {
            return fail(created.error());
        }
        // The caller owns it until bfr_destroy_device.
        *device =
            std::make_unique<bfr_device>(bfr_device{.rhi = instance->rhi, .device = std::move(*created)}).release();
        return BFR_RESULT_SUCCESS;
    });
}

void bfr_destroy_device(bfr_device* device) {
    quietly([&] { std::unique_ptr<bfr_device>{device}.reset(); });
}

bfr_result bfr_device_create_swapchain(bfr_device* device, const bfr_swapchain_desc* desc, bfr_swapchain* swapchain) {
    return guarded([&] {
        if (device == nullptr || desc == nullptr || swapchain == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device, desc and swapchain must not be null");
        }
        *swapchain = {};
        std::string error;
        const auto window = windowIn(desc->next, error);
        if (!window) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, error);
        }
        if (std::holds_alternative<std::monostate>(*window)) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "bfr_swapchain_desc needs a window chained to it");
        }
        const std::uint32_t images = desc->image_count == 0 ? kDefaultSwapchainImages : desc->image_count;
        const auto created = device->device->createSwapchain(*window, images);
        if (!created) {
            return fail(created.error());
        }
        *swapchain = toC(*created);
        return BFR_RESULT_SUCCESS;
    });
}

bfr_result bfr_device_resize_swapchain(bfr_device* device, bfr_swapchain swapchain, uint32_t width, uint32_t height,
                                       bfr_swapchain_images* images) {
    return guarded([&] {
        if (device == nullptr || images == nullptr) {
            return fail(BFR_RESULT_ERROR_INVALID_ARGUMENT, "device and images must not be null");
        }
        const auto resized = device->device->resizeSwapchain(fromC(swapchain), width, height);
        if (!resized) {
            return fail(resized.error());
        }
        *images = {.width = resized->width, .height = resized->height, .count = resized->count};
        return BFR_RESULT_SUCCESS;
    });
}

void bfr_device_destroy_swapchain(bfr_device* device, bfr_swapchain swapchain) {
    quietly([&] {
        if (device == nullptr) {
            return;
        }
        // A stale handle is already gone, which is all this call promises; the
        // reason stays readable for anyone debugging.
        if (const auto destroyed = device->device->destroySwapchain(fromC(swapchain)); !destroyed) {
            lastError() = destroyed.error().message;
        }
    });
}

} // extern "C"
