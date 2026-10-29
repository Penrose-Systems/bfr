#include "vulkan_surface.h"

#include "error.h"
#include "instance_table.h"
#include "vulkan_result.h"
#include "window.h"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <array>
#include <bit>
#include <format>
#include <span>
#include <string_view>
#include <utility>
#include <variant>

#ifdef VK_USE_PLATFORM_XLIB_KHR
    #include <vulkan/vulkan_xlib.h>
    #include <X11/X.h>
    #include <X11/Xlib.h>
#endif
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
    #include <vulkan/vulkan_wayland.h>
#endif
#ifdef VK_USE_PLATFORM_WIN32_KHR
    #include <vulkan/vulkan_win32.h>
    #include <windows.h>
#endif
#ifdef VK_USE_PLATFORM_METAL_EXT
    #include <vulkan/vulkan_metal.h>
#endif

namespace bfr::rhi::vulkan {

namespace {

// The build's platform defines (CMakeLists.txt) decide which window systems
// these cover.
constexpr std::array kSurfaceExtensions{
#ifdef VK_USE_PLATFORM_XLIB_KHR
    VK_KHR_XLIB_SURFACE_EXTENSION_NAME,
#endif
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
    VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME,
#endif
#ifdef VK_USE_PLATFORM_WIN32_KHR
    VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
#endif
#ifdef VK_USE_PLATFORM_METAL_EXT
    VK_EXT_METAL_SURFACE_EXTENSION_NAME,
#endif
    VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME,
};

using Made = core::Result<VkSurfaceKHR>;

// Each window system's surface, made only when the instance enabled its
// extension.
class Maker {
public:
    Maker(VkInstance instance, std::span<const char* const> extensions)
        : instance_(instance), extensions_(extensions) {}

    Made operator()(std::monostate /*none*/) const {
        return core::failure(core::ErrorCode::RhiSurfaceFailed, "there is no window");
    }

#ifdef VK_USE_PLATFORM_XLIB_KHR
    Made operator()(const core::window::Xlib& window) const {
        VkXlibSurfaceCreateInfoKHR info{};
        info.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
        info.dpy = static_cast<Display*>(window.display);
        info.window = static_cast<Window>(window.window);
        return make<PFN_vkCreateXlibSurfaceKHR>(VK_KHR_XLIB_SURFACE_EXTENSION_NAME, "vkCreateXlibSurfaceKHR",
                                                [&](auto create, VkSurfaceKHR* surface) {
                                                    return create(instance_, &info, nullptr, surface);
                                                });
    }
#endif

#ifdef VK_USE_PLATFORM_WAYLAND_KHR
    Made operator()(const core::window::Wayland& window) const {
        VkWaylandSurfaceCreateInfoKHR info{};
        info.sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR;
        info.display = static_cast<wl_display*>(window.display);
        info.surface = static_cast<wl_surface*>(window.surface);
        return make<PFN_vkCreateWaylandSurfaceKHR>(VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME, "vkCreateWaylandSurfaceKHR",
                                                   [&](auto create, VkSurfaceKHR* surface) {
                                                       return create(instance_, &info, nullptr, surface);
                                                   });
    }
#endif

#ifdef VK_USE_PLATFORM_WIN32_KHR
    Made operator()(const core::window::Win32& window) const {
        VkWin32SurfaceCreateInfoKHR info{};
        info.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
        info.hinstance = window.instance != nullptr ? static_cast<HINSTANCE>(window.instance)
                                                    : GetModuleHandleW(nullptr);
        info.hwnd = static_cast<HWND>(window.window);
        return make<PFN_vkCreateWin32SurfaceKHR>(VK_KHR_WIN32_SURFACE_EXTENSION_NAME, "vkCreateWin32SurfaceKHR",
                                                 [&](auto create, VkSurfaceKHR* surface) {
                                                     return create(instance_, &info, nullptr, surface);
                                                 });
    }
#endif

#ifdef VK_USE_PLATFORM_METAL_EXT
    Made operator()(const core::window::Metal& window) const {
        VkMetalSurfaceCreateInfoEXT info{};
        info.sType = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT;
        info.pLayer = static_cast<const CAMetalLayer*>(window.layer);
        return make<PFN_vkCreateMetalSurfaceEXT>(VK_EXT_METAL_SURFACE_EXTENSION_NAME, "vkCreateMetalSurfaceEXT",
                                                 [&](auto create, VkSurfaceKHR* surface) {
                                                     return create(instance_, &info, nullptr, surface);
                                                 });
    }
#endif

    Made operator()(const core::window::Headless& /*window*/) const {
        VkHeadlessSurfaceCreateInfoEXT info{};
        info.sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT;
        return make<PFN_vkCreateHeadlessSurfaceEXT>(VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME,
                                                    "vkCreateHeadlessSurfaceEXT",
                                                    [&](auto create, VkSurfaceKHR* surface) {
                                                        return create(instance_, &info, nullptr, surface);
                                                    });
    }

    // A window system this build has no surface for.
    template <class Handles>
    Made operator()(const Handles& /*window*/) const {
        return core::failure(core::ErrorCode::RhiSurfaceFailed, "this build has no surface for the window's system");
    }

private:
    // The surface create makes once the instance enabled extension, or why not.
    // function is loaded for this instance, as the extension's functions are
    // not in its table.
    template <class Function, class Create>
    [[nodiscard]] Made make(std::string_view extension, const char* function, Create create) const {
        if (std::ranges::none_of(extensions_, [extension](const char* name) { return name == extension; })) {
            return core::failure(core::ErrorCode::RhiSurfaceFailed, std::format("{} is not available", extension));
        }
        const auto load = std::bit_cast<Function>(vkGetInstanceProcAddr(instance_, function));
        if (load == nullptr) {
            return core::failure(core::ErrorCode::RhiSurfaceFailed, std::format("{} is not available", function));
        }
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        if (const VkResult result = create(load, &surface); result != VK_SUCCESS) {
            return failure(core::ErrorCode::RhiSurfaceFailed, {.call = std::string{function}, .result = result});
        }
        return surface;
    }

    VkInstance instance_;
    std::span<const char* const> extensions_;
};

} // namespace

std::span<const char* const> surfaceExtensions() {
    return kSurfaceExtensions;
}

core::Result<VulkanSurface> createSurface(VkInstance instance, const InstanceTable& vk,
                                          std::span<const char* const> extensions,
                                          const core::window::NativeWindow& window) {
    return std::visit(Maker{instance, extensions}, window).transform([&](VkSurfaceKHR surface) {
        return VulkanSurface{instance, vk, surface, std::holds_alternative<core::window::Headless>(window)};
    });
}

VulkanSurface::VulkanSurface(VkInstance instance, const InstanceTable& vk, VkSurfaceKHR surface, bool headless)
    : instance_(instance), vk_(&vk), surface_(surface), headless_(headless) {}

VulkanSurface::VulkanSurface(VulkanSurface&& other) noexcept
    : instance_(other.instance_),
      vk_(other.vk_),
      surface_(std::exchange(other.surface_, VK_NULL_HANDLE)),
      headless_(other.headless_) {}

VulkanSurface::~VulkanSurface() {
    if (surface_ != VK_NULL_HANDLE) {
        vk_->vkDestroySurfaceKHR(instance_, surface_, nullptr);
    }
}

} // namespace bfr::rhi::vulkan
