#include "native_window.h"

#include <cstdint>

#if defined(_WIN32)
    #define GLFW_EXPOSE_NATIVE_WIN32
#elif defined(__linux__)
    #define GLFW_EXPOSE_NATIVE_X11
    #define GLFW_EXPOSE_NATIVE_WAYLAND
#endif
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

NativeWindow nativeWindow(GLFWwindow* window) {
#if defined(_WIN32)
    return {.window = glfwGetWin32Window(window)};
#elif defined(__linux__)
    if (glfwGetPlatform() == GLFW_PLATFORM_WAYLAND) {
        return {.window = glfwGetWaylandWindow(window), .display = glfwGetWaylandDisplay(), .wayland = true};
    }
    // An X11 window is a number, which BFR takes as a pointer.
    const auto xid = static_cast<std::uintptr_t>(glfwGetX11Window(window));
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)
    return {.window = reinterpret_cast<void*>(xid), .display = glfwGetX11Display()};
#else
    return {};
#endif
}
