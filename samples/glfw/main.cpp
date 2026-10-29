// Opens a GLFW window and makes a BFR swapchain for it, resizing the swapchain
// as the window resizes. Close the window to exit.
#include "native_window.h"
#include "window_sample.hpp"

#include <GLFW/glfw3.h>

#include <cstdint>
#include <iostream>

namespace {

constexpr int kWidth = 1280;
constexpr int kHeight = 720;
constexpr double kWaitSeconds = 0.1;

} // namespace

int main() {
    if (glfwInit() == GLFW_FALSE) {
        std::cerr << "can't start GLFW\n";
        return 1;
    }
    // BFR makes the surface, so GLFW is to make none.
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(kWidth, kHeight, "bfr_glfw", nullptr, nullptr);
    if (window == nullptr) {
        std::cerr << "can't open a window\n";
        glfwTerminate();
        return 1;
    }

    int result = 0;
    {
        const NativeWindow native = nativeWindow(window);
        result = sample::show(
            {.window = native.window,
             .display = native.display,
             .system = native.wayland ? bfr::WindowSystem::Wayland : bfr::WindowSystem::Default},
            [window] {
                glfwWaitEventsTimeout(kWaitSeconds);
                return glfwWindowShouldClose(window) == GLFW_FALSE;
            },
            [window] {
                int width = 0;
                int height = 0;
                glfwGetFramebufferSize(window, &width, &height);
                return sample::Size{.width = static_cast<std::uint32_t>(width),
                                    .height = static_cast<std::uint32_t>(height)};
            });
    }
    // The swapchain and device are gone by here, so the window can go.
    glfwDestroyWindow(window);
    glfwTerminate();
    return result;
}
