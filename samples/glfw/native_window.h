// The values BFR needs from a GLFW window. Kept apart from main.cpp because
// GLFW's native header pulls in X11's, whose macros (Success, None) clash with
// names in bfr.hpp.
#pragma once

struct GLFWwindow;

struct NativeWindow {
    void* window = nullptr;
    void* display = nullptr;
    bool wayland = false;
};

// Which window system GLFW is running on decides which handles it gives.
NativeWindow nativeWindow(GLFWwindow* window);
