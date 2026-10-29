# Programs that show how to use BFR, through the C++ wrapper and the shared
# library, as an application would.
add_executable(bfr_caps caps/main.cpp)
bfr_apply_compiler_flags(bfr_caps)
target_link_libraries(bfr_caps PRIVATE bfr_cpp bfr_shared)

# Programs that open a window with a windowing library and give BFR its handles.
# The shared code is in common/window_sample.hpp; each sample's own code is the
# window and the handles it hands over.
find_package(SDL3 CONFIG REQUIRED)
add_executable(bfr_sdl sdl/main.cpp)
bfr_apply_compiler_flags(bfr_sdl)
target_include_directories(bfr_sdl PRIVATE common)
target_link_libraries(bfr_sdl PRIVATE bfr_cpp bfr_shared SDL3::SDL3)

# GLFW has no way to get a CAMetalLayer without Objective-C.
if(NOT APPLE)
    find_package(glfw3 CONFIG REQUIRED)
    add_executable(bfr_glfw glfw/main.cpp glfw/native_window.cpp)
    bfr_apply_compiler_flags(bfr_glfw)
    target_include_directories(bfr_glfw PRIVATE common)
    target_link_libraries(bfr_glfw PRIVATE bfr_cpp bfr_shared glfw)
endif()
