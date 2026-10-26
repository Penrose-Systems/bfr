cmake_minimum_required(VERSION 3.28)
# A plain `cmake -B build` gets vcpkg's toolchain from VCPKG_ROOT, as the
# presets do. Only when BFR is the project being built; this runs before
# project(), so that shows as BFR's directory being the top of the build.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND NOT DEFINED CMAKE_TOOLCHAIN_FILE
   AND DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE FILEPATH "")
endif()
project(bfr LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# C only for checking that the C API header compiles as C.
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
# No C++ modules (code_style.md), so no dependency scan for them.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
# Every build writes compile_commands.json for clangd and clang-tidy.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compile through ccache when it is installed. A launcher the user sets for a
# language wins for that language.
option(BFR_USE_CCACHE "Compile through ccache when it is installed" ON)
if(BFR_USE_CCACHE)
    find_program(BFR_CCACHE ccache)
    foreach(language C CXX)
        if(BFR_CCACHE AND NOT CMAKE_${language}_COMPILER_LAUNCHER)
            set(CMAKE_${language}_COMPILER_LAUNCHER ${BFR_CCACHE})
        endif()
    endforeach()
endif()

option(BFR_LOGGING "Compile the BFR_LOG_* calls in" ON)

# Debug is the default build, and Debug runs under the address sanitizers.
# Only when BFR is the project being built: in a user's build, the build type
# and sanitizers are the user's, and sanitizers must cover the whole program.
get_property(multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(PROJECT_IS_TOP_LEVEL AND NOT multi_config AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()
if(PROJECT_IS_TOP_LEVEL AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(sanitize_default address)
else()
    set(sanitize_default none)
endif()

include(CMakeDependentOption)
cmake_dependent_option(BFR_BUILD_TESTS "Build the tests" ON "PROJECT_IS_TOP_LEVEL" OFF)
set(BFR_SANITIZE ${sanitize_default} CACHE STRING
    "address (AddressSanitizer, UndefinedBehaviorSanitizer, LeakSanitizer), thread (ThreadSanitizer) or none")
set_property(CACHE BFR_SANITIZE PROPERTY STRINGS address thread none)
cmake_dependent_option(BFR_CLANG_TIDY "Compile our sources, tests included, through clang-tidy" OFF "PROJECT_IS_TOP_LEVEL" OFF)

# ThreadSanitizer can't share a build with AddressSanitizer, so it is the other
# choice. The undefined group leaves out float-cast-overflow (GCC) and
# float-divide-by-zero (GCC and Clang). Any finding stops the run.
if(BFR_SANITIZE STREQUAL "address")
    set(sanitize_flags
        -fsanitize=address,undefined,float-cast-overflow,float-divide-by-zero
        -fno-sanitize-recover=all)
elseif(BFR_SANITIZE STREQUAL "thread")
    set(sanitize_flags -fsanitize=thread)
    # ThreadSanitizer's fixed memory layout fails under high address
    # randomisation, so tests run with it off.
    set(CMAKE_CROSSCOMPILING_EMULATOR setarch -R)
elseif(NOT BFR_SANITIZE STREQUAL "none")
    message(FATAL_ERROR "BFR_SANITIZE is '${BFR_SANITIZE}'; use address, thread or none")
endif()
if(sanitize_flags)
    list(APPEND sanitize_flags -fno-omit-frame-pointer)
    add_compile_options(${sanitize_flags})
    add_link_options(${sanitize_flags})
endif()

# Debug checks the standard library's preconditions, such as an index in range,
# before the access. It leaves the library's ABI unchanged.
add_compile_definitions($<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>)
# Debug fills uninitialised locals with a fixed pattern, so reading one goes
# wrong the same way on every run.
add_compile_options($<$<CONFIG:Debug>:-ftrivial-auto-var-init=pattern>)

include(cmake/bfr_build.cmake)

find_package(spdlog CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(SDL3 CONFIG REQUIRED)
find_package(volk CONFIG REQUIRED)
find_package(VulkanUtilityLibraries CONFIG REQUIRED)

# One library. Its public API is include/bfr/public; the rest of include/bfr is
# its private headers.
add_library(bfr STATIC
    src/config.cpp
    src/error.cpp
    src/log.cpp
    src/rhi_config.cpp
    src/sdl_display.cpp
    src/sdl_input.cpp
    src/vulkan/queue_families.cpp
    src/vulkan/swapchain_settings.cpp
    src/vulkan/vulkan_device.cpp
    src/vulkan/vulkan_loader.cpp
    src/vulkan/vulkan_rhi.cpp
    src/vulkan/vulkan_surface.cpp
    src/vulkan/vulkan_swapchain.cpp)
bfr_apply_compiler_flags(bfr)
target_include_directories(bfr
    PUBLIC include/bfr/public
    PRIVATE include/bfr include/bfr/vulkan)
target_link_libraries(bfr
    PUBLIC spdlog::spdlog
    PRIVATE nlohmann_json::nlohmann_json SDL3::SDL3 volk::volk_headers Vulkan::UtilityHeaders ${CMAKE_DL_LIBS})
target_compile_definitions(bfr PUBLIC BFR_LOGGING_ENABLED=$<BOOL:${BFR_LOGGING}>)
bfr_add_header_check(bfr ${CMAKE_CURRENT_SOURCE_DIR}/include/bfr/public)
bfr_add_c_header_check(bfr bfr.h)

# The window systems the Vulkan backend makes surfaces on. Only volk and the
# surface code see them: the Xlib and Windows headers they pull in define
# macros that would leak into the rest. volk keeps its tables the same layout
# without them.
if(WIN32)
    set(bfr_vulkan_platforms VK_USE_PLATFORM_WIN32_KHR WIN32_LEAN_AND_MEAN NOMINMAX)
elseif(APPLE)
    set(bfr_vulkan_platforms VK_USE_PLATFORM_METAL_EXT)
else()
    set(bfr_vulkan_platforms VK_USE_PLATFORM_XLIB_KHR VK_USE_PLATFORM_WAYLAND_KHR)
endif()
set_source_files_properties(src/vulkan/vulkan_surface.cpp
    PROPERTIES COMPILE_DEFINITIONS "${bfr_vulkan_platforms}")
# Third-party code: without our warnings or clang-tidy.
add_library(bfr_volk OBJECT src/vulkan/volk.cpp)
target_link_libraries(bfr_volk PRIVATE volk::volk_headers)
target_compile_definitions(bfr_volk PRIVATE ${bfr_vulkan_platforms})
target_link_libraries(bfr PRIVATE bfr_volk)

if(BFR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
