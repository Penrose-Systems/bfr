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
# Each backend is a build option, on wherever the platform has it, so a build
# holds every backend it can and a program picks one at run time. Rhi::create
# dispatches on those turned on and refuses the rest with RhiBackendUnavailable.
# Turning one on where the platform lacks it is an error. WIN32 and APPLE are
# empty elsewhere.
option(BFR_BACKEND_VULKAN "Build the Vulkan backend" ON)
option(BFR_BACKEND_D3D12 "Build the D3D12 backend; needs the Windows SDK" ${WIN32})
option(BFR_BACKEND_METAL "Build the Metal backend; needs metal-cpp" ${APPLE})

# Debug is the default build. Only when BFR is the project being built: in a
# user's build, the build type is the user's.
get_property(multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(PROJECT_IS_TOP_LEVEL AND NOT multi_config AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

include(CMakeDependentOption)
cmake_dependent_option(BFR_BUILD_TESTS "Build the tests" ON "PROJECT_IS_TOP_LEVEL" OFF)
cmake_dependent_option(BFR_BUILD_SAMPLES "Build the samples" ON "PROJECT_IS_TOP_LEVEL" OFF)
set(BFR_SANITIZE none CACHE STRING
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

# BFR's code, compiled once for both libraries below. include/bfr is its
# public API: bfr.h, the C API, and the C++ layers behind it, bfr/core and
# bfr/rhi. Each backend is one directory under src/backends, private, and
# only src/rhi/rhi.cpp names them. The null backend is the tests' own, built
# into them alone.
add_library(bfr_objects OBJECT
    src/c_api/c_api.cpp
    src/core/error.cpp
    src/rhi/rhi.cpp)
bfr_apply_compiler_flags(bfr_objects)
# Private headers are named from src: "vulkan/vulkan_rhi.h".
target_include_directories(bfr_objects
    PUBLIC include
    PRIVATE src src/backends)
target_link_libraries(bfr_objects PRIVATE ${CMAKE_DL_LIBS})
target_compile_definitions(bfr_objects
    PUBLIC BFR_LOGGING_ENABLED=$<BOOL:${BFR_LOGGING}>
    PRIVATE BFR_BUILDING BFR_HAS_VULKAN=$<BOOL:${BFR_BACKEND_VULKAN}> BFR_HAS_D3D12=$<BOOL:${BFR_BACKEND_D3D12}>
    BFR_HAS_METAL=$<BOOL:${BFR_BACKEND_METAL}>)
# Position independent for the shared library, and hidden, so that it exports
# only the BFR_API entry points.
set_target_properties(bfr_objects PROPERTIES
    POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)

# Third-party code each backend brings, built without our warnings or
# clang-tidy, and linked into both libraries beside bfr_objects.
set(bfr_backend_objects "")

if(BFR_BACKEND_VULKAN)
    find_package(volk CONFIG REQUIRED)
    find_package(VulkanUtilityLibraries CONFIG REQUIRED)
    target_sources(bfr_objects PRIVATE
        src/backends/vulkan/instance_table.cpp
        src/backends/vulkan/queue_families.cpp
        src/backends/vulkan/swapchain_settings.cpp
        src/backends/vulkan/vulkan_caps.cpp
        src/backends/vulkan/vulkan_device.cpp
        src/backends/vulkan/vulkan_loader.cpp
        src/backends/vulkan/vulkan_rhi.cpp
        src/backends/vulkan/vulkan_surface.cpp
        src/backends/vulkan/vulkan_swapchain.cpp)
    target_link_libraries(bfr_objects PRIVATE volk::volk_headers Vulkan::UtilityHeaders)
    # The window systems the Vulkan backend makes surfaces on. Only volk and
    # the surface code see them: the Xlib and Windows headers they pull in
    # define macros that would leak into the rest. volk keeps its tables the
    # same layout without them.
    if(WIN32)
        set(bfr_vulkan_platforms VK_USE_PLATFORM_WIN32_KHR WIN32_LEAN_AND_MEAN NOMINMAX)
    elseif(APPLE)
        set(bfr_vulkan_platforms VK_USE_PLATFORM_METAL_EXT)
    else()
        set(bfr_vulkan_platforms VK_USE_PLATFORM_XLIB_KHR VK_USE_PLATFORM_WAYLAND_KHR)
    endif()
    set_source_files_properties(src/backends/vulkan/vulkan_surface.cpp
        PROPERTIES COMPILE_DEFINITIONS "${bfr_vulkan_platforms}")
    add_library(bfr_volk OBJECT src/backends/vulkan/volk.cpp)
    target_link_libraries(bfr_volk PRIVATE volk::volk_headers)
    target_compile_definitions(bfr_volk PRIVATE ${bfr_vulkan_platforms})
    set_target_properties(bfr_volk PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden)
    list(APPEND bfr_backend_objects bfr_volk)
endif()

if(BFR_BACKEND_D3D12)
    if(NOT WIN32)
        message(FATAL_ERROR "The D3D12 backend builds only on Windows")
    endif()
    target_sources(bfr_objects PRIVATE
        src/backends/d3d12/d3d12_caps.cpp
        src/backends/d3d12/d3d12_device.cpp
        src/backends/d3d12/d3d12_formats.cpp
        src/backends/d3d12/d3d12_rhi.cpp
        src/backends/d3d12/d3d12_swapchain.cpp)
    # The Windows headers define macros, min and max among them, that would
    # leak into every file that includes a backend's header.
    target_compile_definitions(bfr_objects PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_link_libraries(bfr_objects PRIVATE d3d12 dxgi)
endif()

if(BFR_BACKEND_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "The Metal backend builds only on Apple platforms")
    endif()
    # metal-cpp, Apple's header-only C++ wrapper of Metal: the directory that
    # holds Metal/Metal.hpp. Found through CMAKE_PREFIX_PATH or
    # BFR_METAL_CPP_INCLUDE_DIR.
    find_path(BFR_METAL_CPP_INCLUDE_DIR Metal/Metal.hpp REQUIRED)
    target_sources(bfr_objects PRIVATE
        src/backends/metal/metal_caps.cpp
        src/backends/metal/metal_device.cpp
        src/backends/metal/metal_formats.cpp
        src/backends/metal/metal_rhi.cpp
        src/backends/metal/metal_swapchain.cpp)
    target_include_directories(bfr_objects SYSTEM PRIVATE ${BFR_METAL_CPP_INCLUDE_DIR})
    target_link_libraries(bfr_objects PRIVATE "-framework Foundation" "-framework Metal" "-framework QuartzCore"
        "-framework CoreGraphics")
    # metal-cpp wants exactly one source to define its implementation.
    add_library(bfr_metal_cpp OBJECT src/backends/metal/metal_cpp.cpp)
    target_include_directories(bfr_metal_cpp SYSTEM PRIVATE ${BFR_METAL_CPP_INCLUDE_DIR})
    set_target_properties(bfr_metal_cpp PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden)
    list(APPEND bfr_backend_objects bfr_metal_cpp)
endif()

# The static library: the C API, and the C++ classes behind it that the tests
# use.
add_library(bfr STATIC)
target_link_libraries(bfr PUBLIC bfr_objects PRIVATE ${bfr_backend_objects})
target_compile_definitions(bfr INTERFACE BFR_STATIC)
set_target_properties(bfr PROPERTIES OUTPUT_NAME bfr_static)
bfr_add_header_check(bfr ${CMAKE_CURRENT_SOURCE_DIR}/include)
bfr_add_c_header_check(bfr bfr/bfr.h)

# The shared library the language bindings load. It exports only bfr_*.
# Hidden visibility is not enough on its own: libstdc++ gives its templates
# default visibility, so the linker is told the list too.
add_library(bfr_shared SHARED)
target_link_libraries(bfr_shared PRIVATE bfr_objects ${bfr_backend_objects})
# Its users see bfr.h and nothing of the C++ behind it.
target_include_directories(bfr_shared INTERFACE include)
set_target_properties(bfr_shared PROPERTIES OUTPUT_NAME bfr)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(exports ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bfr_exports.map)
    target_link_options(bfr_shared PRIVATE -Wl,--version-script=${exports} -Wl,--no-undefined)
    set_property(TARGET bfr_shared APPEND PROPERTY LINK_DEPENDS ${exports})
elseif(APPLE)
    target_link_options(bfr_shared PRIVATE "LINKER:-exported_symbol,_bfr_*")
endif()

if(BFR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

add_subdirectory(bindings/cpp)

if(BFR_BUILD_SAMPLES)
    add_subdirectory(samples)
endif()
