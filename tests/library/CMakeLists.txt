# One Catch2 executable for every test of the library; each case carries its
# area tag, such as [core] or [rhi], and a kind tag, [unit] or [gpu]
# (code_style.md).
find_package(Catch2 3 CONFIG REQUIRED)
include(Catch)

# The main every Catch2 test program runs, which sets the environment the GPU
# tests need; the C++ wrapper's tests use it too.
add_library(bfr_test_main OBJECT test_main.cpp)
bfr_apply_compiler_flags(bfr_test_main)
target_link_libraries(bfr_test_main PUBLIC Catch2::Catch2)

add_executable(bfr_tests
    c_api/test_c_api.cpp
    config/test_config.cpp
    core/test_error.cpp
    core/test_log.cpp
    core/test_pool.cpp
    core/test_text.cpp
    platform/test_display.cpp
    platform/test_display_input.cpp
    rhi/test_queue_families.cpp
    rhi/test_rhi.cpp
    rhi/test_rhi_config.cpp
    rhi/test_swapchain.cpp
    rhi/test_swapchain_settings.cpp
    rhi/test_vulkan_surface.cpp)
bfr_apply_compiler_flags(bfr_tests)
# Test helpers shared across the tests, such as scoped_env.h.
target_include_directories(bfr_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# The Vulkan backend's queue family choice is tested as data, from its private
# header.
target_include_directories(bfr_tests PRIVATE ${PROJECT_SOURCE_DIR}/include/bfr/vulkan)
# The platform tests drive the display through SDL's event queue.
target_link_libraries(bfr_tests PRIVATE bfr bfr_test_main bfr_sanitizer_options Catch2::Catch2 SDL3::SDL3
    volk::volk_headers)
# The fullscreen tests give a Wayland window a blank buffer, which needs
# wayland-client; the Wayland headers are installed for SDL3 already.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(WAYLAND_CLIENT REQUIRED IMPORTED_TARGET wayland-client)
    target_link_libraries(bfr_tests PRIVATE PkgConfig::WAYLAND_CLIENT)
endif()
# vcpkg's validation layer manifest names its library without a path, so the
# loader finds a system copy of the layer, or none. The tests use a copy of the
# manifest that names vcpkg's library, its release build, by its full path.
set(vcpkg_prefix ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET})
set(validation_layer ${CMAKE_CURRENT_BINARY_DIR}/validation_layer)
file(READ ${vcpkg_prefix}/share/vulkan/explicit_layer.d/VkLayer_khronos_validation.json manifest)
string(JSON library GET "${manifest}" layer library_path)
cmake_path(GET library FILENAME library)
string(JSON manifest SET "${manifest}" layer library_path "\"${vcpkg_prefix}/lib/${library}\"")
file(WRITE ${validation_layer}/VkLayer_khronos_validation.json "${manifest}")
target_compile_definitions(bfr_test_main PRIVATE BFR_TEST_LAYER_PATH="${validation_layer}")
if(BFR_SANITIZE STREQUAL "thread")
    find_file(BFR_LAVAPIPE_ICD NAMES lvp_icd.json lvp_icd.x86_64.json
        PATHS /usr/share/vulkan/icd.d NO_DEFAULT_PATH)
    if(NOT BFR_LAVAPIPE_ICD)
        message(FATAL_ERROR "ThreadSanitizer builds run the GPU tests on lavapipe; install mesa-vulkan-drivers")
    endif()
    target_compile_definitions(bfr_test_main PRIVATE BFR_TEST_DRIVER_FILES="${BFR_LAVAPIPE_ICD}")
endif()

# A Vulkan layer the RHI tests load, written to directory with its manifest.
# Built without the sanitizers, as real layers are. GCC links the sanitizer
# runtime into a sanitized library, which moves the leak check after the GPU
# driver's destructors, and the check then reports memory the driver owned.
function(bfr_add_test_layer target source directory)
    add_library(${target} MODULE ${source})
    bfr_apply_compiler_flags(${target})
    target_link_libraries(${target} PRIVATE Vulkan::Headers)
    set_target_properties(${target} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${directory})
    if(sanitize_flags)
        foreach(property COMPILE_OPTIONS LINK_OPTIONS)
            get_target_property(options ${target} ${property})
            list(REMOVE_ITEM options ${sanitize_flags})
            set_target_properties(${target} PROPERTIES ${property} "${options}")
        endforeach()
    endif()
    add_dependencies(bfr_tests ${target})
endfunction()

# A layer that changes what every adapter reports, and fails calls. The RHI
# tests add its directory to the loader's layer path and enable it by name.
set(adapter_override_layer ${CMAKE_CURRENT_BINARY_DIR}/adapter_override_layer)
bfr_add_test_layer(bfr_test_adapter_override_layer rhi/fixtures/adapter_override_layer.cpp
    ${adapter_override_layer})
file(GENERATE OUTPUT ${adapter_override_layer}/adapter_override_layer.json CONTENT [=[{
    "file_format_version": "1.2.0",
    "layer": {
        "name": "VK_LAYER_BFR_adapter_override",
        "type": "INSTANCE",
        "library_path": "$<TARGET_FILE:bfr_test_adapter_override_layer>",
        "api_version": "1.4.0",
        "implementation_version": "1",
        "description": "Changes what every adapter reports"
    }
}
]=])
target_compile_definitions(bfr_tests PRIVATE BFR_TEST_ADAPTER_LAYER_PATH="${adapter_override_layer}")

# An implicit layer that reports any loader version. The loader finds implicit
# layers in $XDG_DATA_HOME/vulkan/implicit_layer.d, which the tests that need
# it point here.
set(loader_version_data ${CMAKE_CURRENT_BINARY_DIR}/loader_version_data)
set(loader_version_layer ${loader_version_data}/vulkan/implicit_layer.d)
bfr_add_test_layer(bfr_test_loader_version_layer rhi/fixtures/loader_version_layer.cpp ${loader_version_layer})
file(GENERATE OUTPUT ${loader_version_layer}/loader_version_layer.json CONTENT [=[{
    "file_format_version": "1.2.0",
    "layer": {
        "name": "VK_LAYER_BFR_loader_version",
        "type": "GLOBAL",
        "library_path": "$<TARGET_FILE:bfr_test_loader_version_layer>",
        "api_version": "1.4.0",
        "implementation_version": "1",
        "description": "Reports the loader as the version BFR_LOADER_VERSION gives",
        "pre_instance_functions": {"vkEnumerateInstanceVersion": "bfrEnumerateInstanceVersion"},
        "disable_environment": {"BFR_LOADER_VERSION_LAYER_OFF": "1"}
    }
}
]=])
target_compile_definitions(bfr_tests PRIVATE BFR_TEST_LOADER_VERSION_DATA="${loader_version_data}")

# Everything not tagged [gpu] is a unit test, so a case missing its kind tag
# still runs. Hidden cases ([.]) run only through their own entries below.
catch_discover_tests(bfr_tests TEST_SPEC "~[gpu]")
catch_discover_tests(bfr_tests TEST_SPEC "[gpu]~[.]" PROPERTIES LABELS gpu)

# Each loader case runs alone with LD_LIBRARY_PATH at a broken loader, which
# glibc reads only when the process starts.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(broken_loader ${CMAKE_CURRENT_BINARY_DIR}/broken_loader)
    file(WRITE ${broken_loader}/libvulkan.so.1 "")
    file(WRITE ${broken_loader}/libvulkan.so "")
    add_test(NAME rhi.loader_broken COMMAND bfr_tests "Rhi reports why each loader library failed")
    set_tests_properties(rhi.loader_broken PROPERTIES ENVIRONMENT LD_LIBRARY_PATH=${broken_loader})

    set(entryless_loader ${CMAKE_CURRENT_BINARY_DIR}/entryless_loader)
    add_library(bfr_test_entryless_loader SHARED rhi/fixtures/entryless_loader.cpp)
    bfr_apply_compiler_flags(bfr_test_entryless_loader)
    # libvulkan.so.1, with libvulkan.so linking to it.
    set_target_properties(bfr_test_entryless_loader PROPERTIES
        OUTPUT_NAME vulkan SOVERSION 1 LIBRARY_OUTPUT_DIRECTORY ${entryless_loader})
    add_test(NAME rhi.loader_entryless
        COMMAND bfr_tests "Rhi reports a loader library without vkGetInstanceProcAddr")
    set_tests_properties(rhi.loader_entryless PROPERTIES ENVIRONMENT LD_LIBRARY_PATH=${entryless_loader})

    set(fake_loader ${CMAKE_CURRENT_BINARY_DIR}/fake_loader)
    add_library(bfr_test_fake_loader SHARED rhi/fixtures/fake_loader.cpp)
    bfr_apply_compiler_flags(bfr_test_fake_loader)
    target_link_libraries(bfr_test_fake_loader PRIVATE Vulkan::Headers)
    set_target_properties(bfr_test_fake_loader PROPERTIES
        OUTPUT_NAME vulkan SOVERSION 1 LIBRARY_OUTPUT_DIRECTORY ${fake_loader})
    foreach(listing layer extension)
        add_test(NAME rhi.fake_loader_${listing}s
            COMMAND bfr_tests "Rhi with validation reports a failed ${listing} listing")
        set_tests_properties(rhi.fake_loader_${listing}s PROPERTIES
            ENVIRONMENT "LD_LIBRARY_PATH=${fake_loader};BFR_FAKE_LOADER_MODE=${listing}s")
    endforeach()
    add_test(NAME rhi.fake_loader_grow COMMAND bfr_tests "Rhi with validation lists again when a listing grows")
    set_tests_properties(rhi.fake_loader_grow PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=${fake_loader};BFR_FAKE_LOADER_MODE=grow")
endif()

# lavapipe marked as a portability driver, as MoltenVK is: the loader lists it
# only to an instance that enumerates portability drivers.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(portability_driver ${CMAKE_CURRENT_BINARY_DIR}/portability_driver.json)
    file(WRITE ${portability_driver} [=[{
    "file_format_version": "1.0.1",
    "ICD": {
        "library_path": "libvulkan_lvp.so",
        "api_version": "1.4.0",
        "is_portability_driver": true
    }
}
]=])
    add_test(NAME rhi.portability_driver COMMAND bfr_tests "Rhi lists an adapter from a portability driver")
    set_tests_properties(rhi.portability_driver PROPERTIES
        ENVIRONMENT VK_DRIVER_FILES=${portability_driver} LABELS gpu)
endif()
