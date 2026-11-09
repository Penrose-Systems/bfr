# One Catch2 executable for every test of the library; each case carries its
# area tag, such as [core] or [rhi], and a kind tag: [unit], [agnostic] for a
# case on the core interface that runs on each backend, or [gpu] for a case
# that needs a real backend (code_style.md).
find_package(Catch2 3 CONFIG REQUIRED)
include(Catch)

# The main every Catch2 test program runs; the C++ wrapper's tests use it too.
# A backend that needs its environment set up links its test_environment.cpp
# in here, below.
add_library(bfr_test_main OBJECT test_main.cpp)
bfr_apply_compiler_flags(bfr_test_main)
# test_backends.h, which the agnostic cases' backend generators ask.
target_include_directories(bfr_test_main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(bfr_test_main PUBLIC Catch2::Catch2)

# Mesa's lavapipe: Vulkan on the CPU, so the agnostic cases run on a machine
# with no GPU. Without it they run only under the gpu label.
if(BFR_BACKEND_VULKAN)
    find_file(BFR_LAVAPIPE_ICD lvp_icd.json PATHS /usr/share/vulkan/icd.d /usr/local/share/vulkan/icd.d
        /etc/vulkan/icd.d NO_DEFAULT_PATH)
    if(NOT BFR_LAVAPIPE_ICD)
        message(STATUS "lavapipe not found: the agnostic cases run only under the gpu label")
    endif()
endif()

# Registers target's cases with ctest. The unit cases run everywhere, and so
# do the agnostic cases, on lavapipe when it is there. Under the gpu label,
# the [gpu] cases run, and the agnostic cases run again on each GPU backend
# the build has. Vulkan's agnostic entries run again at Vulkan 1.2, BFR's
# minimum, so the extension forms of what 1.3 made core run too. Hidden cases
# ([.]) run only through their own entries.
function(bfr_discover_tests target)
    catch_discover_tests(${target} TEST_SPEC "~[gpu]~[agnostic]")
    # Catch names each entry's script by its spec and arguments alone, so this
    # spec is spelled apart from the gpu label's vulkan entry; no agnostic
    # case is [gpu], so it selects the same cases.
    if(BFR_LAVAPIPE_ICD)
        catch_discover_tests(${target} TEST_SPEC "[agnostic]~[gpu]" TEST_SUFFIX " (lavapipe)"
            EXTRA_ARGS --backend vulkan PROPERTIES ENVIRONMENT "VK_DRIVER_FILES=${BFR_LAVAPIPE_ICD}")
        catch_discover_tests(${target} TEST_SPEC "[agnostic]~[gpu]" TEST_SUFFIX " (lavapipe, Vulkan 1.2)"
            EXTRA_ARGS --backend vulkan --api-version 1.2
            PROPERTIES ENVIRONMENT "VK_DRIVER_FILES=${BFR_LAVAPIPE_ICD}")
    endif()
    catch_discover_tests(${target} TEST_SPEC "[gpu]~[.]" PROPERTIES LABELS gpu)
    if(BFR_BACKEND_VULKAN)
        catch_discover_tests(${target} TEST_SPEC "[agnostic]" TEST_SUFFIX " (vulkan)"
            EXTRA_ARGS --backend vulkan PROPERTIES LABELS gpu)
        catch_discover_tests(${target} TEST_SPEC "[agnostic]" TEST_SUFFIX " (vulkan 1.2)"
            EXTRA_ARGS --backend vulkan --api-version 1.2 PROPERTIES LABELS gpu)
    endif()
    if(BFR_BACKEND_D3D12)
        catch_discover_tests(${target} TEST_SPEC "[agnostic]" TEST_SUFFIX " (d3d12)"
            EXTRA_ARGS --backend d3d12 PROPERTIES LABELS gpu)
    endif()
    if(BFR_BACKEND_METAL)
        catch_discover_tests(${target} TEST_SPEC "[agnostic]" TEST_SUFFIX " (metal)"
            EXTRA_ARGS --backend metal PROPERTIES LABELS gpu)
    endif()
endfunction()

# The tests under rhi speak only to the interface and run on every backend
# the build has; a backend's own tests sit under backends/<name>.
add_executable(bfr_tests
    allocation/allocation_counter.cpp
    c_api/test_c_api.cpp
    core/test_error.cpp
    core/test_log.cpp
    core/test_pool.cpp
    core/test_slot_chains.cpp
    core/test_submissions.cpp
    core/test_text.cpp
    rhi/test_buffer.cpp
    rhi/test_caps.cpp
    rhi/test_frame_allocations.cpp
    rhi/test_instance.cpp
    rhi/test_pipeline.cpp
    rhi/test_draw.cpp
    rhi/test_present.cpp
    rhi/test_sampler.cpp
    rhi/test_sparse.cpp
    rhi/test_timestamp.cpp
    rhi/test_swapchain.cpp
    rhi/test_texture.cpp)
bfr_apply_compiler_flags(bfr_tests)
# Test helpers shared across the tests, such as scoped_env.h, and the
# library's private headers under src.
target_include_directories(bfr_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/backends)
target_link_libraries(bfr_tests PRIVATE bfr bfr_test_main bfr_sanitizer_options Catch2::Catch2)
bfr_discover_tests(bfr_tests)

if(NOT BFR_BACKEND_VULKAN)
    return()
endif()

# The environment the Vulkan tests run in, set before main in every test
# program that links bfr_test_main.
add_library(bfr_test_vulkan_environment OBJECT backends/vulkan/test_environment.cpp)
bfr_apply_compiler_flags(bfr_test_vulkan_environment)
# Its object goes into every program that links bfr_test_main: an object
# library linked through another is not.
target_sources(bfr_test_main INTERFACE $<TARGET_OBJECTS:bfr_test_vulkan_environment>)

# The Vulkan backend's tests, which reach its private headers.
target_sources(bfr_tests PRIVATE
    backends/vulkan/test_queue_families.cpp
    backends/vulkan/test_swapchain_settings.cpp
    backends/vulkan/test_vulkan_copies.cpp
    backends/vulkan/test_vulkan_caps.cpp
    backends/vulkan/test_vulkan_instance.cpp
    backends/vulkan/test_vulkan_surface.cpp
    backends/vulkan/test_vulkan_swapchain.cpp)
target_link_libraries(bfr_tests PRIVATE volk::volk_headers)
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
target_compile_definitions(bfr_test_vulkan_environment PRIVATE BFR_TEST_LAYER_PATH="${validation_layer}")
if(BFR_SANITIZE STREQUAL "thread")
    find_file(BFR_LAVAPIPE_ICD NAMES lvp_icd.json lvp_icd.x86_64.json
        PATHS /usr/share/vulkan/icd.d NO_DEFAULT_PATH)
    if(NOT BFR_LAVAPIPE_ICD)
        message(FATAL_ERROR "ThreadSanitizer builds run the GPU tests on lavapipe; install mesa-vulkan-drivers")
    endif()
    target_compile_definitions(bfr_test_vulkan_environment PRIVATE BFR_TEST_DRIVER_FILES="${BFR_LAVAPIPE_ICD}")
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
bfr_add_test_layer(bfr_test_adapter_override_layer backends/vulkan/fixtures/adapter_override_layer.cpp
    ${adapter_override_layer})
# It answers VK_KHR_portability_subset's features, which are only in vulkan_beta.h.
target_compile_definitions(bfr_test_adapter_override_layer PRIVATE VK_ENABLE_BETA_EXTENSIONS)
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
bfr_add_test_layer(bfr_test_loader_version_layer backends/vulkan/fixtures/loader_version_layer.cpp ${loader_version_layer})
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

# Each loader case runs alone with LD_LIBRARY_PATH at a broken loader, which
# glibc reads only when the process starts.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(broken_loader ${CMAKE_CURRENT_BINARY_DIR}/broken_loader)
    file(WRITE ${broken_loader}/libvulkan.so.1 "")
    file(WRITE ${broken_loader}/libvulkan.so "")
    add_test(NAME vulkan.loader_broken COMMAND bfr_tests "Vulkan reports why each loader library failed")
    set_tests_properties(vulkan.loader_broken PROPERTIES ENVIRONMENT LD_LIBRARY_PATH=${broken_loader})

    set(entryless_loader ${CMAKE_CURRENT_BINARY_DIR}/entryless_loader)
    add_library(bfr_test_entryless_loader SHARED backends/vulkan/fixtures/entryless_loader.cpp)
    bfr_apply_compiler_flags(bfr_test_entryless_loader)
    # libvulkan.so.1, with libvulkan.so linking to it.
    set_target_properties(bfr_test_entryless_loader PROPERTIES
        OUTPUT_NAME vulkan SOVERSION 1 LIBRARY_OUTPUT_DIRECTORY ${entryless_loader})
    add_test(NAME vulkan.loader_entryless
        COMMAND bfr_tests "Vulkan reports a loader library without vkGetInstanceProcAddr")
    set_tests_properties(vulkan.loader_entryless PROPERTIES ENVIRONMENT LD_LIBRARY_PATH=${entryless_loader})

    set(fake_loader ${CMAKE_CURRENT_BINARY_DIR}/fake_loader)
    add_library(bfr_test_fake_loader SHARED backends/vulkan/fixtures/fake_loader.cpp)
    bfr_apply_compiler_flags(bfr_test_fake_loader)
    target_link_libraries(bfr_test_fake_loader PRIVATE Vulkan::Headers)
    set_target_properties(bfr_test_fake_loader PROPERTIES
        OUTPUT_NAME vulkan SOVERSION 1 LIBRARY_OUTPUT_DIRECTORY ${fake_loader})
    add_test(NAME vulkan.fake_loader_extensions COMMAND bfr_tests "Vulkan reports a failed extension listing")
    set_tests_properties(vulkan.fake_loader_extensions PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=${fake_loader};BFR_FAKE_LOADER_MODE=extensions")
    add_test(NAME vulkan.fake_loader_grow COMMAND bfr_tests "Vulkan lists again when a listing grows")
    set_tests_properties(vulkan.fake_loader_grow PROPERTIES
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
    add_test(NAME vulkan.portability_driver COMMAND bfr_tests "Vulkan lists an adapter from a portability driver")
    set_tests_properties(vulkan.portability_driver PROPERTIES
        ENVIRONMENT VK_DRIVER_FILES=${portability_driver} LABELS gpu)
endif()
