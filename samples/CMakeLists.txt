# Programs that show how to use BFR, through the C++ wrapper and the shared
# library, as an application would.
add_executable(bfr_caps caps/main.cpp)
bfr_apply_compiler_flags(bfr_caps)
target_link_libraries(bfr_caps PRIVATE bfr_cpp bfr_shared)

# With the tests, each sample runs as one, every option on, under the tests'
# sanitizer options and Vulkan environment (tests/library/test_main.cpp).
if(BFR_BUILD_TESTS)
    target_link_libraries(bfr_caps PRIVATE bfr_sanitizer_options)
    set(environment VK_LOADER_DISABLE_DYNAMIC_LIBRARY_UNLOADING=1)
    if(BFR_SANITIZE STREQUAL "thread")
        list(APPEND environment VK_DRIVER_FILES=${BFR_LAVAPIPE_ICD})
    endif()
    add_test(NAME samples.caps COMMAND bfr_caps --formats --window --all)
    set_tests_properties(samples.caps PROPERTIES LABELS gpu ENVIRONMENT "${environment}")
endif()
