# Every test program links this, so its sanitizer options apply however it runs.
add_library(bfr_sanitizer_options OBJECT sanitizer_options.cpp)
bfr_apply_compiler_flags(bfr_sanitizer_options)

add_subdirectory(cmake)
add_subdirectory(library)

if(NOT BFR_SANITIZE STREQUAL "none")
    add_subdirectory(sanitizers)
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_subdirectory(assertions)
endif()
