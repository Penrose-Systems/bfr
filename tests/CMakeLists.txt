# Every test program links this, so its sanitizer options apply however it runs.
add_library(bfr_sanitizer_options OBJECT sanitizer_options.cpp)
bfr_apply_compiler_flags(bfr_sanitizer_options)

add_subdirectory(cmake)

# The committed C API header matches api/bfr.yaml.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME c_api.header_up_to_date COMMAND Python3::Interpreter ${PROJECT_SOURCE_DIR}/tools/gen_c_api.py --check)
add_subdirectory(library)

if(NOT BFR_SANITIZE STREQUAL "none")
    add_subdirectory(sanitizers)
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_subdirectory(assertions)
endif()
