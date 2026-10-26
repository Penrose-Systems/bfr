# The C API as its users see it: through the shared library, from C.

# The committed bfr.h matches api/bfr.yaml.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME c_api.header_up_to_date COMMAND Python3::Interpreter ${PROJECT_SOURCE_DIR}/tools/gen_c_api.py --check)

add_executable(bfr_links_from_c links_from_c.c)
bfr_apply_compiler_flags(bfr_links_from_c)
target_link_libraries(bfr_links_from_c PRIVATE bfr_shared bfr_sanitizer_options)
add_test(NAME c_api.links_from_c COMMAND bfr_links_from_c)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_test(NAME c_api.exports_match_api
        COMMAND ${CMAKE_COMMAND} -DNM=${CMAKE_NM} -DLIBRARY=$<TARGET_FILE:bfr_shared>
            -DPYTHON=${Python3_EXECUTABLE} -DGENERATOR=${PROJECT_SOURCE_DIR}/tools/gen_c_api.py
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exports.cmake)
endif()
