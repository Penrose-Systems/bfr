if(BFR_CLANG_TIDY)
    find_program(BFR_CLANG_TIDY_EXE NAMES clang-tidy-19 clang-tidy REQUIRED)
endif()

# The strict warnings, as errors, for the library and every program built from
# our own sources, tests included; with BFR_CLANG_TIDY, clang-tidy too, under
# the nearest .clang-tidy.
function(bfr_apply_compiler_flags target)
    target_compile_options(${target} PRIVATE
        -Wall -Wextra -Wpedantic -Werror -Wconversion -Wsign-conversion
        -Wnull-dereference -Wdouble-promotion -Wformat=2
        -Wimplicit-fallthrough)
    if(BFR_CLANG_TIDY)
        set_target_properties(${target} PROPERTIES CXX_CLANG_TIDY ${BFR_CLANG_TIDY_EXE})
    endif()
endfunction()

# Every public header compiles on its own: <target>_headers compiles one
# generated source per header under public_dir that includes only that header,
# by its path from public_dir, with what linking target provides and nothing
# else. It is not part of target.
function(bfr_add_header_check target public_dir)
    file(GLOB_RECURSE headers CONFIGURE_DEPENDS ${public_dir}/*.h)
    if(NOT headers)
        return()
    endif()
    set(checks "")
    foreach(header ${headers})
        file(RELATIVE_PATH name ${public_dir} ${header})
        set(check ${CMAKE_CURRENT_BINARY_DIR}/header_checks/${target}/${name}.cpp)
        file(CONFIGURE OUTPUT ${check} CONTENT "#include \"${name}\" // IWYU pragma: keep\n")
        list(APPEND checks ${check})
    endforeach()
    add_library(${target}_headers OBJECT ${checks})
    target_link_libraries(${target}_headers PRIVATE ${target})
    bfr_apply_compiler_flags(${target}_headers)
endfunction()

# A C header also compiles on its own as C: <target>_c_headers compiles one
# generated .c source that includes only header, with the strict warnings.
function(bfr_add_c_header_check target header)
    set(check ${CMAKE_CURRENT_BINARY_DIR}/header_checks/${target}/${header}.c)
    file(CONFIGURE OUTPUT ${check} CONTENT "#include \"${header}\"\n")
    add_library(${target}_c_headers OBJECT ${check})
    target_link_libraries(${target}_c_headers PRIVATE ${target})
    bfr_apply_compiler_flags(${target}_c_headers)
endfunction()
