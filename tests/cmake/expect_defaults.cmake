# Configures the real project with nothing but its directories and compiler,
# as a plain `cmake -B build` does, then checks the defaults it gets: the vcpkg
# toolchain from VCPKG_ROOT, a Debug build, the sanitizers, the compile
# commands that clangd and clang-tidy read, and pattern-initialised locals.
file(REMOVE_RECURSE ${BINARY_DIR})
execute_process(
    COMMAND ${CMAKE_COMMAND} -S ${BFR_SOURCE_DIR} -B ${BINARY_DIR} -G Ninja
            -DCMAKE_CXX_COMPILER=${CXX_COMPILER}
    RESULT_VARIABLE configure_result
    OUTPUT_VARIABLE configure_output
    ERROR_VARIABLE configure_output)
if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR "configure failed:\n${configure_output}")
endif()

foreach(entry "CMAKE_BUILD_TYPE:STRING=Debug" "BFR_SANITIZE:STRING=address")
    file(STRINGS ${BINARY_DIR}/CMakeCache.txt found REGEX "^${entry}$")
    if(NOT found)
        message(FATAL_ERROR "expected ${entry} in the cache")
    endif()
endforeach()
if(NOT EXISTS ${BINARY_DIR}/compile_commands.json)
    message(FATAL_ERROR "expected ${BINARY_DIR}/compile_commands.json")
endif()
file(READ ${BINARY_DIR}/compile_commands.json commands)
if(NOT commands MATCHES "-ftrivial-auto-var-init=pattern")
    message(FATAL_ERROR "expected -ftrivial-auto-var-init=pattern in the Debug compile commands")
endif()
