# Configures the real project with fake launchers, then checks which one the
# generated compile rules go through. The sanitizer tests supply the targets.
# USE is BFR_USE_CCACHE, USER_LAUNCHER is ON when the user passes their own
# C and C++ launchers, and EXPECT is "ccache", "user" or "none".
set(tools ${BINARY_DIR}_tools)
file(REMOVE_RECURSE ${BINARY_DIR} ${tools})
foreach(tool ccache user)
    file(WRITE ${tools}/${tool} "#!/bin/sh\nexec \"$@\"\n")
    file(CHMOD ${tools}/${tool}
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)
endforeach()

set(user_arg "")
if(USER_LAUNCHER)
    set(user_arg -DCMAKE_C_COMPILER_LAUNCHER=${tools}/user -DCMAKE_CXX_COMPILER_LAUNCHER=${tools}/user)
endif()

execute_process(
    COMMAND ${CMAKE_COMMAND} -S ${BFR_SOURCE_DIR} -B ${BINARY_DIR} -G Ninja
            -DBFR_SANITIZE=address -DBFR_USE_CCACHE=${USE}
            -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE}
            -DCMAKE_PROGRAM_PATH=${tools}
            -DCMAKE_CXX_COMPILER=${CXX_COMPILER} ${user_arg}
    RESULT_VARIABLE configure_result
    OUTPUT_VARIABLE configure_output
    ERROR_VARIABLE configure_output)
if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR "configure failed:\n${configure_output}")
endif()

file(READ ${BINARY_DIR}/build.ninja rules)
string(FIND "${rules}" "${tools}/ccache" in_ccache)
string(FIND "${rules}" "${tools}/user" in_user)
if(EXPECT STREQUAL "ccache" AND (in_ccache EQUAL -1 OR NOT in_user EQUAL -1))
    message(FATAL_ERROR "expected compiles through ccache only")
elseif(EXPECT STREQUAL "user" AND (in_user EQUAL -1 OR NOT in_ccache EQUAL -1))
    message(FATAL_ERROR "expected compiles through the user launcher only")
elseif(EXPECT STREQUAL "none" AND (NOT in_ccache EQUAL -1 OR NOT in_user EQUAL -1))
    message(FATAL_ERROR "expected compiles through no launcher")
endif()
