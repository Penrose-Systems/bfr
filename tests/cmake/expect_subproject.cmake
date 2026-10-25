# Configures FIXTURE, a user's project that builds BFR with add_subdirectory,
# in BINARY_DIR, with VCPKG_ROOT set. CHECK names what must hold: "link", its
# Debug app builds and links; "build_type", the build type stays the user's,
# unset; "toolchain", BFR sets no toolchain file.
# A script starts with no policies, and CHECK's values must stay strings.
cmake_policy(VERSION 3.28)
file(REMOVE_RECURSE ${BINARY_DIR})
set(ENV{VCPKG_ROOT} ${BINARY_DIR}/vcpkg)
set(build_type "")
if(CHECK STREQUAL "link")
    set(build_type Debug)
endif()
execute_process(
    COMMAND ${CMAKE_COMMAND} -S ${FIXTURE} -B ${BINARY_DIR} -G Ninja
            -DBFR_SOURCE_DIR=${BFR_SOURCE_DIR}
            -DCMAKE_CXX_COMPILER=${CXX_COMPILER}
            -DCMAKE_PREFIX_PATH=${PREFIX_PATH}
            -DCMAKE_BUILD_TYPE=${build_type}
    RESULT_VARIABLE configure_result
    OUTPUT_VARIABLE configure_output
    ERROR_VARIABLE configure_output)
if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR "configure failed:\n${configure_output}")
endif()

if(CHECK STREQUAL "link")
    execute_process(
        COMMAND ${CMAKE_COMMAND} --build ${BINARY_DIR} --target app
        RESULT_VARIABLE build_result
        OUTPUT_VARIABLE build_output
        ERROR_VARIABLE build_output)
    if(NOT build_result EQUAL 0)
        message(FATAL_ERROR "build failed:\n${build_output}")
    endif()
elseif(CHECK STREQUAL "build_type")
    file(STRINGS ${BINARY_DIR}/CMakeCache.txt found REGEX "^CMAKE_BUILD_TYPE:STRING=$")
    if(NOT found)
        message(FATAL_ERROR "expected the build type left unset")
    endif()
elseif(CHECK STREQUAL "toolchain")
    file(STRINGS ${BINARY_DIR}/CMakeCache.txt found REGEX "^CMAKE_TOOLCHAIN_FILE:")
    if(found)
        message(FATAL_ERROR "expected no toolchain file, found ${found}")
    endif()
else()
    message(FATAL_ERROR "CHECK is '${CHECK}'; use link, build_type or toolchain")
endif()
