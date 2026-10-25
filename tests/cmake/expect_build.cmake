# Configures FIXTURE in BINARY_DIR with -DFIXTURE_MODE=MODE and builds TARGET.
# EXPECT is "pass" or "fail"; on "fail" the configure or build output must
# contain EXPECT_OUTPUT, so it fails for the intended reason.
function(expect_failure_output output)
    string(FIND "${output}" "${EXPECT_OUTPUT}" found)
    if(found EQUAL -1)
        message(FATAL_ERROR "failed without '${EXPECT_OUTPUT}':\n${output}")
    endif()
endfunction()

execute_process(
    COMMAND ${CMAKE_COMMAND} -S ${FIXTURE} -B ${BINARY_DIR} -G Ninja
            -DBFR_SOURCE_DIR=${BFR_SOURCE_DIR} -DFIXTURE_MODE=${MODE}
            -DCMAKE_CXX_COMPILER=${CXX_COMPILER}
    RESULT_VARIABLE configure_result
    OUTPUT_VARIABLE configure_output
    ERROR_VARIABLE configure_output)
if(NOT configure_result EQUAL 0)
    if(EXPECT STREQUAL "fail")
        expect_failure_output("${configure_output}")
        return()
    endif()
    message(FATAL_ERROR "configure failed:\n${configure_output}")
endif()

execute_process(
    COMMAND ${CMAKE_COMMAND} --build ${BINARY_DIR} --target ${TARGET}
    RESULT_VARIABLE build_result
    OUTPUT_VARIABLE build_output
    ERROR_VARIABLE build_output)

if(EXPECT STREQUAL "pass" AND NOT build_result EQUAL 0)
    message(FATAL_ERROR "build failed:\n${build_output}")
endif()
if(EXPECT STREQUAL "fail")
    if(build_result EQUAL 0)
        message(FATAL_ERROR "build passed; expected it to fail")
    endif()
    expect_failure_output("${build_output}")
endif()
