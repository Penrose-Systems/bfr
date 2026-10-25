# Each test builds one target of the library fixture with the real build
# helpers.
function(bfr_add_build_test name target mode expect expect_output)
    add_test(NAME ${name}
        COMMAND ${CMAKE_COMMAND}
            -DFIXTURE=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/library
            -DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/${name}
            -DBFR_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            -DCXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DTARGET=${target}
            -DMODE=${mode}
            -DEXPECT=${expect}
            -DEXPECT_OUTPUT=${expect_output}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_build.cmake)
endfunction()

bfr_add_build_test(library.warning_fails w WARN fail unused-parameter)
bfr_add_build_test(executable.warning_fails warned_app WARN fail unused-parameter)
bfr_add_build_test(header.self_contained_compiles a_headers "" pass "")
bfr_add_build_test(header.missing_include_fails h_headers HEADER fail size_t)
# Only where the build itself runs clang-tidy, so the tool is present.
if(BFR_CLANG_TIDY)
    bfr_add_build_test(tidy.clean_library_passes a TIDY pass "")
    bfr_add_build_test(tidy.raw_new_fails t TIDY fail cppcoreguidelines-owning-memory)
endif()

add_test(NAME defaults.plain_configure
    COMMAND ${CMAKE_COMMAND}
        -DBFR_SOURCE_DIR=${PROJECT_SOURCE_DIR}
        -DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/defaults
        -DCXX_COMPILER=${CMAKE_CXX_COMPILER}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_defaults.cmake)

# A user's project that adds BFR with add_subdirectory; check is what must
# hold, as expect_subproject.cmake lists.
function(bfr_add_subproject_test name check)
    add_test(NAME ${name}
        COMMAND ${CMAKE_COMMAND}
            -DFIXTURE=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/subproject
            -DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/${name}
            -DBFR_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            -DCXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DPREFIX_PATH=${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}
            -DCHECK=${check}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_subproject.cmake)
endfunction()

bfr_add_subproject_test(subproject.debug_links link)
bfr_add_subproject_test(subproject.leaves_build_type build_type)
bfr_add_subproject_test(subproject.leaves_toolchain toolchain)

# Each test configures the real project with fake launchers and checks which
# one the compile rules use. The launchers are shell scripts and the targets
# are the sanitizer probes.
function(bfr_add_ccache_test name use user_launcher expect)
    add_test(NAME ${name}
        COMMAND ${CMAKE_COMMAND}
            -DBFR_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            -DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/${name}
            -DCXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DTOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
            -DUSE=${use}
            -DUSER_LAUNCHER=${user_launcher}
            -DEXPECT=${expect}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_ccache.cmake)
endfunction()

bfr_add_ccache_test(ccache.used_when_installed ON OFF ccache)
bfr_add_ccache_test(ccache.skipped_when_off OFF OFF none)
bfr_add_ccache_test(ccache.user_launcher_wins ON ON user)
