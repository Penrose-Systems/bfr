# A library built with the real build helpers. FIXTURE_MODE=WARN adds library w
# and executable warned_app, whose sources have a warning. FIXTURE_MODE=HEADER
# adds library h, whose public header compiles only after its source's
# includes. FIXTURE_MODE=TIDY runs clang-tidy and adds library t, whose source
# has a raw new.
cmake_minimum_required(VERSION 3.28)
project(library LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(FIXTURE_MODE STREQUAL "TIDY")
    set(BFR_CLANG_TIDY ON)
endif()

include(${BFR_SOURCE_DIR}/cmake/bfr_build.cmake)

add_library(a STATIC a.cpp)
bfr_apply_compiler_flags(a)
target_include_directories(a PUBLIC public)
bfr_add_header_check(a ${CMAKE_CURRENT_SOURCE_DIR}/public)

if(FIXTURE_MODE STREQUAL "WARN")
    add_library(w STATIC w.cpp)
    bfr_apply_compiler_flags(w)
    add_executable(warned_app warned.cpp)
    bfr_apply_compiler_flags(warned_app)
endif()

if(FIXTURE_MODE STREQUAL "HEADER")
    add_library(h STATIC h.cpp)
    bfr_apply_compiler_flags(h)
    target_include_directories(h PUBLIC h_public)
    bfr_add_header_check(h ${CMAKE_CURRENT_SOURCE_DIR}/h_public)
endif()

if(FIXTURE_MODE STREQUAL "TIDY")
    add_library(t STATIC t.cpp)
    bfr_apply_compiler_flags(t)
endif()
