# A user's project that builds BFR as part of its own build.
cmake_minimum_required(VERSION 3.28)
project(subproject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(${BFR_SOURCE_DIR} bfr)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE bfr)
