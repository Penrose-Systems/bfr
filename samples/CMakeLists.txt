# Programs that show how to use BFR, through the C++ wrapper and the shared
# library, as an application would.
add_executable(bfr_caps caps/main.cpp)
bfr_apply_compiler_flags(bfr_caps)
target_link_libraries(bfr_caps PRIVATE bfr_cpp bfr_shared)
