# The program indexes past the end of a vector. Its test passes only if the
# standard library's assertions stop it, so a Debug build without them fails.
# clang-tidy skips it, since it would flag the bug.
add_executable(index_out_of_range index_out_of_range.cpp)
bfr_apply_compiler_flags(index_out_of_range)
set_target_properties(index_out_of_range PROPERTIES CXX_CLANG_TIDY "")
target_link_libraries(index_out_of_range PRIVATE bfr_sanitizer_options)
add_test(NAME assertions.stdlib_detects_index_out_of_range COMMAND index_out_of_range)
set_tests_properties(assertions.stdlib_detects_index_out_of_range PROPERTIES
    PASS_REGULAR_EXPRESSION "Assertion '__n < this->size\\(\\)' failed")
