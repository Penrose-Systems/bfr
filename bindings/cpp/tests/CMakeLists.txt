# The C++ wrapper as its users see it.

# The committed bfr.hpp matches api/bfr.yaml.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME cpp_api.header_up_to_date COMMAND Python3::Interpreter ${PROJECT_SOURCE_DIR}/tools/gen_cpp_api.py --check)

# bfr.hpp, its templates instantiated, compiles without exceptions or RTTI, as
# engines often build. Compiled only.
add_library(bfr_cpp_no_exceptions OBJECT compiles_without_exceptions.cpp)
bfr_apply_compiler_flags(bfr_cpp_no_exceptions)
target_compile_options(bfr_cpp_no_exceptions PRIVATE -fno-exceptions -fno-rtti)
target_link_libraries(bfr_cpp_no_exceptions PRIVATE bfr_cpp bfr_shared)
