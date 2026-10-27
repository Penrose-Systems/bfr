# The C++ wrapper: bfr.hpp, generated from api/bfr.yaml, over bfr.h. Header
# only, so it adds nothing to either library's exports; link it with bfr or
# bfr_shared, which provide bfr.h.
add_library(bfr_cpp INTERFACE)
target_include_directories(bfr_cpp INTERFACE include)
target_compile_features(bfr_cpp INTERFACE cxx_std_23)

if(BFR_BUILD_TESTS)
    add_subdirectory(tests)
endif()
