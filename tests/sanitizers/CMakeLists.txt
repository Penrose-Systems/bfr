# Each program has one deliberate bug. A test passes only if its sanitizer
# reports the bug, so a sanitizer build that lost its flags fails. clang-tidy
# skips them, since it would flag the bugs.
if(BFR_SANITIZE STREQUAL "thread")
    find_package(Threads REQUIRED)
    add_executable(data_race data_race.cpp)
    target_link_libraries(data_race PRIVATE bfr_sanitizer_options Threads::Threads)
    add_test(NAME sanitizer.tsan_detects_data_race COMMAND data_race)
    set_tests_properties(sanitizer.tsan_detects_data_race PROPERTIES
        PASS_REGULAR_EXPRESSION "ThreadSanitizer: data race")
    bfr_apply_compiler_flags(data_race)
    set_target_properties(data_race PROPERTIES CXX_CLANG_TIDY "")
    return()
endif()

add_executable(heap_overflow heap_overflow.cpp)
add_test(NAME sanitizer.asan_detects_heap_overflow COMMAND heap_overflow)
set_tests_properties(sanitizer.asan_detects_heap_overflow PROPERTIES
    PASS_REGULAR_EXPRESSION "AddressSanitizer: heap-buffer-overflow")

add_executable(signed_overflow signed_overflow.cpp)
add_test(NAME sanitizer.ubsan_detects_signed_overflow COMMAND signed_overflow)
set_tests_properties(sanitizer.ubsan_detects_signed_overflow PROPERTIES
    PASS_REGULAR_EXPRESSION "runtime error: signed integer overflow")

add_executable(leak leak.cpp)
add_test(NAME sanitizer.lsan_detects_leak COMMAND leak)
set_tests_properties(sanitizer.lsan_detects_leak PROPERTIES
    PASS_REGULAR_EXPRESSION "LeakSanitizer: detected memory leaks")

add_executable(float_divide_by_zero float_divide_by_zero.cpp)
add_test(NAME sanitizer.ubsan_detects_float_divide_by_zero COMMAND float_divide_by_zero)
set_tests_properties(sanitizer.ubsan_detects_float_divide_by_zero PROPERTIES
    PASS_REGULAR_EXPRESSION "runtime error: division by zero")

add_executable(init_order init_order.cpp init_order_source.cpp)
add_test(NAME sanitizer.asan_detects_init_order COMMAND init_order)
set_tests_properties(sanitizer.asan_detects_init_order PROPERTIES
    PASS_REGULAR_EXPRESSION "AddressSanitizer: initialization-order-fiasco")

foreach(probe heap_overflow signed_overflow leak float_divide_by_zero init_order)
    bfr_apply_compiler_flags(${probe})
    set_target_properties(${probe} PROPERTIES CXX_CLANG_TIDY "")
    target_link_libraries(${probe} PRIVATE bfr_sanitizer_options)
endforeach()
