// Sanitizer options for every test program, compiled in so they apply however
// the program is started. A value the caller puts in ASAN_OPTIONS, UBSAN_OPTIONS
// or TSAN_OPTIONS still wins. Builds without that sanitizer never call these.
#include <cstdlib>

namespace {

// Ubuntu points DEBUGINFOD_URLS at its server, so Clang's symbolizer fetches
// debug info for system libraries before it prints a report, and hangs with no
// network. Reports name our functions without it. Priority 101 runs this before
// every ordinary initialiser, so a report raised by one is covered too.
[[gnu::constructor(101)]] void clearDebuginfod() {
    unsetenv("DEBUGINFOD_URLS");
}

} // namespace

extern "C" {

// Init order: a global read by another file's initialiser before its own runs.
// handle_abort: a stack trace when anything aborts, such as a failed assertion.
// detect_leaks and detect_stack_use_after_return stay explicit because their
// defaults have changed between runtime versions.
// strict_string_checks stays off (#53): Mesa's X11 code hands strncmp a name with
// no terminator. Reads past the end are still reported.
const char* __asan_default_options() {
    return "detect_leaks=1:detect_stack_use_after_return=1:"
           "check_initialization_order=1:strict_init_order=1:handle_abort=1";
}

// Leaks in code we don't own, one rule per bug. LeakSanitizer lists the rules it
// used under "Suppressions used" at exit; once a rule stops appearing there and
// its issue is fixed, delete it.
const char* __lsan_default_suppressions() {
    // #50: NVIDIA's driver (580.178.04) leaks memory libdbus allocates while the
    // driver registers on the session bus: from libnvidia-glcore when it builds
    // an X11 swapchain, and from libnvidia-eglcore during driver setup without
    // DISPLAY. The driver unloads libnvidia-eglcore itself before exit, so a rule
    // naming that library can't match; this one names the libdbus call. It hides
    // any leak allocated inside bus registration, and whatever only that leak
    // holds. Nothing of ours registers on the bus.
    return "leak:^dbus_bus_register$\n";
}

// The whole stack for undefined behaviour, not only the line it happened on.
const char* __ubsan_default_options() {
    return "print_stacktrace=1";
}

// Both stacks in a lock-order report.
const char* __tsan_default_options() {
    return "second_deadlock_stack=1";
}

} // extern "C"
