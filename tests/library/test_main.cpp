#include <catch2/catch_session.hpp>

#include <cstdlib>

namespace {

// Sets name unless the caller's environment already has it.
void setDefault(const char* name, const char* value) {
    constexpr int kKeepExisting = 0;
    setenv(name, value, kKeepExisting);
}

} // namespace

// Every test sees the same environment however it is started (ctest, the binary
// directly, a debugger), and a variable the caller set always wins.
int main(int argc, char* argv[]) {
    // The loader takes vcpkg's validation layer ahead of an older system one. It
    // keeps drivers loaded after the instance goes: unloading one strands the
    // memory its globals still hold, which LeakSanitizer then reports as leaked.
    setDefault("VK_ADD_LAYER_PATH", BFR_TEST_LAYER_PATH);
    setDefault("VK_LOADER_DISABLE_DYNAMIC_LIBRARY_UNLOADING", "1");
#ifdef BFR_TEST_DRIVER_FILES
    // NVIDIA's driver crashes ThreadSanitizer; lavapipe runs clean under it.
    setDefault("VK_DRIVER_FILES", BFR_TEST_DRIVER_FILES);
#endif
    return Catch::Session().run(argc, argv);
}
