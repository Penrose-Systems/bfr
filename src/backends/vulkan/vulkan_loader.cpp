#include "vulkan/vulkan_loader.h"

#include "bfr/core/error.h"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <bit>
#include <expected>
#include <format>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
    #include <windows.h>

    #include <array>
    #include <system_error>
#else
    #include <dlfcn.h>
#endif

#if defined(__APPLE__)
    #include <cstdlib>
#endif

namespace bfr::rhi::vulkan {

namespace {

constexpr const char* kEntryPoint = "vkGetInstanceProcAddr";

// The libraries volk would try, in its order.
std::vector<const char*> libraries() {
#if defined(_WIN32)
    return {"vulkan-1.dll"};
#elif defined(__APPLE__)
    std::vector<const char*> names{"libvulkan.dylib", "libvulkan.1.dylib"};
    // The Vulkan SDK installs here, which macOS doesn't search unless the user
    // set their own fallback path.
    if (std::getenv("DYLD_FALLBACK_LIBRARY_PATH") == nullptr) {
        names.push_back("/usr/local/lib/libvulkan.dylib");
    }
    names.insert(names.end(), {"libMoltenVK.dylib", "vulkan.framework/vulkan", "MoltenVK.framework/MoltenVK"});
    return names;
#else
    return {"libvulkan.so.1", "libvulkan.so"};
#endif
}

#if defined(_WIN32)

using Library = HMODULE;

std::expected<Library, std::string> openLibrary(const char* name) {
    if (Library library = LoadLibraryA(name)) {
        return library;
    }
    return std::unexpected(
        std::format("{}: {}", name, std::system_category().message(static_cast<int>(GetLastError()))));
}

std::expected<PFN_vkGetInstanceProcAddr, std::string> entryPoint(Library library) {
    if (FARPROC entry = GetProcAddress(library, kEntryPoint)) {
        return std::bit_cast<PFN_vkGetInstanceProcAddr>(entry);
    }
    const std::string reason = std::system_category().message(static_cast<int>(GetLastError()));
    std::array<char, MAX_PATH> path{};
    GetModuleFileNameA(library, path.data(), static_cast<DWORD>(path.size()));
    return std::unexpected(std::format("{}: {}: {}", path.data(), kEntryPoint, reason));
}

void closeLibrary(Library library) {
    FreeLibrary(library);
}

#else

using Library = void*;

std::expected<Library, std::string> openLibrary(const char* name) {
    if (Library library = dlopen(name, RTLD_NOW | RTLD_LOCAL)) {
        return library;
    }
    // Names the file it found, or the name when it found none.
    return std::unexpected(std::string{dlerror()});
}

std::expected<PFN_vkGetInstanceProcAddr, std::string> entryPoint(Library library) {
    if (void* entry = dlsym(library, kEntryPoint)) {
        return std::bit_cast<PFN_vkGetInstanceProcAddr>(entry);
    }
    // Names the file and the symbol; empty when the symbol is there but null.
    const char* error = dlerror();
    return std::unexpected(error != nullptr ? std::string{error} : std::format("{} is null", kEntryPoint));
}

void closeLibrary(Library library) {
    dlclose(library);
}

#endif

} // namespace

core::Result<void> openLoader() {
    // volkInitializeCustom rewrites volk's global functions, so it runs once
    // and the rest see the same ones. A failure is tried again.
    static std::mutex mutex;
    static bool opened = false;
    const std::scoped_lock lock{mutex};
    if (opened) {
        return {};
    }
    std::vector<std::string> reasons;
    for (const char* name : libraries()) {
        auto library = openLibrary(name);
        if (!library) {
            reasons.push_back(library.error());
            continue;
        }
        auto entry = entryPoint(*library);
        if (!entry) {
            reasons.push_back(entry.error());
            closeLibrary(*library);
            continue;
        }
        volkInitializeCustom(*entry);
        opened = true;
        return {};
    }
    return core::failure(core::ErrorCode::RhiLoaderFailed,
                         core::error::withReasons("no Vulkan loader would load", reasons));
}

} // namespace bfr::rhi::vulkan
