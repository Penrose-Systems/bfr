// Built as libvulkan.so.1: a loader that misbehaves as BFR_FAKE_LOADER_MODE
// says:
// - "layers" or "extensions": that listing fails with VK_ERROR_UNKNOWN.
// - "grow": each listing gains an entry once counted, as if one were installed
//   meanwhile, so its first fill returns VK_INCOMPLETE.
// It lists the validation layer and VK_EXT_debug_utils, so the Rhi's checks
// pass, and vkCreateInstance fails with VK_ERROR_INITIALIZATION_FAILED. It
// serves only those three functions.

#include <vulkan/vk_platform.h>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <string_view>
#include <vector>

namespace {

bool inMode(std::string_view mode) {
    const char* current = std::getenv("BFR_FAKE_LOADER_MODE");
    return current != nullptr && mode == current;
}

// A listing answered in two calls, count then fill.
template <class Properties>
struct Listing {
    std::vector<Properties> entries;
    bool grown = false;

    VkResult answer(std::uint32_t* count, Properties* properties) {
        const auto listed = static_cast<std::uint32_t>(entries.size());
        if (properties == nullptr) {
            *count = listed;
            if (inMode("grow") && !grown) {
                grown = true;
                entries.emplace_back();
            }
            return VK_SUCCESS;
        }
        const std::uint32_t written = std::min(*count, listed);
        std::copy_n(entries.begin(), written, properties);
        *count = written;
        return written < listed ? VK_INCOMPLETE : VK_SUCCESS;
    }
};

VkLayerProperties layerNamed(std::string_view name) {
    VkLayerProperties layer{};
    name.copy(layer.layerName, VK_MAX_EXTENSION_NAME_SIZE - 1);
    return layer;
}

VkExtensionProperties extensionNamed(std::string_view name) {
    VkExtensionProperties extension{};
    name.copy(extension.extensionName, VK_MAX_EXTENSION_NAME_SIZE - 1);
    return extension;
}

Listing<VkLayerProperties> layers{.entries = {layerNamed("VK_LAYER_KHRONOS_validation")}};
Listing<VkExtensionProperties> extensions{.entries = {extensionNamed(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)}};

VKAPI_ATTR VkResult VKAPI_CALL enumerateInstanceLayerProperties(std::uint32_t* count, VkLayerProperties* properties) {
    if (inMode("layers")) {
        return VK_ERROR_UNKNOWN;
    }
    return layers.answer(count, properties);
}

VKAPI_ATTR VkResult VKAPI_CALL enumerateInstanceExtensionProperties(const char*, std::uint32_t* count,
                                                                    VkExtensionProperties* properties) {
    if (inMode("extensions")) {
        return VK_ERROR_UNKNOWN;
    }
    return extensions.answer(count, properties);
}

VKAPI_ATTR VkResult VKAPI_CALL createInstance(const VkInstanceCreateInfo*, const VkAllocationCallbacks*, VkInstance*) {
    return VK_ERROR_INITIALIZATION_FAILED;
}

} // namespace

extern "C" VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char* pName) {
    const std::string_view function{pName};
    if (instance != VK_NULL_HANDLE) {
        return nullptr;
    }
    if (function == "vkEnumerateInstanceLayerProperties") {
        return std::bit_cast<PFN_vkVoidFunction>(&enumerateInstanceLayerProperties);
    }
    if (function == "vkEnumerateInstanceExtensionProperties") {
        return std::bit_cast<PFN_vkVoidFunction>(&enumerateInstanceExtensionProperties);
    }
    if (function == "vkCreateInstance") {
        return std::bit_cast<PFN_vkVoidFunction>(&createInstance);
    }
    return nullptr;
}
