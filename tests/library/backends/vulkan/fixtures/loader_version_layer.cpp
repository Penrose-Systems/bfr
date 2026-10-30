// An implicit Vulkan layer that reports the loader as the version
// BFR_LOADER_VERSION gives ("1.4.319"), so the RHI tests can run what
// depends on the loader's version on any loader. Only an implicit layer can
// answer vkEnumerateInstanceVersion, which comes before any instance. It passes
// everything else through.

#include "layer_link.h"

#include <vulkan/vk_layer.h>
#include <vulkan/vk_platform.h>
#include <vulkan/vulkan_core.h>

#include <array>
#include <bit>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <string_view>
#include <system_error>

namespace {

PFN_vkGetInstanceProcAddr nextGetInstanceProcAddr = nullptr;
PFN_vkGetDeviceProcAddr nextGetDeviceProcAddr = nullptr;

VKAPI_ATTR VkResult VKAPI_CALL createInstance(const VkInstanceCreateInfo* info, const VkAllocationCallbacks* allocator,
                                              VkInstance* instance) {
    auto* link = linkInfo<VkLayerInstanceCreateInfo>(info->pNext, VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO);
    if (link == nullptr) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    nextGetInstanceProcAddr = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    link->u.pLayerInfo = link->u.pLayerInfo->pNext;
    const auto next = std::bit_cast<PFN_vkCreateInstance>(nextGetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance"));
    return next(info, allocator, instance);
}

VKAPI_ATTR VkResult VKAPI_CALL createDevice(VkPhysicalDevice adapter, const VkDeviceCreateInfo* info,
                                            const VkAllocationCallbacks* allocator, VkDevice* device) {
    auto* link = linkInfo<VkLayerDeviceCreateInfo>(info->pNext, VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO);
    if (link == nullptr) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    const PFN_vkGetInstanceProcAddr nextInstanceProcAddr = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    nextGetDeviceProcAddr = link->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    link->u.pLayerInfo = link->u.pLayerInfo->pNext;
    const auto next = std::bit_cast<PFN_vkCreateDevice>(nextInstanceProcAddr(VK_NULL_HANDLE, "vkCreateDevice"));
    return next(adapter, info, allocator, device);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL getDeviceProcAddr(VkDevice device, const char* name) {
    if (std::string_view{name} == "vkGetDeviceProcAddr") {
        return std::bit_cast<PFN_vkVoidFunction>(&getDeviceProcAddr);
    }
    return nextGetDeviceProcAddr(device, name);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL getInstanceProcAddr(VkInstance instance, const char* name) {
    const std::string_view function{name};
    if (function == "vkGetInstanceProcAddr") {
        return std::bit_cast<PFN_vkVoidFunction>(&getInstanceProcAddr);
    }
    if (function == "vkGetDeviceProcAddr") {
        return std::bit_cast<PFN_vkVoidFunction>(&getDeviceProcAddr);
    }
    if (function == "vkCreateInstance") {
        return std::bit_cast<PFN_vkVoidFunction>(&createInstance);
    }
    if (function == "vkCreateDevice") {
        return std::bit_cast<PFN_vkVoidFunction>(&createDevice);
    }
    return nextGetInstanceProcAddr(instance, name);
}

// "major.minor.patch" as a packed Vulkan version; false when malformed.
bool parseVersion(std::string_view text, std::uint32_t& version) {
    std::array<std::uint32_t, 3> parts{};
    const char* position = text.data();
    const char* const end = text.data() + text.size();
    for (std::size_t part = 0; part < parts.size(); ++part) {
        const auto [after, error] = std::from_chars(position, end, parts.at(part));
        const bool separated = part + 1 == parts.size() ? after == end : after != end && *after == '.';
        if (error != std::errc{} || !separated) {
            return false;
        }
        position = after + 1;
    }
    version = VK_MAKE_API_VERSION(0, parts[0], parts[1], parts[2]);
    return true;
}

} // namespace

extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* pVersionStruct) {
    // Version 2 lets the loader take the layer's entry points from this struct.
    constexpr std::uint32_t kInterfaceVersion = 2;
    pVersionStruct->loaderLayerInterfaceVersion = kInterfaceVersion;
    pVersionStruct->pfnGetInstanceProcAddr = getInstanceProcAddr;
    pVersionStruct->pfnGetDeviceProcAddr = getDeviceProcAddr;
    pVersionStruct->pfnGetPhysicalDeviceProcAddr = nullptr;
    return VK_SUCCESS;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL bfrEnumerateInstanceVersion(const VkEnumerateInstanceVersionChain* chain,
                                                                      std::uint32_t* version) {
    const char* reported = std::getenv("BFR_LOADER_VERSION");
    if (reported == nullptr || !parseVersion(reported, *version)) {
        return chain->CallDown(version);
    }
    return VK_SUCCESS;
}
