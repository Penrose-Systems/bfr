#include "vulkan/instance_table.h"

#include <volk.h>
#include <vulkan/vulkan_core.h>

#include <bit>
#include <mutex>

namespace bfr::rhi::vulkan {

namespace {

// volk loads this through the first instance. It is the loader's own function,
// so it stays valid when that instance goes.
void loadDeviceProcAddr(VkInstance instance) {
    static std::once_flag once;
    std::call_once(once, [instance] {
        vkGetDeviceProcAddr = std::bit_cast<PFN_vkGetDeviceProcAddr>(
            vkGetInstanceProcAddr(instance, "vkGetDeviceProcAddr"));
    });
}

} // namespace

InstanceTable loadInstanceTable(VkInstance instance) {
    loadDeviceProcAddr(instance);
    InstanceTable table;
#define BFR_LOAD(name) table.name = std::bit_cast<PFN_##name>(vkGetInstanceProcAddr(instance, #name))
    BFR_LOAD(vkDestroyInstance);
    BFR_LOAD(vkEnumeratePhysicalDevices);
    BFR_LOAD(vkEnumerateDeviceExtensionProperties);
    BFR_LOAD(vkCreateDevice);
    BFR_LOAD(vkDestroySurfaceKHR);
    BFR_LOAD(vkCreateDebugUtilsMessengerEXT);
    BFR_LOAD(vkDestroyDebugUtilsMessengerEXT);
    BFR_LOAD(vkCmdBeginDebugUtilsLabelEXT);
    BFR_LOAD(vkCmdEndDebugUtilsLabelEXT);
    BFR_LOAD(vkCmdInsertDebugUtilsLabelEXT);
    BFR_LOAD(vkQueueBeginDebugUtilsLabelEXT);
    BFR_LOAD(vkQueueEndDebugUtilsLabelEXT);
    BFR_LOAD(vkQueueInsertDebugUtilsLabelEXT);
    BFR_LOAD(vkGetPhysicalDeviceProperties);
    BFR_LOAD(vkGetPhysicalDeviceProperties2);
    BFR_LOAD(vkGetPhysicalDeviceFeatures2);
    BFR_LOAD(vkGetPhysicalDeviceFormatProperties2);
    BFR_LOAD(vkGetPhysicalDeviceImageFormatProperties2);
    BFR_LOAD(vkGetPhysicalDeviceMemoryProperties);
    BFR_LOAD(vkGetPhysicalDeviceMemoryProperties2);
    BFR_LOAD(vkGetPhysicalDeviceQueueFamilyProperties);
    BFR_LOAD(vkGetPhysicalDeviceQueueFamilyProperties2);
    BFR_LOAD(vkGetPhysicalDeviceSparseImageFormatProperties2);
    BFR_LOAD(vkGetPhysicalDeviceCalibrateableTimeDomainsKHR);
    if (table.vkGetPhysicalDeviceCalibrateableTimeDomainsKHR == nullptr) {
        table.vkGetPhysicalDeviceCalibrateableTimeDomainsKHR =
            std::bit_cast<PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR>(
                vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCalibrateableTimeDomainsEXT"));
    }
    BFR_LOAD(vkGetPhysicalDeviceSurfaceSupportKHR);
    BFR_LOAD(vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
    BFR_LOAD(vkGetPhysicalDeviceSurfaceFormatsKHR);
    BFR_LOAD(vkGetPhysicalDeviceSurfacePresentModesKHR);
#undef BFR_LOAD
    return table;
}

} // namespace bfr::rhi::vulkan
