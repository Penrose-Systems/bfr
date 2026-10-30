// A Vulkan layer that changes what every adapter reports, and fails calls, for
// the RHI tests:
// - BFR_ADAPTER_MINOR: the adapter reports Vulkan 1.<minor>.
// - BFR_ADAPTER_DRIVER_ID: the adapter reports this VkDriverId.
// - BFR_ADAPTER_VENDOR_ID: the adapter reports this vendor ID.
// - BFR_ADAPTER_NO_FEATURE: the adapter reports this Vulkan 1.2 or 1.3
//   feature, by its member name, as missing.
// - BFR_ADAPTER_HIDE: the adapter leaves this device extension out.
// - BFR_ADAPTER_ADD: the adapter lists this device extension too, and
//   vkCreateDevice drops it, since the driver underneath lacks it.
// - BFR_ADAPTER_NO_COMPUTE: no queue family reports compute.
// - BFR_ADAPTER_NO_GRAPHICS: no queue family reports graphics.
// - BFR_ADAPTER_SEPARATE_PRESENT: the adapter also lists a queue family that
//   does nothing but present, the only one that presents; vkCreateDevice drops
//   its queue, since the driver underneath lacks the family.
// - BFR_PRESENT_UNSUPPORTED: no queue family presents to any surface.
// - BFR_SURFACE_NO_FORMATS: surfaces list no formats, which Vulkan forbids.
// - BFR_FAIL: the named function fails with VK_ERROR_UNKNOWN, or the result
//   BFR_FAIL_RESULT names (VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_DEVICE_LOST,
//   VK_ERROR_SURFACE_LOST_KHR or VK_ERROR_FORMAT_NOT_SUPPORTED), and leaves
//   junk in what it would have returned, as Vulkan allows: one of
//   vkEnumerateDeviceExtensionProperties, vkCreateDevice, vkCreateHeadlessSurfaceEXT,
//   vkGetPhysicalDeviceImageFormatProperties2,
//   vkGetPhysicalDeviceSurfaceSupportKHR,
//   vkGetPhysicalDeviceSurfaceCapabilitiesKHR,
//   vkGetPhysicalDeviceSurfaceFormatsKHR, vkCreateSwapchainKHR and
//   vkGetSwapchainImagesKHR.
// Each call reads the variables, so a test can change them between calls; an
// unset one changes nothing. It serves one instance and one device at a time,
// which is all a test makes.

#include "layer_link.h"

#include <vulkan/vk_layer.h>
#include <vulkan/vk_platform.h>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

PFN_vkGetInstanceProcAddr nextGetInstanceProcAddr = nullptr;
PFN_vkGetDeviceProcAddr nextGetDeviceProcAddr = nullptr;
PFN_vkGetPhysicalDeviceProperties nextGetPhysicalDeviceProperties = nullptr;
PFN_vkGetPhysicalDeviceProperties2 nextGetPhysicalDeviceProperties2 = nullptr;
PFN_vkGetPhysicalDeviceFeatures2 nextGetPhysicalDeviceFeatures2 = nullptr;
PFN_vkGetPhysicalDeviceImageFormatProperties2 nextGetImageFormatProperties2 = nullptr;
PFN_vkEnumerateDeviceExtensionProperties nextEnumerateDeviceExtensionProperties = nullptr;
PFN_vkGetPhysicalDeviceQueueFamilyProperties nextGetPhysicalDeviceQueueFamilyProperties = nullptr;
PFN_vkGetPhysicalDeviceQueueFamilyProperties2 nextGetPhysicalDeviceQueueFamilyProperties2 = nullptr;
PFN_vkCreateHeadlessSurfaceEXT nextCreateHeadlessSurface = nullptr;
PFN_vkGetPhysicalDeviceSurfaceSupportKHR nextGetSurfaceSupport = nullptr;
PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR nextGetSurfaceCapabilities = nullptr;
PFN_vkGetPhysicalDeviceSurfaceFormatsKHR nextGetSurfaceFormats = nullptr;
PFN_vkCreateSwapchainKHR nextCreateSwapchain = nullptr;
PFN_vkGetSwapchainImagesKHR nextGetSwapchainImages = nullptr;

// What a failing call returns: BFR_FAIL_RESULT's result, or VK_ERROR_UNKNOWN.
VkResult failure() {
    const char* named = std::getenv("BFR_FAIL_RESULT");
    const std::string_view result = named != nullptr ? named : "";
    if (result == "VK_ERROR_OUT_OF_HOST_MEMORY") {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    if (result == "VK_ERROR_DEVICE_LOST") {
        return VK_ERROR_DEVICE_LOST;
    }
    if (result == "VK_ERROR_SURFACE_LOST_KHR") {
        return VK_ERROR_SURFACE_LOST_KHR;
    }
    if (result == "VK_ERROR_FORMAT_NOT_SUPPORTED") {
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }
    return VK_ERROR_UNKNOWN;
}

// What a failed call leaves in a handle it would have returned.
template <class Handle>
Handle junk() {
    constexpr std::uint64_t kJunk = 0xBAD0BAD0;
    return std::bit_cast<Handle>(kJunk);
}

// Whether BFR_FAIL names function.
bool failing(std::string_view function) {
    const char* named = std::getenv("BFR_FAIL");
    return named != nullptr && function == named;
}

bool separatePresent() {
    return std::getenv("BFR_ADAPTER_SEPARATE_PRESENT") != nullptr;
}

// Sets the version and vendor the variables name.
void overrideProperties(VkPhysicalDeviceProperties& properties) {
    if (const char* minor = std::getenv("BFR_ADAPTER_MINOR"); minor != nullptr) {
        properties.apiVersion = VK_MAKE_API_VERSION(0, 1, std::stoul(minor), 0);
    }
    if (const char* vendor = std::getenv("BFR_ADAPTER_VENDOR_ID"); vendor != nullptr) {
        properties.vendorID = static_cast<std::uint32_t>(std::stoul(vendor, nullptr, 0));
    }
}

template <class Function>
Function next(VkInstance instance, const char* name) {
    return std::bit_cast<Function>(nextGetInstanceProcAddr(instance, name));
}

VKAPI_ATTR VkResult VKAPI_CALL createInstance(const VkInstanceCreateInfo* info, const VkAllocationCallbacks* allocator,
                                              VkInstance* instance) {
    auto* link = linkInfo<VkLayerInstanceCreateInfo>(info->pNext, VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO);
    if (link == nullptr) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    nextGetInstanceProcAddr = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    link->u.pLayerInfo = link->u.pLayerInfo->pNext;

    const VkResult result = next<PFN_vkCreateInstance>(VK_NULL_HANDLE, "vkCreateInstance")(info, allocator, instance);
    if (result == VK_SUCCESS) {
        nextGetPhysicalDeviceProperties = next<PFN_vkGetPhysicalDeviceProperties>(*instance,
                                                                                  "vkGetPhysicalDeviceProperties");
        nextGetPhysicalDeviceProperties2 = next<PFN_vkGetPhysicalDeviceProperties2>(*instance,
                                                                                    "vkGetPhysicalDeviceProperties2");
        nextGetPhysicalDeviceFeatures2 = next<PFN_vkGetPhysicalDeviceFeatures2>(*instance,
                                                                                "vkGetPhysicalDeviceFeatures2");
        nextGetImageFormatProperties2 =
            next<PFN_vkGetPhysicalDeviceImageFormatProperties2>(*instance, "vkGetPhysicalDeviceImageFormatProperties2");
        nextEnumerateDeviceExtensionProperties =
            next<PFN_vkEnumerateDeviceExtensionProperties>(*instance, "vkEnumerateDeviceExtensionProperties");
        nextGetPhysicalDeviceQueueFamilyProperties =
            next<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(*instance, "vkGetPhysicalDeviceQueueFamilyProperties");
        nextGetPhysicalDeviceQueueFamilyProperties2 =
            next<PFN_vkGetPhysicalDeviceQueueFamilyProperties2>(*instance, "vkGetPhysicalDeviceQueueFamilyProperties2");
        nextCreateHeadlessSurface = next<PFN_vkCreateHeadlessSurfaceEXT>(*instance, "vkCreateHeadlessSurfaceEXT");
        nextGetSurfaceSupport = next<PFN_vkGetPhysicalDeviceSurfaceSupportKHR>(*instance,
                                                                               "vkGetPhysicalDeviceSurfaceSupportKHR");
        nextGetSurfaceCapabilities =
            next<PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR>(*instance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
        nextGetSurfaceFormats = next<PFN_vkGetPhysicalDeviceSurfaceFormatsKHR>(*instance,
                                                                               "vkGetPhysicalDeviceSurfaceFormatsKHR");
    }
    return result;
}

// The queue families the driver underneath has.
std::uint32_t realFamilyCount(VkPhysicalDevice device) {
    std::uint32_t count = 0;
    nextGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    return count;
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

    if (failing("vkCreateDevice")) {
        *device = junk<VkDevice>();
        return failure();
    }
    const std::span enabled{info->ppEnabledExtensionNames, info->enabledExtensionCount};
    std::vector<const char*> extensions(enabled.begin(), enabled.end());
    if (const char* added = std::getenv("BFR_ADAPTER_ADD"); added != nullptr) {
        std::erase_if(extensions, [added](const char* name) { return std::string_view{name} == added; });
    }
    const std::span requested{info->pQueueCreateInfos, info->queueCreateInfoCount};
    std::vector<VkDeviceQueueCreateInfo> queues(requested.begin(), requested.end());
    const std::uint32_t families = realFamilyCount(adapter);
    std::erase_if(queues,
                  [families](const VkDeviceQueueCreateInfo& queue) { return queue.queueFamilyIndex >= families; });
    VkDeviceCreateInfo passed = *info;
    passed.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    passed.ppEnabledExtensionNames = extensions.data();
    passed.queueCreateInfoCount = static_cast<std::uint32_t>(queues.size());
    passed.pQueueCreateInfos = queues.data();
    const auto nextCreateDevice = std::bit_cast<PFN_vkCreateDevice>(
        nextInstanceProcAddr(VK_NULL_HANDLE, "vkCreateDevice"));
    const VkResult result = nextCreateDevice(adapter, &passed, allocator, device);
    if (result == VK_SUCCESS) {
        nextCreateSwapchain = std::bit_cast<PFN_vkCreateSwapchainKHR>(
            nextGetDeviceProcAddr(*device, "vkCreateSwapchainKHR"));
        nextGetSwapchainImages = std::bit_cast<PFN_vkGetSwapchainImagesKHR>(
            nextGetDeviceProcAddr(*device, "vkGetSwapchainImagesKHR"));
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL getPhysicalDeviceProperties(VkPhysicalDevice device,
                                                       VkPhysicalDeviceProperties* properties) {
    nextGetPhysicalDeviceProperties(device, properties);
    overrideProperties(*properties);
}

// Sets the driver ID in the driver properties chained from properties.
void overrideDriver(VkPhysicalDeviceProperties2& properties) {
    const char* id = std::getenv("BFR_ADAPTER_DRIVER_ID");
    if (id == nullptr) {
        return;
    }
    for (auto* next = static_cast<VkBaseOutStructure*>(properties.pNext); next != nullptr; next = next->pNext) {
        if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES) {
            std::bit_cast<VkPhysicalDeviceDriverProperties*>(next)->driverID = static_cast<VkDriverId>(std::stoi(id));
        }
    }
}

VKAPI_ATTR void VKAPI_CALL getPhysicalDeviceProperties2(VkPhysicalDevice device,
                                                        VkPhysicalDeviceProperties2* properties) {
    nextGetPhysicalDeviceProperties2(device, properties);
    overrideProperties(properties->properties);
    overrideDriver(*properties);
}

// The features BFR_ADAPTER_NO_FEATURE may name.
template <class Features>
struct NamedFeature {
    std::string_view name;
    VkBool32 Features::* member;
};

using Vulkan12 = VkPhysicalDeviceVulkan12Features;
using Vulkan13 = VkPhysicalDeviceVulkan13Features;

constexpr std::array kVulkan12Features{
    NamedFeature<Vulkan12>{.name = "descriptorIndexing", .member = &Vulkan12::descriptorIndexing},
    NamedFeature<Vulkan12>{.name = "runtimeDescriptorArray", .member = &Vulkan12::runtimeDescriptorArray},
    NamedFeature<Vulkan12>{.name = "descriptorBindingPartiallyBound",
                           .member = &Vulkan12::descriptorBindingPartiallyBound},
    NamedFeature<Vulkan12>{.name = "descriptorBindingVariableDescriptorCount",
                           .member = &Vulkan12::descriptorBindingVariableDescriptorCount},
    NamedFeature<Vulkan12>{.name = "descriptorBindingUpdateUnusedWhilePending",
                           .member = &Vulkan12::descriptorBindingUpdateUnusedWhilePending},
    NamedFeature<Vulkan12>{.name = "descriptorBindingSampledImageUpdateAfterBind",
                           .member = &Vulkan12::descriptorBindingSampledImageUpdateAfterBind},
    NamedFeature<Vulkan12>{.name = "descriptorBindingStorageImageUpdateAfterBind",
                           .member = &Vulkan12::descriptorBindingStorageImageUpdateAfterBind},
    NamedFeature<Vulkan12>{.name = "descriptorBindingStorageBufferUpdateAfterBind",
                           .member = &Vulkan12::descriptorBindingStorageBufferUpdateAfterBind},
    NamedFeature<Vulkan12>{.name = "shaderSampledImageArrayNonUniformIndexing",
                           .member = &Vulkan12::shaderSampledImageArrayNonUniformIndexing},
    NamedFeature<Vulkan12>{.name = "shaderStorageBufferArrayNonUniformIndexing",
                           .member = &Vulkan12::shaderStorageBufferArrayNonUniformIndexing},
    NamedFeature<Vulkan12>{.name = "shaderStorageImageArrayNonUniformIndexing",
                           .member = &Vulkan12::shaderStorageImageArrayNonUniformIndexing},
    NamedFeature<Vulkan12>{.name = "timelineSemaphore", .member = &Vulkan12::timelineSemaphore},
    NamedFeature<Vulkan12>{.name = "bufferDeviceAddress", .member = &Vulkan12::bufferDeviceAddress},
    NamedFeature<Vulkan12>{.name = "shaderFloat16", .member = &Vulkan12::shaderFloat16},
};

constexpr std::array kVulkan13Features{
    NamedFeature<Vulkan13>{.name = "dynamicRendering", .member = &Vulkan13::dynamicRendering},
    NamedFeature<Vulkan13>{.name = "synchronization2", .member = &Vulkan13::synchronization2},
    NamedFeature<Vulkan13>{.name = "maintenance4", .member = &Vulkan13::maintenance4},
};

// Clears the feature BFR_ADAPTER_NO_FEATURE names in features, if it has it.
template <class Features, std::size_t Count>
void clearNamed(Features& features, const std::array<NamedFeature<Features>, Count>& named) {
    const char* hidden = std::getenv("BFR_ADAPTER_NO_FEATURE");
    for (const auto& feature : named) {
        if (hidden != nullptr && feature.name == hidden) {
            features.*feature.member = VK_FALSE;
        }
    }
}

VKAPI_ATTR void VKAPI_CALL getPhysicalDeviceFeatures2(VkPhysicalDevice device, VkPhysicalDeviceFeatures2* features) {
    nextGetPhysicalDeviceFeatures2(device, features);
    for (auto* next = static_cast<VkBaseOutStructure*>(features->pNext); next != nullptr; next = next->pNext) {
        if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES) {
            clearNamed(*std::bit_cast<Vulkan12*>(next), kVulkan12Features);
        }
        else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES) {
            clearNamed(*std::bit_cast<Vulkan13*>(next), kVulkan13Features);
        }
    }
}

VKAPI_ATTR VkResult VKAPI_CALL getImageFormatProperties2(VkPhysicalDevice device,
                                                         const VkPhysicalDeviceImageFormatInfo2* info,
                                                         VkImageFormatProperties2* properties) {
    if (failing("vkGetPhysicalDeviceImageFormatProperties2")) {
        properties->imageFormatProperties.sampleCounts = VK_SAMPLE_COUNT_FLAG_BITS_MAX_ENUM;
        return failure();
    }
    return nextGetImageFormatProperties2(device, info, properties);
}

// The driver's queue families as this layer reports them.
std::vector<VkQueueFamilyProperties> reportedFamilies(VkPhysicalDevice device) {
    std::uint32_t count = realFamilyCount(device);
    std::vector<VkQueueFamilyProperties> families(count);
    nextGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());
    if (separatePresent()) {
        families.push_back(
            VkQueueFamilyProperties{.queueFlags = 0,
                                    .queueCount = 1,
                                    .timestampValidBits = 0,
                                    .minImageTransferGranularity = {.width = 1, .height = 1, .depth = 1}});
    }
    VkQueueFlags hidden = 0;
    if (std::getenv("BFR_ADAPTER_NO_COMPUTE") != nullptr) {
        hidden |= VK_QUEUE_COMPUTE_BIT;
    }
    if (std::getenv("BFR_ADAPTER_NO_GRAPHICS") != nullptr) {
        hidden |= VK_QUEUE_GRAPHICS_BIT;
    }
    for (VkQueueFamilyProperties& family : families) {
        family.queueFlags &= ~hidden;
    }
    return families;
}

VKAPI_ATTR void VKAPI_CALL getPhysicalDeviceQueueFamilyProperties(VkPhysicalDevice device, std::uint32_t* count,
                                                                  VkQueueFamilyProperties* families) {
    const auto reported = reportedFamilies(device);
    if (families == nullptr) {
        *count = static_cast<std::uint32_t>(reported.size());
        return;
    }
    *count = std::min(*count, static_cast<std::uint32_t>(reported.size()));
    std::copy_n(reported.begin(), *count, families);
}

// Validation reads the families through this one.
VKAPI_ATTR void VKAPI_CALL getPhysicalDeviceQueueFamilyProperties2(VkPhysicalDevice device, std::uint32_t* count,
                                                                   VkQueueFamilyProperties2* families) {
    const auto reported = reportedFamilies(device);
    if (families == nullptr) {
        *count = static_cast<std::uint32_t>(reported.size());
        return;
    }
    // The driver fills what it chains from its own families; the layer then
    // sets the properties it reports.
    std::uint32_t real = std::min(*count, realFamilyCount(device));
    nextGetPhysicalDeviceQueueFamilyProperties2(device, &real, families);
    *count = std::min(*count, static_cast<std::uint32_t>(reported.size()));
    for (std::uint32_t family = 0; family < *count; ++family) {
        std::span{families, *count}[family].queueFamilyProperties = reported[family];
    }
}

VKAPI_ATTR VkResult VKAPI_CALL enumerateDeviceExtensionProperties(VkPhysicalDevice device, const char* layer,
                                                                  std::uint32_t* count,
                                                                  VkExtensionProperties* properties) {
    if (layer == nullptr && failing("vkEnumerateDeviceExtensionProperties")) {
        return failure();
    }
    const char* hidden = std::getenv("BFR_ADAPTER_HIDE");
    const char* added = std::getenv("BFR_ADAPTER_ADD");
    if (layer != nullptr || (hidden == nullptr && added == nullptr)) {
        return nextEnumerateDeviceExtensionProperties(device, layer, count, properties);
    }
    std::uint32_t available = 0;
    nextEnumerateDeviceExtensionProperties(device, nullptr, &available, nullptr);
    std::vector<VkExtensionProperties> extensions(available);
    nextEnumerateDeviceExtensionProperties(device, nullptr, &available, extensions.data());
    if (hidden != nullptr) {
        std::erase_if(extensions, [hidden](const VkExtensionProperties& extension) {
            return std::string_view{extension.extensionName} == hidden;
        });
    }
    if (added != nullptr) {
        // Any version will do; the RHI reads only the name.
        constexpr std::uint32_t kSpecVersion = 1;
        VkExtensionProperties extension{.extensionName = {}, .specVersion = kSpecVersion};
        std::string_view{added}.copy(extension.extensionName, VK_MAX_EXTENSION_NAME_SIZE - 1);
        extensions.push_back(extension);
    }
    const auto listed = static_cast<std::uint32_t>(extensions.size());
    if (properties == nullptr) {
        *count = listed;
        return VK_SUCCESS;
    }
    const std::uint32_t written = std::min(*count, listed);
    std::copy_n(extensions.begin(), written, properties);
    *count = written;
    return written < listed ? VK_INCOMPLETE : VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL createHeadlessSurface(VkInstance instance, const VkHeadlessSurfaceCreateInfoEXT* info,
                                                     const VkAllocationCallbacks* allocator, VkSurfaceKHR* surface) {
    if (failing("vkCreateHeadlessSurfaceEXT")) {
        *surface = junk<VkSurfaceKHR>();
        return failure();
    }
    return nextCreateHeadlessSurface(instance, info, allocator, surface);
}

VKAPI_ATTR VkResult VKAPI_CALL getSurfaceSupport(VkPhysicalDevice device, std::uint32_t family, VkSurfaceKHR surface,
                                                 VkBool32* supported) {
    if (failing("vkGetPhysicalDeviceSurfaceSupportKHR")) {
        *supported = VK_TRUE;
        return failure();
    }
    if (std::getenv("BFR_PRESENT_UNSUPPORTED") != nullptr) {
        *supported = VK_FALSE;
        return VK_SUCCESS;
    }
    if (separatePresent()) {
        *supported = family == realFamilyCount(device) ? VK_TRUE : VK_FALSE;
        return VK_SUCCESS;
    }
    return nextGetSurfaceSupport(device, family, surface, supported);
}

VKAPI_ATTR VkResult VKAPI_CALL getSurfaceCapabilities(VkPhysicalDevice device, VkSurfaceKHR surface,
                                                      VkSurfaceCapabilitiesKHR* capabilities) {
    if (failing("vkGetPhysicalDeviceSurfaceCapabilitiesKHR")) {
        return failure();
    }
    return nextGetSurfaceCapabilities(device, surface, capabilities);
}

VKAPI_ATTR VkResult VKAPI_CALL getSurfaceFormats(VkPhysicalDevice device, VkSurfaceKHR surface, std::uint32_t* count,
                                                 VkSurfaceFormatKHR* formats) {
    if (failing("vkGetPhysicalDeviceSurfaceFormatsKHR")) {
        return failure();
    }
    if (std::getenv("BFR_SURFACE_NO_FORMATS") != nullptr) {
        *count = 0;
        return VK_SUCCESS;
    }
    return nextGetSurfaceFormats(device, surface, count, formats);
}

VKAPI_ATTR VkResult VKAPI_CALL createSwapchain(VkDevice device, const VkSwapchainCreateInfoKHR* info,
                                               const VkAllocationCallbacks* allocator, VkSwapchainKHR* swapchain) {
    if (failing("vkCreateSwapchainKHR")) {
        *swapchain = junk<VkSwapchainKHR>();
        return failure();
    }
    return nextCreateSwapchain(device, info, allocator, swapchain);
}

VKAPI_ATTR VkResult VKAPI_CALL getSwapchainImages(VkDevice device, VkSwapchainKHR swapchain, std::uint32_t* count,
                                                  VkImage* images) {
    if (failing("vkGetSwapchainImagesKHR")) {
        return failure();
    }
    return nextGetSwapchainImages(device, swapchain, count, images);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL getDeviceProcAddr(VkDevice device, const char* name) {
    const std::string_view function{name};
    if (function == "vkGetDeviceProcAddr") {
        return std::bit_cast<PFN_vkVoidFunction>(&getDeviceProcAddr);
    }
    if (function == "vkCreateSwapchainKHR") {
        return std::bit_cast<PFN_vkVoidFunction>(&createSwapchain);
    }
    if (function == "vkGetSwapchainImagesKHR") {
        return std::bit_cast<PFN_vkVoidFunction>(&getSwapchainImages);
    }
    return nextGetDeviceProcAddr(device, name);
}

// The layer's own version of a physical device function, or null for one it
// passes through.
PFN_vkVoidFunction physicalDeviceFunction(std::string_view function) {
    if (function == "vkGetPhysicalDeviceProperties") {
        return std::bit_cast<PFN_vkVoidFunction>(&getPhysicalDeviceProperties);
    }
    if (function == "vkGetPhysicalDeviceProperties2" || function == "vkGetPhysicalDeviceProperties2KHR") {
        return std::bit_cast<PFN_vkVoidFunction>(&getPhysicalDeviceProperties2);
    }
    if (function == "vkGetPhysicalDeviceFeatures2" || function == "vkGetPhysicalDeviceFeatures2KHR") {
        return std::bit_cast<PFN_vkVoidFunction>(&getPhysicalDeviceFeatures2);
    }
    if (function == "vkGetPhysicalDeviceImageFormatProperties2" ||
        function == "vkGetPhysicalDeviceImageFormatProperties2KHR") {
        return std::bit_cast<PFN_vkVoidFunction>(&getImageFormatProperties2);
    }
    if (function == "vkGetPhysicalDeviceQueueFamilyProperties") {
        return std::bit_cast<PFN_vkVoidFunction>(&getPhysicalDeviceQueueFamilyProperties);
    }
    if (function == "vkGetPhysicalDeviceQueueFamilyProperties2" ||
        function == "vkGetPhysicalDeviceQueueFamilyProperties2KHR") {
        return std::bit_cast<PFN_vkVoidFunction>(&getPhysicalDeviceQueueFamilyProperties2);
    }
    if (function == "vkGetPhysicalDeviceSurfaceSupportKHR") {
        return std::bit_cast<PFN_vkVoidFunction>(&getSurfaceSupport);
    }
    if (function == "vkGetPhysicalDeviceSurfaceCapabilitiesKHR") {
        return std::bit_cast<PFN_vkVoidFunction>(&getSurfaceCapabilities);
    }
    if (function == "vkGetPhysicalDeviceSurfaceFormatsKHR") {
        return std::bit_cast<PFN_vkVoidFunction>(&getSurfaceFormats);
    }
    if (function == "vkEnumerateDeviceExtensionProperties") {
        return std::bit_cast<PFN_vkVoidFunction>(&enumerateDeviceExtensionProperties);
    }
    return nullptr;
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
    if (function == "vkCreateHeadlessSurfaceEXT") {
        return std::bit_cast<PFN_vkVoidFunction>(&createHeadlessSurface);
    }
    if (function == "vkCreateSwapchainKHR") {
        return std::bit_cast<PFN_vkVoidFunction>(&createSwapchain);
    }
    if (function == "vkGetSwapchainImagesKHR") {
        return std::bit_cast<PFN_vkVoidFunction>(&getSwapchainImages);
    }
    if (const PFN_vkVoidFunction own = physicalDeviceFunction(function); own != nullptr) {
        return own;
    }
    return nextGetInstanceProcAddr(instance, name);
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
