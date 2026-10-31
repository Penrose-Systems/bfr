#include "vulkan/vulkan_instance.h"

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/text.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/instance.h"
#include "bfr/rhi/rhi_config.h"
#include "vulkan/instance_table.h"
#include "vulkan/queue_families.h"
#include "vulkan/vulkan_caps.h"
#include "vulkan/vulkan_device.h"
#include "vulkan/vulkan_list.h"
#include "vulkan/vulkan_loader.h"
#include "vulkan/vulkan_result.h"
#include "vulkan/vulkan_surface.h"

#include <volk.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vk_platform.h>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace bfr::rhi::vulkan {

namespace {

constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";

// The oldest Vulkan an adapter may report and still be listed.
constexpr std::uint32_t kFloorVersion = VK_API_VERSION_1_3;

constexpr std::uint32_t kInstanceVersion = VK_API_VERSION_1_3;

// "major.minor" of a packed Vulkan version, with ".patch" when it isn't zero.
std::string versionName(std::uint32_t version) {
    const std::string name = std::format("{}.{}", VK_API_VERSION_MAJOR(version), VK_API_VERSION_MINOR(version));
    const std::uint32_t patch = VK_API_VERSION_PATCH(version);
    return patch == 0 ? name : std::format("{}.{}", name, patch);
}

// The newest Vulkan the loader supports; a 1.0 loader lacks the query.
core::Result<std::uint32_t> loaderVersion() {
    std::uint32_t version = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion != nullptr) {
        if (const VkResult result = vkEnumerateInstanceVersion(&version); result != VK_SUCCESS) {
            return failure(core::ErrorCode::RhiLoaderFailed, {.call = "vkEnumerateInstanceVersion", .result = result});
        }
    }
    return version;
}

// The device extensions every listed adapter needs.
constexpr std::array kFloorExtensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};

// A portability driver, such as MoltenVK, lists it, and a device on that
// adapter must then enable it. It is provisional, so vulkan_core.h lacks its name.
constexpr const char* kPortabilitySubset = "VK_KHR_portability_subset";

// The device extensions adapter lists, or the call that failed.
std::expected<std::vector<VkExtensionProperties>, VulkanFailure> deviceExtensions(const InstanceTable& vk,
                                                                                  VkPhysicalDevice adapter) {
    return listAll<VkExtensionProperties>("vkEnumerateDeviceExtensionProperties",
                                          [&vk, adapter](std::uint32_t* count, VkExtensionProperties* extensions) {
                                              return vk.vkEnumerateDeviceExtensionProperties(adapter, nullptr, count,
                                                                                             extensions);
                                          });
}

std::vector<VkQueueFamilyProperties> queueFamilies(const InstanceTable& vk, VkPhysicalDevice adapter) {
    std::uint32_t count = 0;
    vk.vkGetPhysicalDeviceQueueFamilyProperties(adapter, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vk.vkGetPhysicalDeviceQueueFamilyProperties(adapter, &count, families.data());
    return families;
}

// The queues to open on each family: one on the graphics and present
// families, and the counts asked for on the compute and transfer families.
// Vulkan takes one create info per family, pointing at as many priorities.
struct QueueCreateInfos {
    std::vector<VkDeviceQueueCreateInfo> infos;
    std::vector<float> priorities;
};

QueueCreateInfos queueCreateInfos(const QueueFamilies& families, std::uint32_t present, const DeviceQueues& wanted) {
    QueueCreateInfos queues;
    const auto add = [&queues](std::uint32_t family, std::uint32_t count) {
        if (count == 0) {
            return;
        }
        if (auto found = std::ranges::find(queues.infos, family, &VkDeviceQueueCreateInfo::queueFamilyIndex);
            found != queues.infos.end()) {
            found->queueCount = std::max(found->queueCount, count);
            return;
        }
        VkDeviceQueueCreateInfo& queue = queues.infos.emplace_back();
        queue.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue.queueFamilyIndex = family;
        queue.queueCount = count;
    };
    add(families.graphics, 1);
    add(present, 1);
    add(families.compute, wanted.compute);
    add(families.transfer, wanted.copy);
    std::uint32_t most = 1;
    for (const VkDeviceQueueCreateInfo& queue : queues.infos) {
        most = std::max(most, queue.queueCount);
    }
    queues.priorities.assign(most, 1.0F);
    for (VkDeviceQueueCreateInfo& queue : queues.infos) {
        queue.pQueuePriorities = queues.priorities.data();
    }
    return queues;
}

// "0, 2" for the families queues open, for the log.
std::string familyList(std::span<const VkDeviceQueueCreateInfo> queues) {
    std::vector<std::string> families;
    families.reserve(queues.size());
    for (const VkDeviceQueueCreateInfo& queue : queues) {
        families.push_back(std::to_string(queue.queueFamilyIndex));
    }
    return core::text::join(families, ", ");
}

// Why the adapter is below the floor, empty when it meets it. extensions are
// the device extensions it lists.
std::string belowFloor(const InstanceTable& vk, VkPhysicalDevice adapter, const VkPhysicalDeviceProperties& properties,
                       std::span<const VkExtensionProperties> extensions) {
    if (properties.apiVersion < kFloorVersion) {
        return std::format("Vulkan {}, needs {}", versionName(properties.apiVersion), versionName(kFloorVersion));
    }
    const auto families = queueFamilies(vk, adapter);
    const auto noFamilyRuns = [&families](VkQueueFlags operation) {
        return std::ranges::none_of(families, [operation](const VkQueueFamilyProperties& family) {
            return (family.queueFlags & operation) != 0;
        });
    };
    if (noFamilyRuns(VK_QUEUE_GRAPHICS_BIT)) {
        return "no queue family runs graphics";
    }
    if (noFamilyRuns(VK_QUEUE_COMPUTE_BIT)) {
        return "no queue family runs compute";
    }
    std::vector<std::string> missing;
    for (const char* required : kFloorExtensions) {
        if (!lists(extensions, required)) {
            missing.emplace_back(required);
        }
    }
    std::ranges::move(missingFloorFeatures(vk, adapter), std::back_inserter(missing));
    return missing.empty() ? std::string{} : "lacks " + core::text::join(missing, ", ");
}

AdapterType adapterType(VkPhysicalDeviceType type) {
    switch (type) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            return AdapterType::Discrete;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
            return AdapterType::Integrated;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
            return AdapterType::Virtual;
        case VK_PHYSICAL_DEVICE_TYPE_CPU:
            return AdapterType::Cpu;
        default:
            return AdapterType::Other;
    }
}

std::string_view typeName(AdapterType type) {
    switch (type) {
        case AdapterType::Discrete:
            return "discrete GPU";
        case AdapterType::Integrated:
            return "integrated GPU";
        case AdapterType::Virtual:
            return "virtual GPU";
        case AdapterType::Cpu:
            return "CPU";
        case AdapterType::Other:
            break;
    }
    return "other";
}

// The adapter's driver, which only a Vulkan 1.2 adapter reports.
VkPhysicalDeviceDriverProperties driverProperties(const InstanceTable& vk, VkPhysicalDevice adapter) {
    VkPhysicalDeviceDriverProperties driver{};
    driver.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    VkPhysicalDeviceProperties2 chain{};
    chain.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    chain.pNext = &driver;
    vk.vkGetPhysicalDeviceProperties2(adapter, &chain);
    return driver;
}

// Loaders before this hand a surface to every driver, even one without its
// window system. NVIDIA's driver has no headless surfaces and crashes when
// asked whether it presents to one (580.178.04); later loaders answer no
// themselves.
constexpr std::uint32_t kLoaderAnswersForDrivers = VK_MAKE_API_VERSION(0, 1, 4, 319);

// "name, type, driver, Vulkan version" for the log. Driver properties are core
// from Vulkan 1.2, so an older adapter's description leaves the driver out.
std::string describe(const InstanceTable& vk, VkPhysicalDevice adapter, const VkPhysicalDeviceProperties& properties) {
    std::string driverName;
    if (properties.apiVersion >= VK_API_VERSION_1_2) {
        const VkPhysicalDeviceDriverProperties driver = driverProperties(vk, adapter);
        driverName = std::format(", {} {}", driver.driverName, driver.driverInfo);
    }
    return std::format("{}, {}{}, Vulkan {}", properties.deviceName, typeName(adapterType(properties.deviceType)),
                       driverName, versionName(properties.apiVersion));
}

// What the loader lists for layer, nullopt when it lacks it, or the call that
// failed.
std::expected<std::optional<VkLayerProperties>, VulkanFailure> findLayer(std::string_view layer) {
    return listAll<VkLayerProperties>("vkEnumerateInstanceLayerProperties", vkEnumerateInstanceLayerProperties)
        .transform([layer](const std::vector<VkLayerProperties>& layers) -> std::optional<VkLayerProperties> {
            const auto found = std::ranges::find_if(layers, [layer](const VkLayerProperties& properties) {
                return std::string_view{properties.layerName} == layer;
            });
            if (found == layers.end()) {
                return std::nullopt;
            }
            return *found;
        });
}

// Whether the loader, or layer when given, provides extension, or the call
// that failed.
std::expected<bool, VulkanFailure> hasExtension(const char* layer, std::string_view extension) {
    return listAll<VkExtensionProperties>("vkEnumerateInstanceExtensionProperties",
                                          [layer](std::uint32_t* count, VkExtensionProperties* extensions) {
                                              return vkEnumerateInstanceExtensionProperties(layer, count, extensions);
                                          })
        .transform(
            [extension](const std::vector<VkExtensionProperties>& extensions) { return lists(extensions, extension); });
}

// Validation needs the layer and VK_EXT_debug_utils, which the loader or the
// layer provides. Checked before the instance exists, so a missing piece is
// reported as such, not as a failed vkCreateInstance.
core::Result<void> checkValidation(const core::Logger& logger) {
    const auto found = findLayer(kValidationLayer);
    if (!found) {
        return failure(core::ErrorCode::RhiValidationUnavailable, found.error());
    }
    // Named, so clang-tidy sees the check before the access.
    const std::optional<VkLayerProperties>& layer = *found;
    if (!layer) {
        return core::failure(core::ErrorCode::RhiValidationUnavailable,
                             std::format("{} is not installed", kValidationLayer));
    }
    BFR_LOG_INFO(logger, "{} {}", kValidationLayer, versionName(layer->specVersion));
    for (const char* provider : std::to_array<const char*>({nullptr, kValidationLayer})) {
        const auto provided = hasExtension(provider, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        if (!provided) {
            return failure(core::ErrorCode::RhiValidationUnavailable, provided.error());
        }
        if (*provided) {
            return {};
        }
    }
    return core::failure(core::ErrorCode::RhiValidationUnavailable,
                         std::format("{} is not available", VK_EXT_DEBUG_UTILS_EXTENSION_NAME));
}

core::Severity severityOf(VkDebugUtilsMessageSeverityFlagBitsEXT severity) {
    switch (severity) {
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
            return core::Severity::Error;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
            return core::Severity::Warning;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
            return core::Severity::Info;
        default:
            return core::Severity::Trace;
    }
}

// Hands each message to the sink userData points at. Validation messages are
// the API's behaviour, not BFR's diagnostics, so BFR_LOGGING doesn't drop them.
VKAPI_ATTR VkBool32 VKAPI_CALL onMessage(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                         VkDebugUtilsMessageTypeFlagsEXT,
                                         const VkDebugUtilsMessengerCallbackDataEXT* data, void* userData) {
    const core::Severity level = severityOf(severity);
    const std::string_view message = data->pMessage != nullptr ? data->pMessage : "";
    const auto& sink = *static_cast<const VulkanInstance::MessageSink*>(userData);
    if (sink.reasons != nullptr && level >= core::Severity::Warning) {
        sink.reasons->emplace_back(message);
    }
    sink.logger.send(level, message);
    return VK_FALSE;
}

// The instance extensions the Instance enables when the loader has them, or the
// listing that failed. Portability enumeration shows portability drivers, such
// as MoltenVK. The surface extensions cover every window system this build
// supports, and each needs VK_KHR_surface.
core::Result<std::vector<const char*>> optionalExtensions() {
    const auto listed = listAll<VkExtensionProperties>("vkEnumerateInstanceExtensionProperties",
                                                       [](std::uint32_t* count, VkExtensionProperties* extensions) {
                                                           return vkEnumerateInstanceExtensionProperties(nullptr, count,
                                                                                                         extensions);
                                                       });
    if (!listed) {
        return failure(core::ErrorCode::RhiInstanceFailed, listed.error());
    }
    const auto available = [&listed](const char* extension) {
        return lists(*listed, extension);
    };
    std::vector<const char*> extensions;
    if (available(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    }
    if (available(VK_KHR_SURFACE_EXTENSION_NAME)) {
        extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
        std::ranges::copy_if(surfaceExtensions(), std::back_inserter(extensions), available);
        // Lets surfaces list color spaces other than sRGB, such as HDR10.
        if (available(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME)) {
            extensions.push_back(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
        }
    }
    return extensions;
}

// Warnings and errors of every type, and the quieter messages when the logger
// wants them.
VkDebugUtilsMessengerCreateInfoEXT messengerInfo(VulkanInstance::MessageSink* sink) {
    VkDebugUtilsMessengerCreateInfoEXT info{};
    info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    if (sink->logger.wants(core::Severity::Info)) {
        info.messageSeverity |= VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;
    }
    if (sink->logger.wants(core::Severity::Trace)) {
        info.messageSeverity |= VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT;
    }
    info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    info.pfnUserCallback = onMessage;
    info.pUserData = sink;
    return info;
}

} // namespace

core::Result<std::unique_ptr<VulkanInstance>> VulkanInstance::create(const InstanceDesc& desc,
                                                                     const RhiConfig& config) {
    if (auto opened = openLoader(); !opened) {
        return std::unexpected(std::move(opened.error()));
    }
    const auto loader = loaderVersion();
    if (!loader) {
        return std::unexpected(loader.error());
    }
    BFR_LOG_INFO(desc.logger, "Vulkan loader {}", versionName(*loader));

    if (config.validation) {
        if (auto available = checkValidation(desc.logger); !available) {
            return std::unexpected(std::move(available.error()));
        }
    }

    auto rhi = std::make_unique<VulkanInstance>(config, desc.logger);
    rhi->loaderVersion_ = *loader;
    if (auto created = rhi->createInstance(); !created) {
        return std::unexpected(std::move(created.error()));
    }
    if (auto listed = rhi->listAdapters(); !listed) {
        return std::unexpected(std::move(listed.error()));
    }
    return rhi;
}

VulkanInstance::VulkanInstance(const RhiConfig& config, core::Logger logger)
    : Instance(config), messages_{.logger = std::move(logger)} {}

VulkanInstance::~VulkanInstance() {
    // Without a VkInstance, there is no table to destroy it through.
    if (instance_ == VK_NULL_HANDLE) {
        return;
    }
    if (messenger_ != VK_NULL_HANDLE) {
        vk_.vkDestroyDebugUtilsMessengerEXT(instance_, messenger_, nullptr);
    }
    vk_.vkDestroyInstance(instance_, nullptr);
    BFR_LOG_DEBUG(logger(), "Vulkan instance destroyed");
}

std::span<const Adapter> VulkanInstance::adapters() const {
    return adapters_;
}

core::Result<std::unique_ptr<Device>> VulkanInstance::createDevice(std::size_t adapter,
                                                                   const core::window::NativeWindow& presentTo,
                                                                   const DeviceQueues& wanted) {
    if (auto listed = checkAdapter(adapter); !listed) {
        return std::unexpected(std::move(listed.error()));
    }
    const Adapter& listed = adapters_[adapter];
    VkPhysicalDevice physicalDevice = physicalDevices_[adapter];
    const auto properties = queueFamilies(vk_, physicalDevice);
    const auto presenting = presentingFamilies(adapter, properties.size(), presentTo);
    if (!presenting) {
        return std::unexpected(presenting.error());
    }
    const auto families = chooseQueueFamilies(properties, *presenting);
    if (!families) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, std::format("{} has no graphics queue", listed.name));
    }
    if (!families->present) {
        return core::failure(core::ErrorCode::RhiPresentUnsupported,
                             std::format("no queue family on {} presents to the window", listed.name));
    }
    if (wanted.compute != 0) {
        const std::uint32_t has = families->compute == families->graphics ? 0
                                                                          : properties[families->compute].queueCount;
        if (has < wanted.compute) {
            return core::failure(core::ErrorCode::RhiUnsupported,
                                 std::format("{} has {} compute queues; {} were asked for", listed.name, has,
                                             wanted.compute));
        }
    }
    if (wanted.copy != 0) {
        const bool dedicated = families->transfer != families->graphics && families->transfer != families->compute;
        const std::uint32_t has = dedicated ? properties[families->transfer].queueCount : 0;
        if (has < wanted.copy) {
            return core::failure(core::ErrorCode::RhiUnsupported,
                                 std::format("{} has {} copy queues; {} were asked for", listed.name, has,
                                             wanted.copy));
        }
    }
    const auto available = deviceExtensions(vk_, physicalDevice);
    if (!available) {
        return failure(core::ErrorCode::RhiDeviceFailed,
                       {.call = std::format("{} on {}", available.error().call, listed.name),
                        .result = available.error().result});
    }
    std::vector<const char*> extensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    if (lists(*available, kPortabilitySubset)) {
        extensions.push_back(kPortabilitySubset);
    }

    const auto queues = queueCreateInfos(*families, *families->present, wanted);
    const FloorFeatures features;
    VkDeviceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.pNext = &features.head();
    info.queueCreateInfoCount = static_cast<std::uint32_t>(queues.infos.size());
    info.pQueueCreateInfos = queues.infos.data();
    info.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    info.ppEnabledExtensionNames = extensions.data();

    VkDevice device = VK_NULL_HANDLE;
    const VkResult result = vk_.vkCreateDevice(physicalDevice, &info, nullptr, &device);
    if (result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiDeviceFailed,
                       {.call = std::format("vkCreateDevice on {}", listed.name), .result = result});
    }
    BFR_LOG_INFO(logger(), "device on {}, queue families {}, extensions: {}", listed.name, familyList(queues.infos),
                 core::text::join(extensions, ", "));
    auto made = VulkanDevice::create(*this, physicalDevice, device, *families, wanted, hasBudget_[adapter]);
    if (!made) {
        return std::unexpected(std::move(made.error()));
    }
    return std::unique_ptr<Device>{std::move(*made)};
}

core::Result<FormatCaps> VulkanInstance::formatCaps(std::size_t adapter, Format format) const {
    if (auto listed = checkAdapter(adapter); !listed) {
        return std::unexpected(std::move(listed.error()));
    }
    return readFormatCaps(vk_, physicalDevices_[adapter], format);
}

core::Result<SwapchainCaps> VulkanInstance::swapchainCaps(std::size_t adapter,
                                                          const core::window::NativeWindow& window) const {
    return askAboutSurface(adapter, window, readSurfaceCaps);
}

core::Result<std::vector<SwapchainFormat>> VulkanInstance::swapchainFormats(
    std::size_t adapter, const core::window::NativeWindow& window) const {
    return askAboutSurface(adapter, window, readSurfaceFormats);
}

core::Result<void> VulkanInstance::checkAdapter(std::size_t adapter) const {
    if (adapter >= adapters_.size()) {
        return core::failure(core::ErrorCode::RhiAdapterOutOfRange,
                             std::format("adapter {} of {}", adapter, adapters_.size()));
    }
    return {};
}

template <class Query>
auto VulkanInstance::askAboutSurface(std::size_t adapter, const core::window::NativeWindow& window, Query query) const
    -> std::invoke_result_t<Query, const InstanceTable&, VkPhysicalDevice, VkSurfaceKHR> {
    if (auto listed = checkAdapter(adapter); !listed) {
        return std::unexpected(std::move(listed.error()));
    }
    const auto surface = surfaceFor(window);
    if (!surface) {
        return std::unexpected(surface.error());
    }
    VkPhysicalDevice physicalDevice = physicalDevices_[adapter];
    if (presentQueryCrashes(physicalDevice, *surface)) {
        BFR_LOG_INFO(logger(),
                     "{} does not present to headless surfaces; not asked, since loader {} would crash its driver",
                     adapters_[adapter].name, versionName(loaderVersion_));
        return core::failure(core::ErrorCode::RhiPresentUnsupported,
                             std::format("{} can't present to this window", adapters_[adapter].name));
    }
    return query(vk_, physicalDevice, surface->handle());
}

core::Result<std::vector<VkBool32>> VulkanInstance::presentingFamilies(
    std::size_t adapter, std::size_t familyCount, const core::window::NativeWindow& presentTo) const {
    std::vector<VkBool32> presenting;
    if (std::holds_alternative<std::monostate>(presentTo)) {
        return presenting;
    }
    // The surface lives only while the families are asked: some window systems
    // allow one surface per window, and a swapchain makes its own.
    const auto surface = surfaceFor(presentTo);
    if (!surface) {
        return std::unexpected(surface.error());
    }
    VkPhysicalDevice physicalDevice = physicalDevices_[adapter];
    if (presentQueryCrashes(physicalDevice, *surface)) {
        BFR_LOG_INFO(logger(),
                     "{} does not present to headless surfaces; not asked, since loader {} would crash its driver",
                     adapters_[adapter].name, versionName(loaderVersion_));
    }
    presenting.resize(familyCount);
    for (std::uint32_t family = 0; family < presenting.size(); ++family) {
        const auto answer = presents(physicalDevice, family, *surface);
        if (!answer) {
            return std::unexpected(answer.error());
        }
        presenting[family] = *answer ? VK_TRUE : VK_FALSE;
    }
    return presenting;
}

core::Result<VulkanSurface> VulkanInstance::surfaceFor(const core::window::NativeWindow& window) const {
    return createSurface(instance_, vk_, extensions_, window);
}

core::Result<bool> VulkanInstance::presents(VkPhysicalDevice adapter, std::uint32_t family,
                                            const VulkanSurface& surface) const {
    if (presentQueryCrashes(adapter, surface)) {
        return false;
    }
    VkBool32 supported = VK_FALSE;
    if (const VkResult result = vk_.vkGetPhysicalDeviceSurfaceSupportKHR(adapter, family, surface.handle(), &supported);
        result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiWindowFailed,
                       {.call = "vkGetPhysicalDeviceSurfaceSupportKHR", .result = result});
    }
    return supported == VK_TRUE;
}

bool VulkanInstance::presentQueryCrashes(VkPhysicalDevice adapter, const VulkanSurface& surface) const {
    return loaderVersion_ < kLoaderAnswersForDrivers && surface.headless() &&
           driverProperties(vk_, adapter).driverID == VK_DRIVER_ID_NVIDIA_PROPRIETARY;
}

core::Result<void> VulkanInstance::createInstance() {
    VkApplicationInfo application{};
    application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application.pEngineName = "BFR";
    application.apiVersion = kInstanceVersion;

    VkInstanceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.pApplicationInfo = &application;

    std::vector<const char*> layers;
    auto optional = optionalExtensions();
    if (!optional) {
        return std::unexpected(optional.error());
    }
    std::vector<const char*> extensions = std::move(*optional);
    if (std::ranges::find(extensions, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) != extensions.end()) {
        info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }
    // Chained into the create info, the messenger reports what goes wrong
    // while the loader creates and destroys the instance.
    const VkDebugUtilsMessengerCreateInfoEXT messenger = messengerInfo(&messages_);
    if (config().validation) {
        layers.push_back(kValidationLayer);
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        info.pNext = &messenger;
    }
    info.enabledLayerCount = static_cast<std::uint32_t>(layers.size());
    info.ppEnabledLayerNames = layers.data();
    info.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    info.ppEnabledExtensionNames = extensions.data();

    // What the loader and layers report while they try, which a failure carries.
    std::vector<std::string> reasons;
    messages_.reasons = &reasons;
    VkInstance handle = VK_NULL_HANDLE;
    const VkResult result = vkCreateInstance(&info, nullptr, &handle);
    messages_.reasons = nullptr;
    if (result != VK_SUCCESS) {
        // Validation is the only layer the Instance enables, so a layer that is
        // listed yet fails to load is validation.
        const core::ErrorCode code = result == VK_ERROR_LAYER_NOT_PRESENT
                                         ? core::ErrorCode::RhiValidationUnavailable
                                         : codeFor(core::ErrorCode::RhiInstanceFailed, result);
        return core::failure(code,
                             core::error::withReasons(std::format("vkCreateInstance: {}", string_VkResult(result)),
                                                      reasons));
    }
    instance_ = handle;
    extensions_ = std::move(extensions);
    vk_ = loadInstanceTable(instance_);
    BFR_LOG_INFO(logger(), "Vulkan instance {}, validation {}", versionName(kInstanceVersion),
                 config().validation ? "on" : "off");

    // The chained messenger covers only creation and destruction; this one
    // reports everything in between.
    return config().validation ? createMessenger(messenger) : core::Result<void>{};
}

core::Result<void> VulkanInstance::createMessenger(const VkDebugUtilsMessengerCreateInfoEXT& info) {
    if (const VkResult result = vk_.vkCreateDebugUtilsMessengerEXT(instance_, &info, nullptr, &messenger_);
        result != VK_SUCCESS) {
        return failure(core::ErrorCode::RhiInstanceFailed,
                       {.call = "vkCreateDebugUtilsMessengerEXT", .result = result});
    }
    BFR_LOG_DEBUG(logger(), "Vulkan validation messenger created");
    return {};
}

core::Result<void> VulkanInstance::listAdapters() {
    const auto devices = listAll<VkPhysicalDevice>("vkEnumeratePhysicalDevices",
                                                   [this](std::uint32_t* count, VkPhysicalDevice* listed) {
                                                       return vk_.vkEnumeratePhysicalDevices(instance_, count, listed);
                                                   });
    if (!devices) {
        return failure(core::ErrorCode::RhiInstanceFailed, devices.error());
    }

    // Each adapter not listed, and why: below the floor, or a call that failed
    // while checking. One that fails is skipped, so the others still work.
    std::vector<std::string> rejected;
    for (VkPhysicalDevice device : *devices) {
        VkPhysicalDeviceProperties properties{};
        vk_.vkGetPhysicalDeviceProperties(device, &properties);
        const auto extensions = deviceExtensions(vk_, device);
        if (!extensions) {
            const std::string why = "skipped, " + extensions.error().message();
            BFR_LOG_WARN(logger(), "adapter {}: {}", describe(vk_, device, properties), why);
            rejected.push_back(std::format("{}: {}", properties.deviceName, why));
            continue;
        }
        const std::string reason = belowFloor(vk_, device, properties, *extensions);
        BFR_LOG_INFO(logger(), "adapter {}: {}", describe(vk_, device, properties),
                     reason.empty() ? std::string{"listed"} : "below the floor, " + reason);
        if (!reason.empty()) {
            rejected.push_back(std::format("{}: below the floor, {}", properties.deviceName, reason));
            continue;
        }
        adapters_.push_back({.name = properties.deviceName,
                             .type = adapterType(properties.deviceType),
                             .caps = readCaps(vk_, device, *extensions)});
        physicalDevices_.push_back(device);
        hasBudget_.push_back(lists(*extensions, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME));
    }
    if (adapters_.empty()) {
        return core::failure(core::ErrorCode::RhiNoAdapter, core::error::withReasons("no usable adapter", rejected));
    }
    return {};
}

} // namespace bfr::rhi::vulkan
