#include "vulkan_rhi.h"

#include "device.h"
#include "error.h"
#include "log.h"
#include "queue_families.h"
#include "rhi.h"
#include "rhi_config.h"
#include "text.h"
#include "vulkan_device.h"
#include "vulkan_list.h"
#include "vulkan_loader.h"
#include "vulkan_surface.h"
#include "window.h"

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
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace bfr::rhi {

core::Result<std::unique_ptr<Rhi>> Rhi::create(const RhiDesc& desc, const RhiConfig& config) {
    return vulkan::VulkanRhi::create(desc, config);
}

namespace vulkan {

namespace {

constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";

// The oldest Vulkan an adapter may report and still be listed.
constexpr std::uint32_t kFloorVersion = VK_API_VERSION_1_2;

constexpr std::uint32_t kInstanceVersion = VK_API_VERSION_1_3;

// "major.minor" of a packed Vulkan version, with ".patch" when it isn't zero.
std::string versionName(std::uint32_t version) {
    const std::string name = std::format("{}.{}", VK_API_VERSION_MAJOR(version), VK_API_VERSION_MINOR(version));
    const std::uint32_t patch = VK_API_VERSION_PATCH(version);
    return patch == 0 ? name : std::format("{}.{}", name, patch);
}

// The newest Vulkan the loader supports; a 1.0 loader lacks the query.
std::uint32_t loaderVersion() {
    std::uint32_t version = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion != nullptr) {
        static_cast<void>(vkEnumerateInstanceVersion(&version));
    }
    return version;
}

// A device extension every listed adapter needs, unless its Vulkan version
// made the extension core.
struct FloorExtension {
    const char* name;
    std::uint32_t coreFrom;
};

constexpr std::uint32_t kNeverCore = std::numeric_limits<std::uint32_t>::max();
constexpr std::array kFloorExtensions{
    FloorExtension{.name = VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME, .coreFrom = VK_API_VERSION_1_3},
    FloorExtension{.name = VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME, .coreFrom = VK_API_VERSION_1_3},
    FloorExtension{.name = VK_KHR_SWAPCHAIN_EXTENSION_NAME, .coreFrom = kNeverCore},
};

// A portability driver, such as MoltenVK, lists it, and a device on that
// adapter must then enable it. It is provisional, so vulkan_core.h lacks its name.
constexpr const char* kPortabilitySubset = "VK_KHR_portability_subset";

// The device extensions adapter lists, or the call that failed.
std::expected<std::vector<VkExtensionProperties>, std::string> deviceExtensions(VkPhysicalDevice adapter) {
    return listAll<VkExtensionProperties>("vkEnumerateDeviceExtensionProperties",
                                          [adapter](std::uint32_t* count, VkExtensionProperties* extensions) {
                                              return vkEnumerateDeviceExtensionProperties(adapter, nullptr, count,
                                                                                          extensions);
                                          });
}

bool lists(const std::vector<VkExtensionProperties>& extensions, std::string_view name) {
    return std::ranges::any_of(extensions, [name](const VkExtensionProperties& extension) {
        return std::string_view{extension.extensionName} == name;
    });
}

std::vector<VkQueueFamilyProperties> queueFamilies(VkPhysicalDevice adapter) {
    std::uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(adapter, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(adapter, &count, families.data());
    return families;
}

// One queue per family, present's included: roles on the same family share it,
// and Vulkan takes one create info per family.
std::vector<VkDeviceQueueCreateInfo> queueCreateInfos(const QueueFamilies& families, std::uint32_t present) {
    static constexpr float kQueuePriority = 1.0F;
    std::vector<VkDeviceQueueCreateInfo> queues;
    for (const std::uint32_t family : {families.graphics, families.compute, families.transfer, present}) {
        if (std::ranges::find(queues, family, &VkDeviceQueueCreateInfo::queueFamilyIndex) != queues.end()) {
            continue;
        }
        VkDeviceQueueCreateInfo& queue = queues.emplace_back();
        queue.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue.queueFamilyIndex = family;
        queue.queueCount = 1;
        queue.pQueuePriorities = &kQueuePriority;
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

// Why the adapter is below the floor, or empty when it meets it.
std::string belowFloor(VkPhysicalDevice adapter, const VkPhysicalDeviceProperties& properties) {
    if (properties.apiVersion < kFloorVersion) {
        return std::format("Vulkan {}, needs {}", versionName(properties.apiVersion), versionName(kFloorVersion));
    }
    const auto families = queueFamilies(adapter);
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
    const auto extensions = deviceExtensions(adapter);
    if (!extensions) {
        return extensions.error();
    }
    std::vector<const char*> missing;
    for (const FloorExtension& required : kFloorExtensions) {
        if (properties.apiVersion < required.coreFrom && !lists(*extensions, required.name)) {
            missing.push_back(required.name);
        }
    }
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
VkPhysicalDeviceDriverProperties driverProperties(VkPhysicalDevice adapter) {
    VkPhysicalDeviceDriverProperties driver{};
    driver.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    VkPhysicalDeviceProperties2 chain{};
    chain.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    chain.pNext = &driver;
    vkGetPhysicalDeviceProperties2(adapter, &chain);
    return driver;
}

// Loaders before this hand a surface to every driver, even one without its
// window system. NVIDIA's driver has no headless surfaces and crashes when
// asked whether it presents to one (580.178.04); later loaders answer no
// themselves.
constexpr std::uint32_t kLoaderAnswersForDrivers = VK_MAKE_API_VERSION(0, 1, 4, 319);

// "name, type, driver, Vulkan version" for the log. Driver properties are core
// from Vulkan 1.2, so an older adapter's description leaves the driver out.
std::string describe(VkPhysicalDevice adapter, const VkPhysicalDeviceProperties& properties) {
    std::string driverName;
    if (properties.apiVersion >= VK_API_VERSION_1_2) {
        const VkPhysicalDeviceDriverProperties driver = driverProperties(adapter);
        driverName = std::format(", {} {}", driver.driverName, driver.driverInfo);
    }
    return std::format("{}, {}{}, Vulkan {}", properties.deviceName, typeName(adapterType(properties.deviceType)),
                       driverName, versionName(properties.apiVersion));
}

// What the loader lists for layer, nullopt when it lacks it, or the call that
// failed.
std::expected<std::optional<VkLayerProperties>, std::string> findLayer(std::string_view layer) {
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
std::expected<bool, std::string> hasExtension(const char* layer, std::string_view extension) {
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
core::Result<void> checkValidation() {
    const auto found = findLayer(kValidationLayer);
    if (!found) {
        return core::failure(core::ErrorCode::RhiValidationUnavailable, found.error());
    }
    // Named, so clang-tidy sees the check before the access.
    const std::optional<VkLayerProperties>& layer = *found;
    if (!layer) {
        return core::failure(core::ErrorCode::RhiValidationUnavailable,
                             std::format("{} is not installed", kValidationLayer));
    }
    BFR_LOG_INFO("{} {}", kValidationLayer, versionName(layer->specVersion));
    for (const char* provider : std::to_array<const char*>({nullptr, kValidationLayer})) {
        const auto provided = hasExtension(provider, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        if (!provided) {
            return core::failure(core::ErrorCode::RhiValidationUnavailable, provided.error());
        }
        if (*provided) {
            return {};
        }
    }
    return core::failure(core::ErrorCode::RhiValidationUnavailable,
                         std::format("{} is not available", VK_EXT_DEBUG_UTILS_EXTENSION_NAME));
}

// Logs each message, then hands it to the sink userData points at.
VKAPI_ATTR VkBool32 VKAPI_CALL onMessage(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                         VkDebugUtilsMessageTypeFlagsEXT,
                                         const VkDebugUtilsMessengerCallbackDataEXT* data, void* userData) {
    const bool isError = severity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    const std::string_view message = data->pMessage != nullptr ? data->pMessage : "";
    if (isError) {
        BFR_LOG_ERROR("{}", message);
    }
    else {
        BFR_LOG_WARN("{}", message);
    }
    const auto& sink = *static_cast<const VulkanRhi::MessageSink*>(userData);
    if (sink.reasons != nullptr) {
        sink.reasons->emplace_back(message);
    }
    if (sink.callback) {
        sink.callback(isError, message);
    }
    return VK_FALSE;
}

// The instance extensions the Rhi enables when the loader has them; a loader
// that can't list its extensions goes without, and vkCreateInstance reports
// what is wrong with it. Portability enumeration shows portability drivers,
// such as MoltenVK. The surface extensions cover every window system this
// build supports, and each needs VK_KHR_surface.
std::vector<const char*> optionalExtensions() {
    const auto available = [](const char* extension) {
        return hasExtension(nullptr, extension).value_or(false);
    };
    std::vector<const char*> extensions;
    if (available(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    }
    if (available(VK_KHR_SURFACE_EXTENSION_NAME)) {
        extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
        std::ranges::copy_if(surfaceExtensions(), std::back_inserter(extensions), available);
    }
    return extensions;
}

// Warnings and errors of every type.
VkDebugUtilsMessengerCreateInfoEXT messengerInfo(VulkanRhi::MessageSink* sink) {
    VkDebugUtilsMessengerCreateInfoEXT info{};
    info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    info.pfnUserCallback = onMessage;
    info.pUserData = sink;
    return info;
}

} // namespace

core::Result<std::unique_ptr<VulkanRhi>> VulkanRhi::create(const RhiDesc& desc, const RhiConfig& config) {
    if (auto opened = openLoader(); !opened) {
        return std::unexpected(std::move(opened.error()));
    }
    const std::uint32_t loader = loaderVersion();
    BFR_LOG_INFO("Vulkan loader {}", versionName(loader));

    if (config.validation) {
        if (auto available = checkValidation(); !available) {
            return std::unexpected(std::move(available.error()));
        }
    }

    auto rhi = std::make_unique<VulkanRhi>(config, desc.onValidationMessage);
    rhi->loaderVersion_ = loader;
    if (auto created = rhi->createInstance(); !created) {
        return std::unexpected(std::move(created.error()));
    }
    if (auto listed = rhi->listAdapters(); !listed) {
        return std::unexpected(std::move(listed.error()));
    }
    return rhi;
}

VulkanRhi::VulkanRhi(const RhiConfig& config, ValidationCallback onValidationMessage)
    : Rhi(config), messages_{.callback = std::move(onValidationMessage)} {}

VulkanRhi::~VulkanRhi() {
    // Without a VkInstance, volk never loaded vkDestroyInstance.
    if (instance_ == VK_NULL_HANDLE) {
        return;
    }
    if (messenger_ != VK_NULL_HANDLE) {
        vkDestroyDebugUtilsMessengerEXT(instance_, messenger_, nullptr);
    }
    vkDestroyInstance(instance_, nullptr);
    BFR_LOG_DEBUG("Vulkan instance destroyed");
}

std::span<const Adapter> VulkanRhi::adapters() const {
    return adapters_;
}

core::Result<std::unique_ptr<Device>> VulkanRhi::createDevice(std::size_t adapter,
                                                              const core::window::NativeWindow& presentTo) {
    if (adapter >= adapters_.size()) {
        return core::failure(core::ErrorCode::RhiAdapterOutOfRange,
                             std::format("adapter {} of {}", adapter, adapters_.size()));
    }
    const Adapter& listed = adapters_[adapter];
    VkPhysicalDevice physicalDevice = physicalDevices_[adapter];
    const auto properties = queueFamilies(physicalDevice);
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
    const auto available = deviceExtensions(physicalDevice);
    if (!available) {
        return core::failure(core::ErrorCode::RhiDeviceFailed, std::format("{}: {}", listed.name, available.error()));
    }
    std::vector<const char*> extensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    if (lists(*available, kPortabilitySubset)) {
        extensions.push_back(kPortabilitySubset);
    }

    const auto queues = queueCreateInfos(*families, *families->present);
    VkDeviceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.queueCreateInfoCount = static_cast<std::uint32_t>(queues.size());
    info.pQueueCreateInfos = queues.data();
    info.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    info.ppEnabledExtensionNames = extensions.data();

    VkDevice device = VK_NULL_HANDLE;
    const VkResult result = vkCreateDevice(physicalDevice, &info, nullptr, &device);
    if (result != VK_SUCCESS) {
        return core::failure(core::ErrorCode::RhiDeviceFailed,
                             std::format("vkCreateDevice on {}: {}", listed.name, string_VkResult(result)));
    }
    BFR_LOG_INFO("device on {}, queue families {}, extensions: {}", listed.name, familyList(queues),
                 core::text::join(extensions, ", "));
    return std::make_unique<VulkanDevice>(*this, physicalDevice, device, *families->present);
}

core::Result<std::vector<VkBool32>> VulkanRhi::presentingFamilies(std::size_t adapter, std::size_t familyCount,
                                                                  const core::window::NativeWindow& presentTo) const {
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
        BFR_LOG_WARN("{} does not present to headless surfaces; not asked, since loader {} would crash its driver",
                     adapters_[adapter].name, versionName(loaderVersion_));
    }
    presenting.resize(familyCount);
    for (std::uint32_t family = 0; family < presenting.size(); ++family) {
        presenting[family] = presents(physicalDevice, family, *surface) ? VK_TRUE : VK_FALSE;
    }
    return presenting;
}

core::Result<VulkanSurface> VulkanRhi::surfaceFor(const core::window::NativeWindow& window) const {
    return createSurface(instance_, extensions_, window);
}

bool VulkanRhi::presents(VkPhysicalDevice adapter, std::uint32_t family, const VulkanSurface& surface) const {
    if (presentQueryCrashes(adapter, surface)) {
        return false;
    }
    VkBool32 supported = VK_FALSE;
    return vkGetPhysicalDeviceSurfaceSupportKHR(adapter, family, surface.handle(), &supported) == VK_SUCCESS &&
           supported == VK_TRUE;
}

bool VulkanRhi::presentQueryCrashes(VkPhysicalDevice adapter, const VulkanSurface& surface) const {
    return loaderVersion_ < kLoaderAnswersForDrivers && surface.headless() &&
           driverProperties(adapter).driverID == VK_DRIVER_ID_NVIDIA_PROPRIETARY;
}

core::Result<void> VulkanRhi::createInstance() {
    VkApplicationInfo application{};
    application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application.pEngineName = "BFR";
    application.apiVersion = kInstanceVersion;

    VkInstanceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.pApplicationInfo = &application;

    std::vector<const char*> layers;
    std::vector<const char*> extensions = optionalExtensions();
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
        // Validation is the only layer the Rhi enables, so a layer that is
        // listed yet fails to load is validation.
        const core::ErrorCode code = result == VK_ERROR_LAYER_NOT_PRESENT ? core::ErrorCode::RhiValidationUnavailable
                                                                          : core::ErrorCode::RhiInstanceFailed;
        return core::failure(code,
                             core::error::withReasons(std::format("vkCreateInstance: {}", string_VkResult(result)),
                                                      reasons));
    }
    instance_ = handle;
    extensions_ = std::move(extensions);
    volkLoadInstanceOnly(instance_);
    BFR_LOG_INFO("Vulkan instance {}, validation {}", versionName(kInstanceVersion),
                 config().validation ? "on" : "off");

    // The chained messenger covers only creation and destruction; this one
    // reports everything in between.
    if (config().validation) {
        if (const VkResult result = vkCreateDebugUtilsMessengerEXT(instance_, &messenger, nullptr, &messenger_);
            result != VK_SUCCESS) {
            return core::failure(core::ErrorCode::RhiInstanceFailed,
                                 std::format("vkCreateDebugUtilsMessengerEXT: {}", string_VkResult(result)));
        }
        BFR_LOG_DEBUG("Vulkan validation messenger created");
    }
    return {};
}

core::Result<void> VulkanRhi::listAdapters() {
    const auto devices = listAll<VkPhysicalDevice>("vkEnumeratePhysicalDevices",
                                                   [this](std::uint32_t* count, VkPhysicalDevice* listed) {
                                                       return vkEnumeratePhysicalDevices(instance_, count, listed);
                                                   });
    if (!devices) {
        return core::failure(core::ErrorCode::RhiInstanceFailed, devices.error());
    }

    std::vector<std::string> rejected;
    for (VkPhysicalDevice device : *devices) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(device, &properties);
        const std::string reason = belowFloor(device, properties);
        BFR_LOG_INFO("adapter {}: {}", describe(device, properties),
                     reason.empty() ? std::string{"listed"} : "below the floor, " + reason);
        if (!reason.empty()) {
            rejected.push_back(std::format("{}: {}", properties.deviceName, reason));
            continue;
        }
        adapters_.push_back({.name = properties.deviceName, .type = adapterType(properties.deviceType)});
        physicalDevices_.push_back(device);
    }
    if (adapters_.empty()) {
        return core::failure(core::ErrorCode::RhiNoAdapter,
                             core::error::withReasons("no adapter meets the floor", rejected));
    }
    return {};
}

} // namespace vulkan

} // namespace bfr::rhi
