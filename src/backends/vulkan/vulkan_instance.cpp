#include "vulkan/vulkan_instance.h"

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/text.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/instance.h"
#include "bfr/rhi/rhi_config.h"
#include "rhi/device_limits.h"
#include "rhi/format_check.h"
#include "vulkan/instance_table.h"
#include "vulkan/queue_families.h"
#include "vulkan/vulkan_caps.h"
#include "vulkan/vulkan_device.h"
#include "vulkan/vulkan_list.h"
#include "vulkan/vulkan_loader.h"
#include "vulkan/vulkan_pipeline.h"
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

// The largest major and minor VK_MAKE_API_VERSION packs: 7 bits and 10
// (vulkan_core.h:62-63).
constexpr std::uint32_t kMaxMajor = 0x7F;
constexpr std::uint32_t kMaxMinor = 0x3FF;

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

// VK_QUEUE_GLOBAL_PRIORITY_MEDIUM is the level queues have when none is set,
// as Normal is.
VkQueueGlobalPriorityKHR vulkanLevel(QueueLevel level) {
    switch (level) {
        case QueueLevel::Normal:
            return VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR;
        case QueueLevel::Low:
            return VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR;
        case QueueLevel::High:
            return VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR;
        case QueueLevel::Realtime:
            return VK_QUEUE_GLOBAL_PRIORITY_REALTIME_KHR;
    }
    return VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR;
}

// The queues to open: one create info for each kind asked for, on its family,
// pointing at its queues' ranks, and with the kind's level where the device
// has VK_KHR_global_priority. No two kinds share a family: countQueues gives
// a kind none on a family an earlier kind takes. The arrays are sized before
// they are filled, so the pointers into them stay.
struct QueueCreateInfos {
    std::vector<VkDeviceQueueCreateInfo> infos;
    std::vector<VkDeviceQueueGlobalPriorityCreateInfoKHR> levels;
    std::vector<float> ranks;
};

// Fails with RhiUnsupported when a kind's queues ask for different levels,
// which Vulkan sets per family, or for a level other than normal on a device
// with no VK_KHR_global_priority to set it.
core::Result<QueueCreateInfos> queueCreateInfos(const QueueFamilies& families, const DeviceDesc& wanted,
                                                bool hasLevels) {
    if (!hasLevels) {
        if (auto checked = checkNormalLevels(wanted,
                                             "the device has neither Vulkan 1.4 nor VK_KHR_global_priority to set it");
            !checked) {
            return std::unexpected(std::move(checked.error()));
        }
    }
    constexpr std::size_t kKinds = 3;
    QueueCreateInfos queues;
    queues.infos.reserve(kKinds);
    queues.levels.reserve(kKinds);
    queues.ranks.reserve(wanted.graphics.size() + wanted.compute.size() + wanted.copy.size());
    // A kind with no family has no queues: createDevice refuses asking for any.
    for (const auto& [family, kind] :
         {std::pair{families.graphics, wanted.graphics}, std::pair{families.compute, wanted.compute},
          std::pair{families.transfer, wanted.copy}}) {
        if (kind.empty() || !family) {
            continue;
        }
        const QueueLevel level = kind.front().level;
        if (!std::ranges::all_of(kind, [level](const QueueDesc& queue) { return queue.level == level; })) {
            return core::failure(core::ErrorCode::RhiUnsupported,
                                 "Vulkan sets one level for all of a kind's queues; family {} was asked for several",
                                 *family);
        }
        const std::size_t first = queues.ranks.size();
        std::ranges::transform(kind, std::back_inserter(queues.ranks), &QueueDesc::rank);
        VkDeviceQueueCreateInfo& queue = queues.infos.emplace_back();
        queue.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue.queueFamilyIndex = *family;
        queue.queueCount = static_cast<std::uint32_t>(kind.size());
        queue.pQueuePriorities = std::span{queues.ranks}.subspan(first).data();
        if (hasLevels) {
            VkDeviceQueueGlobalPriorityCreateInfoKHR& global = queues.levels.emplace_back();
            global.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR;
            global.globalPriority = vulkanLevel(level);
            queue.pNext = &global;
        }
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

// Why the adapter is below BFR's minimum at version, the one a device on it
// uses; empty when it meets it.
std::string belowMinimum(const InstanceTable& vk, VkPhysicalDevice adapter, std::uint32_t version,
                         std::span<const VkExtensionProperties> extensions) {
    if (version < kMinimumVersion) {
        return std::format("Vulkan {}, needs {}", versionName(version), versionName(kMinimumVersion));
    }
    const DeviceFeatures features(vk, adapter, version, extensions);
    std::vector<std::string> missing = missingRequirements(features);
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
    if (desc.apiMajor > kMaxMajor || desc.apiMinor > kMaxMinor) {
        return core::failure(core::ErrorCode::RhiInvalidCall,
                             "Vulkan names no version {}.{}: a major takes 7 bits and a minor 10", desc.apiMajor,
                             desc.apiMinor);
    }
    const std::uint32_t version = VK_MAKE_API_VERSION(0, desc.apiMajor, desc.apiMinor, 0);
    if (version < kMinimumVersion) {
        return core::failure(core::ErrorCode::RhiUnsupported, "Vulkan {} is below BFR's minimum, {}",
                             versionName(version), versionName(kMinimumVersion));
    }
    if (auto opened = openLoader(); !opened) {
        return std::unexpected(std::move(opened.error()));
    }
    const auto loader = loaderVersion();
    if (!loader) {
        return std::unexpected(loader.error());
    }
    BFR_LOG_INFO(desc.logger, "Vulkan loader {}", versionName(*loader));

    auto rhi = std::make_unique<VulkanInstance>(config, desc.logger);
    rhi->loaderVersion_ = *loader;
    rhi->version_ = version;
    if (auto created = rhi->createInstance(desc); !created) {
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
                                                                   const DeviceDesc& wanted) {
    if (auto listed = checkAdapter(adapter); !listed) {
        return std::unexpected(std::move(listed.error()));
    }
    if (auto checked = checkLimits(wanted); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    const Adapter& listed = adapters_[adapter];
    VkPhysicalDevice physicalDevice = physicalDevices_[adapter];
    const auto properties = queueFamilies(vk_, physicalDevice);
    const auto presenting = presentingFamilies(adapter, properties.size(), presentTo);
    if (!presenting) {
        return std::unexpected(presenting.error());
    }
    const auto families = chooseQueueFamilies(properties, *presenting);
    const QueueCounts has = countQueues(families, properties);
    if (has.graphics < wanted.graphics.size()) {
        return core::failure(core::ErrorCode::RhiUnsupported, "{} has {} graphics queues; {} were asked for",
                             listed.name, has.graphics, wanted.graphics.size());
    }
    if (has.compute < wanted.compute.size()) {
        return core::failure(core::ErrorCode::RhiUnsupported, "{} has {} compute queues; {} were asked for",
                             listed.name, has.compute, wanted.compute.size());
    }
    if (has.copy < wanted.copy.size()) {
        return core::failure(core::ErrorCode::RhiUnsupported, "{} has {} copy queues; {} were asked for", listed.name,
                             has.copy, wanted.copy.size());
    }
    // With a window, a queue the device opens must present to it.
    const auto opens = [](std::optional<std::uint32_t> family, std::span<const QueueDesc> kind) {
        return family && !kind.empty();
    };
    if (!presenting->empty() && !std::ranges::any_of(std::array{std::pair{families.graphics, wanted.graphics},
                                                                std::pair{families.compute, wanted.compute},
                                                                std::pair{families.transfer, wanted.copy}},
                                                     [&](const auto& kind) {
                                                         return opens(kind.first, kind.second) &&
                                                                (*presenting)[*kind.first] == VK_TRUE;
                                                     })) {
        return core::failure(core::ErrorCode::RhiPresentUnsupported,
                             "no queue the device opens on {} presents to the window", listed.name);
    }
    const std::vector<VkExtensionProperties>* available = &adapterExtensions_[adapter];
    // Presenting needs it; a device without it computes and renders offscreen.
    const bool swapchains = lists(*available, VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    std::vector<const char*> extensions;
    if (swapchains) {
        extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }
    // calibrateTimestamps needs it; the caps list no host clock without it.
    if (const char* calibration = calibrationExtension(*available); calibration != nullptr) {
        extensions.push_back(calibration);
    }

    // Every feature the adapter reports, read as its caps were, so the caps
    // say what the device has; but robust access, which changes what an
    // out-of-bounds access does and costs speed, stays off.
    DeviceFeatures features(vk_, physicalDevice, versions_[adapter], *available);
    features.core.features.robustBufferAccess = VK_FALSE;
    features.vulkan13.robustImageAccess = VK_FALSE;
    features.imageRobustness.robustImageAccess = VK_FALSE;
    features.mesh.primitiveFragmentShadingRateMeshShader = VK_FALSE;
    extensions.insert(extensions.end(), features.enabledExtensions.begin(), features.enabledExtensions.end());

    const auto queues = queueCreateInfos(families, wanted, hasQueueLevels(versions_[adapter], *available));
    if (!queues) {
        return std::unexpected(queues.error());
    }
    // VUID-VkDeviceCreateInfo-None-10778.
    if (queues->infos.empty() && features.maintenance9.maintenance9 != VK_TRUE) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "{} has no maintenance9, which Vulkan needs to open a device with no queues", listed.name);
    }
    VkDeviceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.pNext = &features.core;
    info.queueCreateInfoCount = static_cast<std::uint32_t>(queues->infos.size());
    info.pQueueCreateInfos = queues->infos.data();
    info.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    info.ppEnabledExtensionNames = extensions.data();

    VkDevice device = VK_NULL_HANDLE;
    const VkResult result = vk_.vkCreateDevice(physicalDevice, &info, nullptr, &device);
    if (result != VK_SUCCESS) {
        return core::failure(codeFor(core::ErrorCode::RhiDeviceFailed, result), "vkCreateDevice on {}: {}", listed.name,
                             string_VkResult(result));
    }
    BFR_LOG_INFO(logger(), "device on {}, queue families {}, extensions: {}", listed.name, familyList(queues->infos),
                 core::text::join(extensions, ", "));
    VkPhysicalDeviceProperties adapterProperties{};
    vk_.vkGetPhysicalDeviceProperties(physicalDevice, &adapterProperties);
    auto made = VulkanDevice::create(*this, physicalDevice, device, families, wanted,
                                     {.apiVersion = versions_[adapter],
                                      .formats13 = formats13Of(versions_[adapter], *available),
                                      .hasBudget = hasBudget_[adapter],
                                      .hasSwapchain = swapchains,
                                      .pushConstantsSize = listed.caps.limits.maxPushConstantsSize,
                                      .timestampPeriod = adapterProperties.limits.timestampPeriod,
                                      .timestamps = listed.caps.timestamps,
                                      .multiDrawIndirect = listed.caps.features.multiDrawIndirect,
                                      .drawIndirectCount = listed.caps.features.drawIndirectCount,
                                      .features = listed.caps.features,
                                      .resolve = listed.caps.resolve});
    if (!made) {
        return std::unexpected(std::move(made.error()));
    }
    return std::unique_ptr<Device>{std::move(*made)};
}

core::Result<FormatCaps> VulkanInstance::formatCaps(std::size_t adapter, Format format) const {
    if (auto listed = checkAdapter(adapter); !listed) {
        return std::unexpected(std::move(listed.error()));
    }
    if (auto checked = checkFormat(format); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    return readFormatCaps(vk_, physicalDevices_[adapter], versions_[adapter], adapterExtensions_[adapter], format);
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
        return core::failure(core::ErrorCode::RhiAdapterOutOfRange, "adapter {} of {}", adapter, adapters_.size());
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
        return core::failure(core::ErrorCode::RhiPresentUnsupported, "{} can't present to this window",
                             adapters_[adapter].name);
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

core::Result<void> VulkanInstance::createInstance(const InstanceDesc& desc) {
    // VkApplicationInfo takes null-terminated names; null is none.
    const std::string applicationName{desc.applicationName};
    const std::string engineName{desc.engineName};
    VkApplicationInfo application{};
    application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application.pApplicationName = applicationName.empty() ? nullptr : applicationName.c_str();
    application.applicationVersion = desc.applicationVersion;
    application.pEngineName = engineName.empty() ? nullptr : engineName.c_str();
    application.engineVersion = desc.engineVersion;
    application.apiVersion = version_;

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
        // Validation is the only layer the Instance enables, and
        // VK_EXT_debug_utils the only extension it enables without finding it
        // listed, so either one missing is validation unavailable.
        const bool validationMissing = result == VK_ERROR_LAYER_NOT_PRESENT ||
                                       (config().validation && result == VK_ERROR_EXTENSION_NOT_PRESENT);
        const core::ErrorCode code = validationMissing ? core::ErrorCode::RhiValidationUnavailable
                                                       : codeFor(core::ErrorCode::RhiInstanceFailed, result);
        return core::failure(code,
                             core::error::withReasons(std::format("vkCreateInstance: {}", string_VkResult(result)),
                                                      reasons));
    }
    instance_ = handle;
    extensions_ = std::move(extensions);
    if (config().validation) {
        // For the log only: the loader has already loaded the layer.
        if (const auto layer = findLayer(kValidationLayer); layer && *layer) {
            BFR_LOG_INFO(logger(), "{} {}", kValidationLayer, versionName((*layer)->specVersion));
        }
    }
    vk_ = loadInstanceTable(instance_);
    BFR_LOG_INFO(logger(), "Vulkan instance {}, validation {}", versionName(version_),
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

    // Each adapter not listed, and why: below BFR's minimum, or a call that
    // failed while checking. One that fails is skipped, so the others still
    // work. A device on an adapter uses the lower of the instance's version
    // and the adapter's (initialization.adoc:883-910).
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
        const std::uint32_t version = std::min(version_, properties.apiVersion);
        const std::string reason = belowMinimum(vk_, device, version, *extensions);
        BFR_LOG_INFO(logger(), "adapter {}: {}", describe(vk_, device, properties),
                     reason.empty() ? std::string{"listed"} : "below BFR's minimum, " + reason);
        if (!reason.empty()) {
            rejected.push_back(std::format("{}: below BFR's minimum, {}", properties.deviceName, reason));
            continue;
        }
        adapters_.push_back({.name = properties.deviceName,
                             .type = adapterType(properties.deviceType),
                             .caps = readCaps(vk_, device, version, *extensions)});
        physicalDevices_.push_back(device);
        versions_.push_back(version);
        hasBudget_.push_back(lists(*extensions, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME));
        adapterExtensions_.push_back(*extensions);
    }
    if (adapters_.empty()) {
        return core::failure(core::ErrorCode::RhiNoAdapter, core::error::withReasons("no usable adapter", rejected));
    }
    return {};
}

} // namespace bfr::rhi::vulkan
