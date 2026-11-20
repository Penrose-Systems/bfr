#pragma once

#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "rhi/feature_request.h"
#include "vulkan/vulkan_caps.h"

#include <volk.h>

#include <array>
#include <span>
#include <vector>

namespace bfr::rhi::vulkan {

/// What a requestable feature is on Vulkan, on one chain: the extensions it
/// takes and the members it turns on, either none where the chain's version
/// has it with nothing to enable; or none at all, available false, where
/// BFR has no form of it at that version.
struct VulkanForm {
    bool available = true;
    std::array<const char*, 2> extensions{};
    std::array<VkBool32*, 2> members{};
};

/// Each requestable feature's Vulkan form on chain, in kRequestableFeatures'
/// order; listed is what the adapter lists, which decides between an
/// extension and the one it was promoted from.
using VulkanFormOf = VulkanForm (*)(DeviceFeatures& chain, std::span<const VkExtensionProperties> listed);

/// Whether chain, with extensions, has each requestable feature, in
/// kRequestableFeatures' order: its form available, its extensions in
/// extensions, its members on, and its prerequisites had too. Queried
/// with the extensions the adapter lists, what the adapter offers; enabled,
/// with the device's, what the device has.
[[nodiscard]] std::array<bool, kRequestableFeatures.size()> requestableIn(
    DeviceFeatures& chain, std::span<const VkExtensionProperties> listed,
    std::span<const VkExtensionProperties> extensions);

/// What a device enables for request, made with desc, on an adapter that has
/// supported and lists listed: BFR's own needs (kRequired12, dynamic
/// rendering, synchronization2; missingRequirements), what the Vulkan spec
/// makes it enable (VK_KHR_portability_subset where listed,
/// VUID-VkDeviceCreateInfo-pProperties-04451, with the members the adapter
/// reports other than those requestable), queue levels and devices with no
/// queues as the adapter has them (global priority, maintenance9), each
/// requested feature's form, and the uniform buffer and acceleration
/// structure tables' update-after-bind where desc asks for slots in them and
/// the adapter has it. checkFeatures has passed request.
class EnabledFeatures {
public:
    EnabledFeatures(const DeviceFeatures& supported, std::span<const VkExtensionProperties> listed,
                    const FeatureRequest& request, const DeviceDesc& desc);
    ~EnabledFeatures() = default;
    EnabledFeatures(const EnabledFeatures&) = delete;
    EnabledFeatures& operator=(const EnabledFeatures&) = delete;
    EnabledFeatures(EnabledFeatures&&) = delete;
    EnabledFeatures& operator=(EnabledFeatures&&) = delete;

    /// The extensions' names, as a create info takes them, and as the
    /// listing the caps' readers take.
    std::vector<const char*> names;
    std::vector<VkExtensionProperties> extensions;
    /// The features, core at its head.
    DeviceFeatures chain;
};

} // namespace bfr::rhi::vulkan
