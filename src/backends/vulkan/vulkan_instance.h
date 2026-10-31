#pragma once

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/instance.h"
#include "bfr/rhi/rhi_config.h"
#include "vulkan/instance_table.h"
#include "vulkan/vulkan_surface.h"

#include <volk.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace bfr::rhi::vulkan {

class VulkanInstance final : public Instance {
public:
    /// Where the messenger sends each message: the logger, and while
    /// vkCreateInstance runs, the reasons its failure carries.
    struct MessageSink {
        core::Logger logger;
        std::vector<std::string>* reasons = nullptr;
    };

    [[nodiscard]] static core::Result<std::unique_ptr<VulkanInstance>> create(const InstanceDesc& desc,
                                                                              const RhiConfig& config);

    /// Holds no VkInstance until create makes one.
    VulkanInstance(const RhiConfig& config, core::Logger logger);
    ~VulkanInstance() override;
    VulkanInstance(const VulkanInstance&) = delete;
    VulkanInstance& operator=(const VulkanInstance&) = delete;
    VulkanInstance(VulkanInstance&&) = delete;
    VulkanInstance& operator=(VulkanInstance&&) = delete;

    [[nodiscard]] std::span<const Adapter> adapters() const override;
    [[nodiscard]] core::Result<std::unique_ptr<Device>> createDevice(std::size_t adapter,
                                                                     const core::window::NativeWindow& presentTo,
                                                                     const DeviceQueues& queues) override;
    [[nodiscard]] core::Result<FormatCaps> formatCaps(std::size_t adapter, Format format) const override;
    [[nodiscard]] core::Result<SwapchainCaps> swapchainCaps(std::size_t adapter,
                                                            const core::window::NativeWindow& window) const override;
    [[nodiscard]] core::Result<std::vector<SwapchainFormat>> swapchainFormats(
        std::size_t adapter, const core::window::NativeWindow& window) const override;

    using Instance::config;

    /// Where the Instance and its devices send their messages.
    [[nodiscard]] const core::Logger& logger() const { return messages_.logger; }

    /// The instance's functions, which this Instance and its devices call through.
    [[nodiscard]] const InstanceTable& instanceTable() const { return vk_; }

    /// A surface on window, which goes before this Instance. Fails as
    /// Device::createSwapchain describes.
    [[nodiscard]] core::Result<VulkanSurface> surfaceFor(const core::window::NativeWindow& window) const;

    /// Whether family on adapter presents to surface. A driver the query would
    /// crash is not asked and counts as no. Fails as the query does.
    [[nodiscard]] core::Result<bool> presents(VkPhysicalDevice adapter, std::uint32_t family,
                                              const VulkanSurface& surface) const;

private:
    [[nodiscard]] core::Result<void> createInstance();
    // The messenger that reports validation's messages between the instance's
    // creation and its destruction.
    [[nodiscard]] core::Result<void> createMessenger(const VkDebugUtilsMessengerCreateInfoEXT& info);
    [[nodiscard]] core::Result<void> listAdapters();
    // Fails with RhiAdapterOutOfRange when adapter is not an index into adapters_.
    [[nodiscard]] core::Result<void> checkAdapter(std::size_t adapter) const;
    // What query, given adapters()[adapter] and a surface on window, answers.
    // Fails as the query does, as surfaceFor does, and with
    // RhiPresentUnsupported for a driver the query would crash.
    template <class Query>
    [[nodiscard]] auto askAboutSurface(std::size_t adapter, const core::window::NativeWindow& window, Query query) const
        -> std::invoke_result_t<Query, const InstanceTable&, VkPhysicalDevice, VkSurfaceKHR>;
    // Which of adapters()[adapter]'s familyCount queue families present to
    // presentTo, index-aligned; empty without a window.
    [[nodiscard]] core::Result<std::vector<VkBool32>> presentingFamilies(
        std::size_t adapter, std::size_t familyCount, const core::window::NativeWindow& presentTo) const;
    // Whether asking adapter about surface would crash its driver.
    [[nodiscard]] bool presentQueryCrashes(VkPhysicalDevice adapter, const VulkanSurface& surface) const;

    // The messenger reaches this through its user data, so it lives here, at
    // a fixed address, from before the VkInstance exists until after it goes.
    MessageSink messages_;
    VkInstance instance_ = VK_NULL_HANDLE;
    InstanceTable vk_;
    VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
    // What the instance enabled; each name has static storage.
    std::vector<const char*> extensions_;
    std::vector<Adapter> adapters_;
    std::vector<VkPhysicalDevice> physicalDevices_; // index-aligned with adapters_
    // Whether each adapter has VK_EXT_memory_budget; index-aligned with adapters_.
    std::vector<bool> hasBudget_;
    std::uint32_t loaderVersion_ = 0;
};

} // namespace bfr::rhi::vulkan
