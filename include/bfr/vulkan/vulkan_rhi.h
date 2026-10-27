#pragma once

#include "caps.h"
#include "device.h"
#include "error.h"
#include "format.h"
#include "log.h"
#include "rhi.h"
#include "rhi_config.h"
#include "vulkan_surface.h"
#include "window.h"

#include <volk.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace bfr::rhi::vulkan {

class VulkanRhi final : public Rhi {
public:
    /// Where the messenger sends each message: the logger, and while
    /// vkCreateInstance runs, the reasons its failure carries.
    struct MessageSink {
        core::Logger logger;
        std::vector<std::string>* reasons = nullptr;
    };

    [[nodiscard]] static core::Result<std::unique_ptr<VulkanRhi>> create(const RhiDesc& desc, const RhiConfig& config);

    /// Holds no VkInstance until create makes one.
    VulkanRhi(const RhiConfig& config, core::Logger logger);
    ~VulkanRhi() override;
    VulkanRhi(const VulkanRhi&) = delete;
    VulkanRhi& operator=(const VulkanRhi&) = delete;
    VulkanRhi(VulkanRhi&&) = delete;
    VulkanRhi& operator=(VulkanRhi&&) = delete;

    [[nodiscard]] std::span<const Adapter> adapters() const override;
    [[nodiscard]] core::Result<std::unique_ptr<Device>> createDevice(
        std::size_t adapter, const core::window::NativeWindow& presentTo) override;
    [[nodiscard]] core::Result<FormatCaps> formatCaps(std::size_t adapter, Format format) const override;
    [[nodiscard]] core::Result<SurfaceCaps> surfaceCaps(std::size_t adapter,
                                                        const core::window::NativeWindow& window) const override;
    [[nodiscard]] core::Result<std::vector<SurfaceFormat>> surfaceFormats(
        std::size_t adapter, const core::window::NativeWindow& window) const override;
    [[nodiscard]] core::Result<void*> native(std::size_t adapter) const override;

    using Rhi::config;

    /// Where the Rhi and its devices send their messages.
    [[nodiscard]] const core::Logger& logger() const { return messages_.logger; }

    /// A surface on window, which goes before this Rhi. Fails as
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
        -> std::invoke_result_t<Query, VkPhysicalDevice, VkSurfaceKHR>;
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
