#pragma once

#include "bfr/core/error.h"
#include "bfr/core/pool.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/swapchain.h"
#include "vulkan/queue_families.h"
#include "vulkan/vulkan_swapchain.h"

#include <volk.h>

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace bfr::rhi::vulkan {

class VulkanRhi;

/// Commands recorded on one frame's buffer: the swapchain rendering is open
/// on, and every swapchain rendered to, whose images the submit waits for and
/// signals.
struct VulkanCommandList {
    VkCommandBuffer buffer = VK_NULL_HANDLE;
    std::uint32_t frame = 0;
    std::optional<SwapchainHandle> rendering;
    std::vector<SwapchainHandle> rendered;
};

class VulkanDevice final : public Device {
public:
    /// A device over device, which rhi made on adapter with a queue in each of
    /// families; rhi outlives it. hasBudget says whether adapter has
    /// VK_EXT_memory_budget. Fails with RhiDeviceFailed, destroying device,
    /// when the API refuses what a frame needs.
    [[nodiscard]] static core::Result<std::unique_ptr<VulkanDevice>> create(VulkanRhi& rhi, VkPhysicalDevice adapter,
                                                                            VkDevice device,
                                                                            const QueueFamilies& families,
                                                                            bool hasBudget);

    /// For create.
    VulkanDevice(VulkanRhi& rhi, VkPhysicalDevice adapter, VkDevice device, const QueueFamilies& families,
                 bool hasBudget);
    ~VulkanDevice() override;
    VulkanDevice(const VulkanDevice&) = delete;
    VulkanDevice& operator=(const VulkanDevice&) = delete;
    VulkanDevice(VulkanDevice&&) = delete;
    VulkanDevice& operator=(VulkanDevice&&) = delete;

    [[nodiscard]] core::Result<SwapchainHandle> createSwapchain(const core::window::NativeWindow& window,
                                                                const SwapchainRequest& request) override;
    [[nodiscard]] core::Result<SwapchainImages> resizeSwapchain(SwapchainHandle swapchain, std::uint32_t width,
                                                                std::uint32_t height) override;
    [[nodiscard]] core::Result<void> destroySwapchain(SwapchainHandle swapchain) override;
    [[nodiscard]] core::Result<void> acquire(SwapchainHandle swapchain) override;
    [[nodiscard]] core::Result<void> present(SwapchainHandle swapchain) override;
    [[nodiscard]] core::Result<CommandListHandle> beginCommands() override;
    [[nodiscard]] core::Result<void> beginRendering(CommandListHandle commandList, SwapchainHandle swapchain,
                                                    const RenderingDesc& desc) override;
    [[nodiscard]] core::Result<void> endRendering(CommandListHandle commandList) override;
    [[nodiscard]] core::Result<void> submit(CommandListHandle commandList) override;
    [[nodiscard]] core::Result<void> destroyCommandList(CommandListHandle commandList) override;
    [[nodiscard]] core::Result<MemoryBudget> budget() const override;

private:
    // One command buffer that may be on the GPU: its pool, which is reset
    // before it records again, and the fence its submit signals.
    struct Frame {
        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandBuffer buffer = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        // Whether a submit is on the GPU, and whether a command list records.
        bool pending = false;
        bool recording = false;
    };

    // Makes every frame's pool, buffer and fence.
    [[nodiscard]] core::Result<void> openFrames();
    // Waits for the frame's last submit, if one is pending.
    [[nodiscard]] core::Result<void> waitFrame(Frame& frame);
    // Waits for everything the device has submitted.
    [[nodiscard]] core::Result<void> waitIdle();
    // Fails with RhiInvalidHandle for a swapchain or command list this device
    // no longer has.
    [[nodiscard]] core::Result<VulkanSwapchain*> swapchainFor(SwapchainHandle swapchain);
    [[nodiscard]] core::Result<VulkanCommandList*> commandListFor(CommandListHandle commandList);

    VulkanRhi* rhi_;
    VkPhysicalDevice adapter_;
    VkDevice device_;
    // Each device calls through its own table: volk's global pointers hold one
    // device's functions, and two adapters can have different drivers.
    VolkDeviceTable table_{};
    std::uint32_t graphicsFamily_;
    std::uint32_t presentFamily_;
    bool hasBudget_;
    VkQueue graphicsQueue_ = VK_NULL_HANDLE;
    VkQueue presentQueue_ = VK_NULL_HANDLE;
    // Room for a few windows' swapchains before the pool grows, and a limit no
    // tool should reach.
    static constexpr std::uint32_t kSwapchainCapacity = 4;
    static constexpr std::uint32_t kSwapchainLimit = 256;
    core::Pool<VulkanSwapchain, Swapchain> swapchains_;
    // Two frames: one recording while the GPU runs the other.
    static constexpr std::uint32_t kFramesInFlight = 2;
    std::array<Frame, kFramesInFlight> frames_{};
    std::uint32_t nextFrame_ = 0;
    core::Pool<VulkanCommandList, CommandList> commandLists_;
    // The queues and frames are used by one thread at a time: Vulkan has the
    // caller synchronize a queue, and the device's functions may be called
    // from several threads at once.
    std::mutex queues_;
};

} // namespace bfr::rhi::vulkan
