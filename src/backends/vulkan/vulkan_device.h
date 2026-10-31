#pragma once

#include "bfr/core/error.h"
#include "bfr/core/pool.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/swapchain.h"
#include "core/fixed_vector.h"
#include "core/inline_vector.h"
#include "vulkan/queue_families.h"
#include "vulkan/vulkan_swapchain.h"

#include <volk.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace bfr::rhi::vulkan {

class VulkanInstance;

// Room for a few windows' swapchains and command pools before their pools
// grow, and limits no tool should reach. Command lists are made on the frame
// path, so their pool, and every array a frame fills, is made at its limit.
inline constexpr std::uint32_t kSwapchainCapacity = 4;
inline constexpr std::uint32_t kSwapchainLimit = 256;
inline constexpr std::uint32_t kCommandPoolCapacity = 4;
inline constexpr std::uint32_t kCommandPoolLimit = 256;
inline constexpr std::uint32_t kCommandListLimit = 4096;

/// A command pool and the buffers allocated from it: how many are handed out
/// since the last reset, the lists still recording from it, and the timeline
/// value its last submit signals. Destroying it frees its buffers.
class VulkanCommandPool {
public:
    VulkanCommandPool(VkDevice device, const VolkDeviceTable& table, VkCommandPool pool, std::uint32_t queue);
    ~VulkanCommandPool();
    VulkanCommandPool(const VulkanCommandPool&) = delete;
    VulkanCommandPool& operator=(const VulkanCommandPool&) = delete;
    VulkanCommandPool(VulkanCommandPool&&) = delete;
    VulkanCommandPool& operator=(VulkanCommandPool&&) = delete;

    /// Gives up the VkCommandPool, which the caller then owns.
    [[nodiscard]] VkCommandPool release();

    VkCommandPool pool;
    std::uint32_t queue;
    core::FixedVector<VkCommandBuffer> buffers{kCommandListLimit};
    std::uint32_t used = 0;
    core::FixedVector<CommandListHandle> recording{kCommandListLimit};
    std::uint64_t lastSubmit = 0;

private:
    VkDevice device_;
    const VolkDeviceTable* table_;
};

/// One of the device's queues: its VkQueue and family, the timeline each
/// submit signals the next value of, the value last submitted, the arrays a
/// submit is built in, and the pools destroyed while the GPU may still run
/// their lists, each with the value that frees it. Used by one thread at a
/// time under lock, which present shares; submitted is read from any thread.
struct VulkanQueue {
    QueueType type = QueueType::Graphics;
    std::uint32_t family = 0;
    VkQueue queue = VK_NULL_HANDLE;
    VkSemaphore timeline = VK_NULL_HANDLE;
    std::atomic<std::uint64_t> submitted{0};
    core::FixedVector<VkCommandBufferSubmitInfo> buffers;
    core::FixedVector<VkSemaphoreSubmitInfo> waits;
    core::FixedVector<VkSemaphoreSubmitInfo> signals;
    core::FixedVector<VulkanSwapchain*> chains;
    // The largest value a submit waits for on each queue, by queue index.
    std::vector<std::uint64_t> waitValues;
    std::vector<std::pair<VkCommandPool, std::uint64_t>> retired;
    std::atomic<std::uint32_t> retiredCount{0};
    std::mutex lock;
};

/// Commands recorded on one buffer: the pool it came from, the swapchain
/// rendering is open on, and every swapchain rendered to, whose images the
/// submit waits for and signals.
struct VulkanCommandList {
    VkCommandBuffer buffer = VK_NULL_HANDLE;
    CommandPoolHandle pool;
    std::optional<SwapchainHandle> rendering;
    core::InlineVector<SwapchainHandle, kSwapchainLimit> rendered;
};

class VulkanDevice final : public Device {
public:
    /// A device over device, which rhi made on adapter with one graphics queue
    /// on families.graphics, queues.compute on families.compute and queues.copy
    /// on families.transfer; rhi outlives it. hasBudget says whether adapter
    /// has VK_EXT_memory_budget. Fails with RhiDeviceFailed, destroying device,
    /// when the API refuses a queue's timeline.
    [[nodiscard]] static core::Result<std::unique_ptr<VulkanDevice>> create(VulkanInstance& rhi,
                                                                            VkPhysicalDevice adapter, VkDevice device,
                                                                            const QueueFamilies& families,
                                                                            const DeviceQueues& queues, bool hasBudget);

    /// For create.
    VulkanDevice(VulkanInstance& rhi, VkPhysicalDevice adapter, VkDevice device, const QueueFamilies& families,
                 bool hasBudget);
    ~VulkanDevice() override;
    VulkanDevice(const VulkanDevice&) = delete;
    VulkanDevice& operator=(const VulkanDevice&) = delete;
    VulkanDevice(VulkanDevice&&) = delete;
    VulkanDevice& operator=(VulkanDevice&&) = delete;

    [[nodiscard]] core::Result<SwapchainHandle> createSwapchain(const core::window::NativeWindow& window,
                                                                const SwapchainDesc& desc) override;
    [[nodiscard]] core::Result<SwapchainImages> resizeSwapchain(SwapchainHandle swapchain, std::uint32_t width,
                                                                std::uint32_t height) override;
    [[nodiscard]] core::Result<void> destroySwapchain(SwapchainHandle swapchain) override;
    [[nodiscard]] core::Result<void> acquire(SwapchainHandle swapchain) override;
    [[nodiscard]] core::Result<void> present(SwapchainHandle swapchain) override;
    [[nodiscard]] core::Result<CommandPoolHandle> createCommandPool(std::uint32_t queue) override;
    [[nodiscard]] core::Result<void> resetCommandPool(CommandPoolHandle commandPool) override;
    [[nodiscard]] core::Result<void> destroyCommandPool(CommandPoolHandle commandPool) override;
    [[nodiscard]] core::Result<CommandListHandle> beginCommands(CommandPoolHandle commandPool) override;
    [[nodiscard]] core::Result<void> beginRendering(CommandListHandle commandList, SwapchainHandle swapchain,
                                                    const RenderingDesc& desc) override;
    [[nodiscard]] core::Result<void> endRendering(CommandListHandle commandList) override;
    [[nodiscard]] core::Result<Token> submit(const SubmitDesc& desc) override;
    [[nodiscard]] core::Result<void> wait(Token token) override;
    [[nodiscard]] core::Result<bool> poll(Token token) override;
    [[nodiscard]] core::Result<void> destroyCommandList(CommandListHandle commandList) override;
    [[nodiscard]] core::Result<MemoryBudget> budget() const override;

private:
    [[nodiscard]] core::Result<void> openQueues(const QueueFamilies& families, const DeviceQueues& queues);
    [[nodiscard]] core::Result<std::uint64_t> completedValue(const VulkanQueue& queue) const;
    [[nodiscard]] core::Result<void> checkQueue(std::uint32_t queue) const;
    [[nodiscard]] core::Result<void> checkToken(Token token) const;
    void collectPools(VulkanQueue& queue);
    // Waits for everything the device has submitted.
    [[nodiscard]] core::Result<void> waitIdle();
    [[nodiscard]] core::Result<VulkanSwapchain*> swapchainFor(SwapchainHandle swapchain);
    [[nodiscard]] core::Result<VulkanCommandPool*> commandPoolFor(CommandPoolHandle commandPool);
    [[nodiscard]] core::Result<VulkanCommandList*> commandListFor(CommandListHandle commandList);

    VulkanInstance* instance_;
    VkPhysicalDevice adapter_;
    VkDevice device_;
    // Each device calls through its own table: volk's global pointers hold one
    // device's functions, and two adapters can have different drivers.
    VolkDeviceTable table_{};
    std::uint32_t graphicsFamily_;
    std::uint32_t presentFamily_;
    bool hasBudget_;
    std::vector<std::unique_ptr<VulkanQueue>> queues_;
    VkQueue presentQueue_ = VK_NULL_HANDLE;
    VulkanQueue* presentOn_ = nullptr;
    core::Pool<VulkanSwapchain, Swapchain> swapchains_;
    core::Pool<VulkanCommandPool, CommandPool> commandPools_;
    core::Pool<VulkanCommandList, CommandList> commandLists_;
};

} // namespace bfr::rhi::vulkan
