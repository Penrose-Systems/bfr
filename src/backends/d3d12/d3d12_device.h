#pragma once

#include "bfr/core/error.h"
#include "bfr/core/pool.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/swapchain.h"
#include "d3d12/d3d12_swapchain.h"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <windows.h>
#include <wrl/client.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace bfr::rhi::d3d12 {

class D3D12Rhi;

/// A command pool: an allocator and a list for each list recording at once,
/// since D3D12 lets one list record from an allocator at a time; how many are
/// handed out since the last reset, the lists still recording, and the fence
/// value its last submit signals.
struct D3D12CommandPool {
    struct Entry {
        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> list;
    };
    std::uint32_t queue = 0;
    std::vector<Entry> entries;
    std::uint32_t used = 0;
    std::vector<CommandListHandle> recording;
    std::uint64_t lastSubmit = 0;
};

/// One of the device's queues: its ID3D12CommandQueue and the list type it
/// takes, the fence each submit signals the next value of with the event waits
/// use, the value last submitted, the array a submit is built in, and the
/// allocators of pools destroyed while the GPU may still run their lists, each
/// with the value that frees them. Used by one thread at a time under lock,
/// which present shares; submitted is read from any thread.
struct D3D12Queue {
    QueueType type = QueueType::Graphics;
    D3D12_COMMAND_LIST_TYPE listType = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    HANDLE event = nullptr;
    std::atomic<std::uint64_t> submitted{0};
    std::vector<ID3D12CommandList*> lists;
    std::vector<std::pair<std::vector<D3D12CommandPool::Entry>, std::uint64_t>> retired;
    std::atomic<std::uint32_t> retiredCount{0};
    std::mutex lock;
};

/// Commands recorded on one list: the pool it came from, the swapchain
/// rendering is open on, what becomes of its image when rendering ends, and
/// every swapchain rendered to, whose images the submit marks as sent.
struct D3D12CommandList {
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> list;
    CommandPoolHandle pool;
    std::optional<SwapchainHandle> rendering;
    StoreOp storeOp = StoreOp::Store;
    std::vector<SwapchainHandle> rendered;
};

class D3D12Device final : public Device {
public:
    /// A device on device, which rhi listed from adapter, with heapCount memory
    /// heaps, a graphics queue and the queues asked for. Fails with
    /// RhiDeviceFailed when D3D12 gives no queue or fence.
    [[nodiscard]] static core::Result<std::unique_ptr<D3D12Device>> create(
        D3D12Rhi& rhi, Microsoft::WRL::ComPtr<ID3D12Device> device, Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter,
        std::size_t heapCount, const DeviceQueues& queues);

    /// For create.
    D3D12Device(D3D12Rhi& rhi, Microsoft::WRL::ComPtr<ID3D12Device> device,
                Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter, std::size_t heapCount);
    ~D3D12Device() override;
    D3D12Device(const D3D12Device&) = delete;
    D3D12Device& operator=(const D3D12Device&) = delete;
    D3D12Device(D3D12Device&&) = delete;
    D3D12Device& operator=(D3D12Device&&) = delete;

    [[nodiscard]] core::Result<SwapchainHandle> createSwapchain(const core::window::NativeWindow& window,
                                                                const SwapchainRequest& request) override;
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
    [[nodiscard]] core::Result<void> open(const DeviceQueues& queues);
    [[nodiscard]] core::Result<void> waitFor(D3D12Queue& queue, std::uint64_t value);
    [[nodiscard]] core::Result<void> checkQueue(std::uint32_t queue) const;
    [[nodiscard]] core::Result<void> checkToken(Token token) const;
    void collectPools(D3D12Queue& queue);
    // Waits for everything the device has submitted.
    [[nodiscard]] core::Result<void> waitIdle();
    void discard(ID3D12GraphicsCommandList* list);
    [[nodiscard]] core::Result<D3D12Swapchain*> swapchainFor(SwapchainHandle swapchain);
    [[nodiscard]] core::Result<D3D12CommandPool*> commandPoolFor(CommandPoolHandle commandPool);
    [[nodiscard]] core::Result<D3D12CommandList*> commandListFor(CommandListHandle commandList);

    D3D12Rhi* rhi_;
    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter_;
    std::size_t heapCount_;
    std::vector<std::unique_ptr<D3D12Queue>> queues_;
    // Room for a few windows' swapchains before the pool grows, and a limit no
    // tool should reach.
    static constexpr std::uint32_t kSwapchainCapacity = 4;
    static constexpr std::uint32_t kSwapchainLimit = 256;
    core::Pool<D3D12Swapchain, Swapchain> swapchains_;
    static constexpr std::uint32_t kCommandPoolCapacity = 4;
    static constexpr std::uint32_t kCommandPoolLimit = 256;
    static constexpr std::uint32_t kCommandListCapacity = 8;
    static constexpr std::uint32_t kCommandListLimit = 4096;
    core::Pool<D3D12CommandPool, CommandPool> commandPools_;
    core::Pool<D3D12CommandList, CommandList> commandLists_;
};

} // namespace bfr::rhi::d3d12
