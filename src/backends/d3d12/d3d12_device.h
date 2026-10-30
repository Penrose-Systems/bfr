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

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace bfr::rhi::d3d12 {

class D3D12Rhi;

/// Commands recorded on one frame's list: the swapchain rendering is open on,
/// what becomes of its image when rendering ends, and every swapchain rendered
/// to, whose images the submit marks as sent.
struct D3D12CommandList {
    std::uint32_t frame = 0;
    std::optional<SwapchainHandle> rendering;
    StoreOp storeOp = StoreOp::Store;
    std::vector<SwapchainHandle> rendered;
};

class D3D12Device final : public Device {
public:
    /// A device on device, which rhi listed from adapter, with heapCount memory
    /// heaps. Fails with RhiDeviceFailed when D3D12 gives no queue, fence, or
    /// the lists a frame needs.
    [[nodiscard]] static core::Result<std::unique_ptr<D3D12Device>> create(
        D3D12Rhi& rhi, Microsoft::WRL::ComPtr<ID3D12Device> device, Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter,
        std::size_t heapCount);

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
    [[nodiscard]] core::Result<CommandListHandle> beginCommands() override;
    [[nodiscard]] core::Result<void> beginRendering(CommandListHandle commandList, SwapchainHandle swapchain,
                                                    const RenderingDesc& desc) override;
    [[nodiscard]] core::Result<void> endRendering(CommandListHandle commandList) override;
    [[nodiscard]] core::Result<void> submit(CommandListHandle commandList) override;
    [[nodiscard]] core::Result<void> destroyCommandList(CommandListHandle commandList) override;
    [[nodiscard]] core::Result<MemoryBudget> budget() const override;

private:
    // One command list that may be on the GPU: its allocator, which is reset
    // before it records again, and the fence value its last submit signals.
    struct Frame {
        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> list;
        std::uint64_t doneValue = 0;
        // Whether a submit is on the GPU, and whether a command list records.
        bool pending = false;
        bool recording = false;
    };

    // Makes the queue, the fence and every frame's allocator and list.
    [[nodiscard]] core::Result<void> open();
    // Waits until the fence reaches value.
    [[nodiscard]] core::Result<void> waitFor(std::uint64_t value);
    // Waits for the frame's last submit, if one is pending.
    [[nodiscard]] core::Result<void> waitFrame(Frame& frame);
    // Waits for everything the device has submitted.
    [[nodiscard]] core::Result<void> waitIdle();
    // Closes the list of a frame that will not be submitted.
    void discard(Frame& frame);
    // The frame at index, which is less than kFramesInFlight.
    [[nodiscard]] Frame& frameAt(std::uint32_t index);
    // Fails with RhiInvalidHandle for a swapchain or command list this device
    // no longer has.
    [[nodiscard]] core::Result<D3D12Swapchain*> swapchainFor(SwapchainHandle swapchain);
    [[nodiscard]] core::Result<D3D12CommandList*> commandListFor(CommandListHandle commandList);

    D3D12Rhi* rhi_;
    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter_;
    std::size_t heapCount_;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue_;
    // Signaled by the queue after each submit, to the value submitted_ counts.
    Microsoft::WRL::ComPtr<ID3D12Fence> fence_;
    HANDLE event_ = nullptr;
    std::uint64_t submitted_ = 0;
    // Room for a few windows' swapchains before the pool grows, and a limit no
    // tool should reach.
    static constexpr std::uint32_t kSwapchainCapacity = 4;
    static constexpr std::uint32_t kSwapchainLimit = 256;
    core::Pool<D3D12Swapchain, Swapchain> swapchains_;
    // Two frames: one recording while the GPU runs the other.
    static constexpr std::uint32_t kFramesInFlight = 2;
    std::array<Frame, kFramesInFlight> frames_{};
    std::uint32_t nextFrame_ = 0;
    core::Pool<D3D12CommandList, CommandList> commandLists_;
    // The queue, fence and frames are used by one thread at a time; the
    // device's functions may be called from several threads at once.
    std::mutex lock_;
};

} // namespace bfr::rhi::d3d12
