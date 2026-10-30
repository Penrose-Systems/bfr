#pragma once

#include "bfr/core/error.h"
#include "bfr/core/pool.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/command_list.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/swapchain.h"
#include "metal/metal_swapchain.h"
#include "metal/metal_util.h"

#include <CoreGraphics/CGColorSpace.h>
#include <CoreGraphics/CGGeometry.h>
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <objc/message.h>
#include <objc/runtime.h>
#include <QuartzCore/QuartzCore.hpp>
#include <TargetConditionals.h>

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace bfr::rhi::metal {

class MetalRhi;

/// Commands recorded on one frame's buffer: the swapchain rendering is open
/// on, and every swapchain rendered to, whose drawables the submit waits for
/// and signals.
struct MetalCommandList {
    std::uint32_t frame = 0;
    std::optional<SwapchainHandle> rendering;
    std::vector<SwapchainHandle> rendered;
};

class MetalDevice final : public Device {
public:
    /// A device on device, which rhi listed and whose largest 2D texture is
    /// maxTexture2d pixels. Fails with RhiDeviceFailed when Metal gives no
    /// queue, event, or the buffers a frame needs.
    [[nodiscard]] static core::Result<std::unique_ptr<MetalDevice>> create(MetalRhi& rhi, Ref<MTL::Device> device,
                                                                           std::uint32_t maxTexture2d);

    /// For create.
    MetalDevice(MetalRhi& rhi, Ref<MTL::Device> device, std::uint32_t maxTexture2d);
    ~MetalDevice() override;
    MetalDevice(const MetalDevice&) = delete;
    MetalDevice& operator=(const MetalDevice&) = delete;
    MetalDevice(MetalDevice&&) = delete;
    MetalDevice& operator=(MetalDevice&&) = delete;

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
    // One command buffer that may be on the GPU: its allocator, which is reset
    // before it records again, the encoder rendering is open on, and the event
    // value its last submit signals.
    struct Frame {
        Ref<MTL4::CommandAllocator> allocator;
        Ref<MTL4::CommandBuffer> buffer;
        Ref<MTL4::RenderCommandEncoder> encoder;
        std::uint64_t doneValue = 0;
        // Whether a submit is on the GPU, and whether a command list records.
        bool pending = false;
        bool recording = false;
    };

    // Makes the queue, the event and every frame's allocator and buffer.
    [[nodiscard]] core::Result<void> open();
    // Waits for the frame's last submit, if one is pending.
    [[nodiscard]] core::Result<void> waitFrame(Frame& frame);
    // Waits for everything the device has submitted.
    [[nodiscard]] core::Result<void> waitIdle();
    // Ends the encoder and the buffer of a frame whose list will not be submitted.
    static void discard(Frame& frame);
    // The frame at index, which is less than kFramesInFlight.
    [[nodiscard]] Frame& frameAt(std::uint32_t index);
    // Fails with RhiInvalidHandle for a swapchain or command list this device
    // no longer has.
    [[nodiscard]] core::Result<MetalSwapchain*> swapchainFor(SwapchainHandle swapchain);
    [[nodiscard]] core::Result<MetalCommandList*> commandListFor(CommandListHandle commandList);

    MetalRhi* rhi_;
    Ref<MTL::Device> device_;
    std::uint32_t maxTexture2d_;
    Ref<MTL4::CommandQueue> queue_;
    // Signaled by the queue after each submit, to the value submitted_ counts.
    Ref<MTL::SharedEvent> event_;
    std::uint64_t submitted_ = 0;
    // Room for a few windows' swapchains before the pool grows, and a limit no
    // tool should reach.
    static constexpr std::uint32_t kSwapchainCapacity = 4;
    static constexpr std::uint32_t kSwapchainLimit = 256;
    core::Pool<MetalSwapchain, Swapchain> swapchains_;
    // Two frames: one recording while the GPU runs the other.
    static constexpr std::uint32_t kFramesInFlight = 2;
    std::array<Frame, kFramesInFlight> frames_{};
    std::uint32_t nextFrame_ = 0;
    core::Pool<MetalCommandList, CommandList> commandLists_;
    // The queue, event and frames are used by one thread at a time; the
    // device's functions may be called from several threads at once.
    std::mutex lock_;
};

} // namespace bfr::rhi::metal
