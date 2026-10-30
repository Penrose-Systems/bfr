#pragma once

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/window.h"
#include "bfr/rhi/swapchain.h"
#include "rhi/swapchain_build.h"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <windows.h>
#include <wrl/client.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace bfr::rhi::d3d12 {

/// Whether a swapchain can be made on window: a Win32 window or a headless
/// one. Fails with RhiSurfaceFailed as Device::createSwapchain describes.
[[nodiscard]] core::Result<void> checkWindow(const core::window::NativeWindow& window);

/// One DXGI swapchain, and what it was last built as; empty until built. It
/// presents to a Win32 window, or to nothing, as a composition swapchain, for a
/// headless one.
class D3D12Swapchain {
public:
    /// A swapchain on window, which checkWindow allowed, built as request says each
    /// time. It presents through queue, with tearing when DXGI allows it, and
    /// reports to logger, which outlives it.
    D3D12Swapchain(Microsoft::WRL::ComPtr<ID3D12Device> device, Microsoft::WRL::ComPtr<IDXGIFactory2> factory,
                   Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue, const core::window::NativeWindow& window,
                   const SwapchainRequest& request, bool tearing, const core::Logger& logger);
    ~D3D12Swapchain() = default;
    D3D12Swapchain(const D3D12Swapchain&) = delete;
    D3D12Swapchain& operator=(const D3D12Swapchain&) = delete;
    D3D12Swapchain(D3D12Swapchain&&) = delete;
    D3D12Swapchain& operator=(D3D12Swapchain&&) = delete;

    /// Whether the surface allows the request, or why not: RhiUnsupported,
    /// saying what it allows.
    [[nodiscard]] core::Result<void> check() const;

    /// Builds the swapchain's images at size in pixels, dropping any image
    /// acquired; a window with no area leaves it empty. The GPU must be done
    /// with the old images. Fails with RhiUnsupported when the surface no
    /// longer allows the request, and RhiSwapchainFailed when DXGI refuses.
    [[nodiscard]] core::Result<SwapchainImages> build(Extent size);

    /// Takes the swapchain's next image. Fails with RhiOutOfDate when the
    /// swapchain has no images and RhiInvalidCall when an image is already
    /// acquired.
    [[nodiscard]] core::Result<void> acquire();

    /// The acquired image, null when none is.
    [[nodiscard]] ID3D12Resource* image() const;

    /// The render target view of the acquired image; unset when none is.
    [[nodiscard]] D3D12_CPU_DESCRIPTOR_HANDLE view() const;

    /// Whether a submit already renders to the acquired image.
    [[nodiscard]] bool submitted() const { return submitted_; }

    /// Records that a submit renders to the acquired image.
    void setSubmitted() { submitted_ = true; }

    /// Shows the acquired image after the work submitted for it, which the
    /// queue runs first; no image is acquired afterwards. Fails with
    /// RhiInvalidCall when none is acquired, and RhiSwapchainFailed, or
    /// RhiDeviceLost, when DXGI refuses.
    [[nodiscard]] core::Result<void> present();

private:
    // Lets go of the images, and the one acquired.
    void release();
    // Makes the DXGI swapchain, or resizes the one it has, as settings say.
    [[nodiscard]] core::Result<void> makeChain(const SwapchainBuild& settings);
    // Gets the swapchain's images and makes a render target view of each;
    // answers what the swapchain turned out as.
    [[nodiscard]] core::Result<DXGI_SWAP_CHAIN_DESC1> makeImages();

    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    Microsoft::WRL::ComPtr<IDXGIFactory2> factory_;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue_;
    // Null for a headless window.
    HWND window_ = nullptr;
    SwapchainRequest request_;
    bool tearing_;
    const core::Logger* logger_;
    // Made by the first build; the images change under it afterwards.
    Microsoft::WRL::ComPtr<IDXGISwapChain3> chain_;
    // The images, and a render target view of each in a heap of their own.
    std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> images_;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> viewHeap_;
    D3D12_CPU_DESCRIPTOR_HANDLE firstView_{};
    UINT viewStride_ = 0;
    // How many images the swapchain was last built with; 0 while empty.
    std::uint32_t count_ = 0;
    // The index of the acquired image.
    std::optional<std::uint32_t> acquired_;
    bool submitted_ = false;
};

} // namespace bfr::rhi::d3d12
