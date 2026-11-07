#include "d3d12/d3d12_swapchain.h"

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/swapchain.h"
#include "d3d12/d3d12_formats.h"
#include "d3d12/d3d12_result.h"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <dxgiformat.h>
#include <windows.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <expected>
#include <format>
#include <utility>
#include <variant>

namespace bfr::rhi::d3d12 {

namespace {

using Microsoft::WRL::ComPtr;

// Each window system a swapchain can be made on, or why not.
struct Checker {
    core::Result<void> operator()(std::monostate /*none*/) const {
        return core::failure(core::ErrorCode::RhiWindowFailed, "there is no window");
    }

    core::Result<void> operator()(const core::window::Win32& window) const {
        if (window.window == nullptr) {
            return core::failure(core::ErrorCode::RhiWindowFailed, "the Win32 window has no handle");
        }
        return {};
    }

    core::Result<void> operator()(const core::window::Headless& /*window*/) const { return {}; }

    // A window system this build has no surface for.
    template <class Handles>
    core::Result<void> operator()(const Handles& /*window*/) const {
        return core::failure(core::ErrorCode::RhiWindowFailed, "this build has no surface for the window's system");
    }
};

// The window handle of window, null for a headless one.
HWND handleOf(const core::window::NativeWindow& window) {
    if (const auto* win32 = std::get_if<core::window::Win32>(&window)) {
        return static_cast<HWND>(win32->window);
    }
    return nullptr;
}

// How a present is made, as the present mode says: the sync interval and the
// flags. Godot's rendering_device_driver_d3d12.cpp (swap_chain_resize) sets
// them the same way, mailbox as DXGI_PRESENT_RESTART.
struct PresentSetup {
    UINT syncInterval = 1;
    UINT presentFlags = 0;
    UINT creationFlags = 0;
};

// Fails with RhiUnsupported for a mode DXGI has no setting for.
core::Result<PresentSetup> presentSetupFor(PresentMode mode) {
    switch (mode) {
        case PresentMode::Fifo:
            return PresentSetup{};
        case PresentMode::Immediate:
            return PresentSetup{.syncInterval = 0,
                                .presentFlags = DXGI_PRESENT_ALLOW_TEARING,
                                .creationFlags = DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING};
        case PresentMode::Mailbox:
            return PresentSetup{.syncInterval = 1, .presentFlags = DXGI_PRESENT_RESTART};
        case PresentMode::FifoRelaxed:
            break;
    }
    return core::failure(core::ErrorCode::RhiUnsupported, "DXGI has no setting for {} presenting", name(mode));
}

// Each alpha mode and DXGI's for it; DXGI answers whether the window takes
// it.
struct AlphaEntry {
    AlphaMode mode;
    DXGI_ALPHA_MODE dxgi;
};

constexpr std::array kAlphaEntries{
    AlphaEntry{.mode = AlphaMode::Opaque, .dxgi = DXGI_ALPHA_MODE_IGNORE},
    AlphaEntry{.mode = AlphaMode::Premultiplied, .dxgi = DXGI_ALPHA_MODE_PREMULTIPLIED},
    AlphaEntry{.mode = AlphaMode::Postmultiplied, .dxgi = DXGI_ALPHA_MODE_STRAIGHT},
    AlphaEntry{.mode = AlphaMode::Inherit, .dxgi = DXGI_ALPHA_MODE_UNSPECIFIED},
};

bool hasAlphaMode(AlphaMode mode) {
    return std::ranges::contains(kAlphaEntries, mode, &AlphaEntry::mode);
}

// Fails with RhiUnsupported for a mode DXGI has no setting for.
core::Result<DXGI_ALPHA_MODE> alphaModeFor(AlphaMode mode) {
    const auto* const found = std::ranges::find(kAlphaEntries, mode, &AlphaEntry::mode);
    if (found == kAlphaEntries.end()) {
        return core::failure(core::ErrorCode::RhiUnsupported, "DXGI has no setting for {} alpha", name(mode));
    }
    return found->dxgi;
}

// Fails with RhiSwapchainFailed for a mode DXGI built that BFR has none for.
core::Result<AlphaMode> alphaModeOf(DXGI_ALPHA_MODE mode) {
    const auto* const found = std::ranges::find(kAlphaEntries, mode, &AlphaEntry::dxgi);
    if (found == kAlphaEntries.end()) {
        return core::failure(core::ErrorCode::RhiSwapchainFailed, "DXGI built the swapchain with alpha mode {}",
                             static_cast<int>(mode));
    }
    return found->mode;
}

std::unexpected<core::Error> swapchainFailed(const char* call, HRESULT result) {
    return failure(core::ErrorCode::RhiSwapchainFailed, {.call = call, .result = result});
}

} // namespace

core::Result<void> checkWindow(const core::window::NativeWindow& window) {
    return std::visit(Checker{}, window);
}

SwapchainCaps surfaceCapsFor(bool tearing, const core::window::NativeWindow& window) {
    const auto presents = [](PresentMode mode) {
        return presentSetupFor(mode).has_value();
    };
    // A swapchain for a Win32 window is opaque: Godot sets
    // DXGI_ALPHA_MODE_IGNORE on every CreateSwapChainForHwnd and blends only
    // through a composition swapchain (rendering_device_driver_d3d12.cpp,
    // swap_chain_resize). DXGI has no call that says which a window takes.
    const bool composed = handleOf(window) == nullptr;
    const auto hasAlpha = [composed](AlphaMode mode) {
        return hasAlphaMode(mode) && (composed || mode == AlphaMode::Opaque);
    };
    return {.maxImageCount = DXGI_MAX_SWAP_CHAIN_BUFFERS,
            .fifo = presents(PresentMode::Fifo),
            .fifoRelaxed = presents(PresentMode::FifoRelaxed),
            .mailbox = presents(PresentMode::Mailbox),
            .immediate = tearing && presents(PresentMode::Immediate),
            .alphaOpaque = hasAlpha(AlphaMode::Opaque),
            .alphaPremultiplied = hasAlpha(AlphaMode::Premultiplied),
            .alphaPostmultiplied = hasAlpha(AlphaMode::Postmultiplied),
            .alphaInherit = hasAlpha(AlphaMode::Inherit)};
}

D3D12Swapchain::D3D12Swapchain(ComPtr<ID3D12Device> device, ComPtr<IDXGIFactory2> factory,
                               ComPtr<ID3D12CommandQueue> queue, const core::window::NativeWindow& window,
                               const SwapchainDesc& desc, const core::Logger& logger)
    : device_(std::move(device)),
      factory_(std::move(factory)),
      queue_(std::move(queue)),
      window_(handleOf(window)),
      desc_(desc),
      logger_(&logger) {}

void D3D12Swapchain::release() {
    acquired_.reset();
    submitted_ = false;
    count_ = 0;
    images_.clear();
    viewHeap_.Reset();
}

core::Result<void> D3D12Swapchain::makeChain(std::uint32_t width, std::uint32_t height) {
    const DxgiSwapchainFormat format = swapchainFormatFor(desc_.format, desc_.colorSpace);
    if (format.format == DXGI_FORMAT_UNKNOWN) {
        return core::failure(core::ErrorCode::RhiUnsupported,
                             "DXGI has no setting for the format with the color space that were asked");
    }
    const auto setup = presentSetupFor(desc_.presentMode);
    if (!setup) {
        return std::unexpected(setup.error());
    }
    const auto alpha = alphaModeFor(desc_.alphaMode);
    if (!alpha) {
        return std::unexpected(alpha.error());
    }
    if (chain_) {
        if (const HRESULT resized = chain_->ResizeBuffers(desc_.imageCount, width, height, DXGI_FORMAT_UNKNOWN,
                                                          setup->creationFlags);
            FAILED(resized)) {
            return swapchainFailed("IDXGISwapChain::ResizeBuffers", resized);
        }
        return {};
    }
    const DXGI_SWAP_CHAIN_DESC1 desc{.Width = width,
                                     .Height = height,
                                     .Format = format.format,
                                     .Stereo = FALSE,
                                     .SampleDesc = {.Count = 1, .Quality = 0},
                                     .BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT,
                                     .BufferCount = desc_.imageCount,
                                     .Scaling = DXGI_SCALING_STRETCH,
                                     .SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD,
                                     .AlphaMode = *alpha,
                                     .Flags = setup->creationFlags};
    ComPtr<IDXGISwapChain1> created;
    if (window_ != nullptr) {
        if (const HRESULT result = factory_->CreateSwapChainForHwnd(queue_.Get(), window_, &desc, nullptr, nullptr,
                                                                    &created);
            FAILED(result)) {
            return swapchainFailed("IDXGIFactory2::CreateSwapChainForHwnd", result);
        }
    }
    else if (const HRESULT result = factory_->CreateSwapChainForComposition(queue_.Get(), &desc, nullptr, &created);
             FAILED(result)) {
        return swapchainFailed("IDXGIFactory2::CreateSwapChainForComposition", result);
    }
    ComPtr<IDXGISwapChain3> chain;
    if (const HRESULT queried = created.As(&chain); FAILED(queried)) {
        return swapchainFailed("IDXGISwapChain1::QueryInterface(IDXGISwapChain3)", queried);
    }
    if (const HRESULT colored = chain->SetColorSpace1(format.colorSpace); FAILED(colored)) {
        return swapchainFailed("IDXGISwapChain3::SetColorSpace1", colored);
    }
    chain_ = std::move(chain);
    return {};
}

core::Result<DXGI_SWAP_CHAIN_DESC1> D3D12Swapchain::makeImages() {
    DXGI_SWAP_CHAIN_DESC1 built{};
    if (const HRESULT described = chain_->GetDesc1(&built); FAILED(described)) {
        return swapchainFailed("IDXGISwapChain1::GetDesc1", described);
    }
    const D3D12_DESCRIPTOR_HEAP_DESC heapDesc{.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                                              .NumDescriptors = built.BufferCount,
                                              .Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                                              .NodeMask = 0};
    if (const HRESULT made = device_->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&viewHeap_)); FAILED(made)) {
        return swapchainFailed("ID3D12Device::CreateDescriptorHeap", made);
    }
    viewStride_ = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    firstView_ = viewHeap_->GetCPUDescriptorHandleForHeapStart();
    for (UINT index = 0; index < built.BufferCount; ++index) {
        ComPtr<ID3D12Resource> image;
        if (const HRESULT got = chain_->GetBuffer(index, IID_PPV_ARGS(&image)); FAILED(got)) {
            images_.clear();
            return swapchainFailed("IDXGISwapChain::GetBuffer", got);
        }
        const D3D12_CPU_DESCRIPTOR_HANDLE view{.ptr = firstView_.ptr + (static_cast<SIZE_T>(index) * viewStride_)};
        device_->CreateRenderTargetView(image.Get(), nullptr, view);
        images_.push_back(std::move(image));
    }
    return built;
}

core::Result<SwapchainImages> D3D12Swapchain::build(std::uint32_t width, std::uint32_t height) {
    release();
    if (width == 0 || height == 0) {
        BFR_LOG_DEBUG(*logger_, "swapchain left empty: the window has no area");
        return SwapchainImages{.width = 0,
                               .height = 0,
                               .count = 0,
                               .format = desc_.format,
                               .colorSpace = desc_.colorSpace,
                               .presentMode = desc_.presentMode,
                               .alphaMode = desc_.alphaMode};
    }
    if (auto made = makeChain(width, height); !made) {
        return std::unexpected(std::move(made.error()));
    }
    const auto built = makeImages();
    if (!built) {
        return std::unexpected(built.error());
    }
    const auto alpha = alphaModeOf(built->AlphaMode);
    if (!alpha) {
        return std::unexpected(alpha.error());
    }
    count_ = built->BufferCount;
    BFR_LOG_INFO(*logger_, "swapchain {}x{}, {} images, {}", built->Width, built->Height, count_,
                 name(desc_.presentMode));
    // What DXGI built, as GetDesc1 says, but for the color space and present
    // mode, which DXGI has no getter for: IDXGISwapChain3 sets a color space
    // but never reports one, and a present mode is the sync interval and
    // flags each Present is given.
    return SwapchainImages{.width = built->Width,
                           .height = built->Height,
                           .count = count_,
                           .format = swapchainFormatOf(built->Format),
                           .colorSpace = desc_.colorSpace,
                           .presentMode = desc_.presentMode,
                           .alphaMode = *alpha};
}

core::Result<SwapchainStatus> D3D12Swapchain::acquire() {
    if (count_ == 0) {
        return core::failure(core::ErrorCode::RhiOutOfDate, "the swapchain has no images; resize it");
    }
    if (acquired_.has_value()) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "an image is already acquired; present it first");
    }
    acquired_ = chain_->GetCurrentBackBufferIndex();
    submitted_ = false;
    return SwapchainStatus::Matches;
}

ID3D12Resource* D3D12Swapchain::image() const {
    if (!acquired_.has_value()) {
        return nullptr;
    }
    return images_[*acquired_].Get();
}

D3D12_CPU_DESCRIPTOR_HANDLE D3D12Swapchain::view() const {
    if (!acquired_.has_value()) {
        return {};
    }
    return {.ptr = firstView_.ptr + (static_cast<SIZE_T>(*acquired_) * viewStride_)};
}

core::Result<SwapchainStatus> D3D12Swapchain::present() {
    if (!acquired_.has_value()) {
        return core::failure(core::ErrorCode::RhiInvalidCall, "no image is acquired");
    }
    const auto setup = presentSetupFor(desc_.presentMode);
    if (!setup) {
        return std::unexpected(setup.error());
    }
    const HRESULT result = chain_->Present(setup->syncInterval, setup->presentFlags);
    acquired_.reset();
    submitted_ = false;
    if (FAILED(result)) {
        return swapchainFailed("IDXGISwapChain::Present", result);
    }
    return SwapchainStatus::Matches;
}

} // namespace bfr::rhi::d3d12
