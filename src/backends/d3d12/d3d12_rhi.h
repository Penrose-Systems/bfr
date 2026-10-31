#pragma once

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/rhi.h"
#include "bfr/rhi/rhi_config.h"

#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <windows.h>
#include <wrl/client.h>

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace bfr::rhi::d3d12 {

/// What an Rhi keeps of each adapter it lists, beside its Adapter.
struct D3D12Adapter {
    Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter;
    /// The device its caps were read from; every Device on the adapter uses it.
    Microsoft::WRL::ComPtr<ID3D12Device> device;
    /// With validation on, where the device sends the debug layer's messages,
    /// and the cookie of the callback that sends them to the logger.
    Microsoft::WRL::ComPtr<ID3D12InfoQueue1> messages;
    DWORD cookie = 0;
};

class D3D12Rhi final : public Rhi {
public:
    /// An Rhi over the adapters DXGI lists. With validation on, the D3D12 debug
    /// layer is switched on and its messages go to the logger. Fails with
    /// RhiValidationUnavailable when the debug layer is missing, and
    /// RhiNoAdapter when no adapter meets the floor, naming each and why.
    [[nodiscard]] static core::Result<std::unique_ptr<D3D12Rhi>> create(const RhiDesc& desc, const RhiConfig& config);

    D3D12Rhi(const RhiConfig& config, core::Logger logger);
    ~D3D12Rhi() override;
    D3D12Rhi(const D3D12Rhi&) = delete;
    D3D12Rhi& operator=(const D3D12Rhi&) = delete;
    D3D12Rhi(D3D12Rhi&&) = delete;
    D3D12Rhi& operator=(D3D12Rhi&&) = delete;

    [[nodiscard]] std::span<const Adapter> adapters() const override { return adapters_; }
    [[nodiscard]] core::Result<std::unique_ptr<Device>> createDevice(
        std::size_t adapter, const core::window::NativeWindow& presentTo) override;
    [[nodiscard]] core::Result<FormatCaps> formatCaps(std::size_t adapter, Format format) const override;
    [[nodiscard]] core::Result<SurfaceCaps> surfaceCaps(std::size_t adapter,
                                                        const core::window::NativeWindow& window) const override;
    [[nodiscard]] core::Result<std::vector<SurfaceFormat>> surfaceFormats(
        std::size_t adapter, const core::window::NativeWindow& window) const override;

    using Rhi::config;

    /// Where the Rhi and its devices send their messages.
    [[nodiscard]] const core::Logger& logger() const { return logger_; }

    /// The DXGI factory swapchains are made with.
    [[nodiscard]] IDXGIFactory6* factory() const { return factory_.Get(); }

private:
    [[nodiscard]] static core::Result<void> enableDebugLayer();
    [[nodiscard]] core::Result<void> createFactory();
    [[nodiscard]] core::Result<void> listAdapters();
    // Lists found, an adapter DXGI listed, when it meets the floor; otherwise adds
    // why not to rejected.
    [[nodiscard]] core::Result<void> listAdapter(IDXGIAdapter1* found, std::vector<std::string>& rejected);
    // Has native's device send the debug layer's messages to the logger.
    [[nodiscard]] core::Result<void> reportMessages(D3D12Adapter& native);
    // Fails with RhiAdapterOutOfRange when adapter is not an index into adapters_.
    [[nodiscard]] core::Result<void> checkAdapter(std::size_t adapter) const;

    core::Logger logger_;
    Microsoft::WRL::ComPtr<IDXGIFactory6> factory_;
    bool tearing_ = false;
    std::vector<Adapter> adapters_;
    // Index-aligned with adapters_.
    std::vector<D3D12Adapter> native_;
};

} // namespace bfr::rhi::d3d12
