#include "d3d12/d3d12_instance.h"

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/instance.h"
#include "bfr/rhi/rhi_config.h"
#include "d3d12/d3d12_caps.h"
#include "d3d12/d3d12_device.h"
#include "d3d12/d3d12_formats.h"
#include "d3d12/d3d12_result.h"
#include "d3d12/d3d12_swapchain.h"

#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <dxgitype.h>
#include <windows.h>
#include <wrl/client.h>

#include <cstddef>
#include <expected>
#include <format>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace bfr::rhi::d3d12 {

namespace {

using Microsoft::WRL::ComPtr;

core::Severity severityOf(D3D12_MESSAGE_SEVERITY severity) {
    switch (severity) {
        case D3D12_MESSAGE_SEVERITY_CORRUPTION:
        case D3D12_MESSAGE_SEVERITY_ERROR:
            return core::Severity::Error;
        case D3D12_MESSAGE_SEVERITY_WARNING:
            return core::Severity::Warning;
        case D3D12_MESSAGE_SEVERITY_INFO:
            return core::Severity::Info;
        case D3D12_MESSAGE_SEVERITY_MESSAGE:
            break;
    }
    return core::Severity::Trace;
}

// Hands each message to the logger context points at. Validation messages are
// the API's behaviour, not BFR's diagnostics, so BFR_LOGGING doesn't drop them.
void WINAPI onMessage(D3D12_MESSAGE_CATEGORY /*category*/, D3D12_MESSAGE_SEVERITY severity, D3D12_MESSAGE_ID /*id*/,
                      LPCSTR description, void* context) {
    const auto& logger = *static_cast<const core::Logger*>(context);
    logger.send(severityOf(severity), description != nullptr ? description : "");
}

} // namespace

core::Result<std::unique_ptr<D3D12Instance>> D3D12Instance::create(const InstanceDesc& desc, const RhiConfig& config) {
    auto rhi = std::make_unique<D3D12Instance>(config, desc.logger);
    // The debug layer is switched on before the first device or factory.
    if (config.validation) {
        if (auto enabled = enableDebugLayer(); !enabled) {
            return std::unexpected(std::move(enabled.error()));
        }
    }
    if (auto created = rhi->createFactory(); !created) {
        return std::unexpected(std::move(created.error()));
    }
    if (auto listed = rhi->listAdapters(); !listed) {
        return std::unexpected(std::move(listed.error()));
    }
    return rhi;
}

D3D12Instance::D3D12Instance(const RhiConfig& config, core::Logger logger)
    : Instance(config), logger_(std::move(logger)) {}

D3D12Instance::~D3D12Instance() {
    for (const D3D12Adapter& native : native_) {
        if (native.messages) {
            native.messages->UnregisterMessageCallback(native.cookie);
        }
    }
}

core::Result<void> D3D12Instance::enableDebugLayer() {
    ComPtr<ID3D12Debug> debug;
    if (const HRESULT result = D3D12GetDebugInterface(IID_PPV_ARGS(&debug)); FAILED(result)) {
        return core::failure(
            core::ErrorCode::RhiValidationUnavailable,
            std::format("the D3D12 debug layer is not available; Windows supplies it with the Graphics Tools "
                        "optional feature: {}",
                        D3D12Failure{.call = "D3D12GetDebugInterface", .result = result}.message()));
    }
    debug->EnableDebugLayer();
    return {};
}

core::Result<void> D3D12Instance::createFactory() {
    const UINT flags = config().validation ? DXGI_CREATE_FACTORY_DEBUG : 0;
    if (const HRESULT result = CreateDXGIFactory2(flags, IID_PPV_ARGS(&factory_)); FAILED(result)) {
        return failure(core::ErrorCode::RhiInstanceFailed, {.call = "CreateDXGIFactory2", .result = result});
    }
    // The type is BOOL, whose size is not bool's.
    BOOL allowed = FALSE;
    if (const HRESULT result = factory_->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allowed,
                                                             sizeof(allowed));
        SUCCEEDED(result)) {
        tearing_ = allowed != FALSE;
    }
    else {
        BFR_LOG_WARN(logger_, "{}",
                     D3D12Failure{.call = "IDXGIFactory5::CheckFeatureSupport(PRESENT_ALLOW_TEARING)", .result = result}
                         .message());
    }
    return {};
}

core::Result<void> D3D12Instance::reportMessages(D3D12Adapter& native) {
    const auto unavailable = [](const char* call, HRESULT result) {
        return core::failure(core::ErrorCode::RhiValidationUnavailable,
                             D3D12Failure{.call = call, .result = result}.message());
    };
    if (const HRESULT result = native.device.As(&native.messages); FAILED(result)) {
        return unavailable("ID3D12Device::QueryInterface(ID3D12InfoQueue1)", result);
    }
    // The messages go to the logger and not to the debugger as well.
    native.messages->SetMuteDebugOutput(TRUE);
    // Messages quieter than the logger wants are not stored, so not sent.
    std::vector<D3D12_MESSAGE_SEVERITY> quiet;
    if (!logger_.wants(core::Severity::Info)) {
        quiet.push_back(D3D12_MESSAGE_SEVERITY_INFO);
    }
    if (!logger_.wants(core::Severity::Trace)) {
        quiet.push_back(D3D12_MESSAGE_SEVERITY_MESSAGE);
    }
    D3D12_INFO_QUEUE_FILTER filter{};
    filter.DenyList.NumSeverities = static_cast<UINT>(quiet.size());
    filter.DenyList.pSeverityList = quiet.data();
    if (const HRESULT result = native.messages->PushStorageFilter(&filter); FAILED(result)) {
        return unavailable("ID3D12InfoQueue::PushStorageFilter", result);
    }
    if (const HRESULT result = native.messages->RegisterMessageCallback(onMessage, D3D12_MESSAGE_CALLBACK_FLAG_NONE,
                                                                        &logger_, &native.cookie);
        FAILED(result)) {
        native.messages.Reset();
        return unavailable("ID3D12InfoQueue1::RegisterMessageCallback", result);
    }
    return {};
}

core::Result<void> D3D12Instance::listAdapter(IDXGIAdapter1* found, std::vector<std::string>& rejected) {
    DXGI_ADAPTER_DESC1 desc{};
    if (const HRESULT described = found->GetDesc1(&desc); FAILED(described)) {
        return failure(core::ErrorCode::RhiInstanceFailed, {.call = "IDXGIAdapter1::GetDesc1", .result = described});
    }
    const std::string name = nameOf(desc);

    D3D12Adapter native;
    if (const HRESULT queried = found->QueryInterface(IID_PPV_ARGS(&native.adapter)); FAILED(queried)) {
        rejected.push_back(std::format(
            "{}: {}", name,
            D3D12Failure{.call = "IDXGIAdapter1::QueryInterface(IDXGIAdapter3)", .result = queried}.message()));
        return {};
    }
    if (const HRESULT made = D3D12CreateDevice(native.adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                               IID_PPV_ARGS(&native.device));
        FAILED(made)) {
        const std::string reason = D3D12Failure{.call = "D3D12CreateDevice", .result = made}.message();
        BFR_LOG_INFO(logger_, "adapter {}: {}", name, reason);
        rejected.push_back(std::format("{}: {}", name, reason));
        return {};
    }
    const std::string reason = belowFloor(native.device.Get());
    BFR_LOG_INFO(logger_, "adapter {}: {}", name,
                 reason.empty() ? std::string{"listed"} : "below the floor, " + reason);
    if (!reason.empty()) {
        rejected.push_back(std::format("{}: below the floor, {}", name, reason));
        return {};
    }
    if (config().validation) {
        if (auto reported = reportMessages(native); !reported) {
            return reported;
        }
    }
    adapters_.push_back({.name = name, .type = typeOf(desc), .caps = readCaps(desc, native.device.Get())});
    native_.push_back(std::move(native));
    return {};
}

core::Result<void> D3D12Instance::listAdapters() {
    // Each adapter not listed, and why.
    std::vector<std::string> rejected;
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> found;
        const HRESULT result = factory_->EnumAdapterByGpuPreference(index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                                                    IID_PPV_ARGS(&found));
        if (result == DXGI_ERROR_NOT_FOUND) {
            break;
        }
        if (FAILED(result)) {
            return failure(core::ErrorCode::RhiInstanceFailed,
                           {.call = "IDXGIFactory6::EnumAdapterByGpuPreference", .result = result});
        }
        if (auto listed = listAdapter(found.Get(), rejected); !listed) {
            return listed;
        }
    }
    if (adapters_.empty()) {
        return core::failure(core::ErrorCode::RhiNoAdapter, core::error::withReasons("no usable adapter", rejected));
    }
    return {};
}

core::Result<void> D3D12Instance::checkAdapter(std::size_t adapter) const {
    if (adapter >= adapters_.size()) {
        return core::failure(core::ErrorCode::RhiAdapterOutOfRange,
                             std::format("adapter {} is not one of the {} listed", adapter, adapters_.size()));
    }
    return {};
}

core::Result<std::unique_ptr<Device>> D3D12Instance::createDevice(std::size_t adapter,
                                                                  const core::window::NativeWindow& presentTo,
                                                                  const DeviceQueues& queues) {
    if (auto checked = checkAdapter(adapter); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // Every device presents to a Win32 window or to none, so a window only has
    // to be one of those.
    if (!std::holds_alternative<std::monostate>(presentTo)) {
        if (auto checked = checkWindow(presentTo); !checked) {
            return std::unexpected(std::move(checked.error()));
        }
    }
    const D3D12Adapter& native = native_[adapter];
    const Caps& caps = adapters_[adapter].caps;
    return D3D12Device::create(*this, native.device, native.adapter, caps.memory.heaps.size(), queues);
}

core::Result<FormatCaps> D3D12Instance::formatCaps(std::size_t adapter, Format format) const {
    if (auto checked = checkAdapter(adapter); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    return formatCapsFor(native_[adapter].device.Get(), format);
}

core::Result<SwapchainCaps> D3D12Instance::swapchainCaps(std::size_t adapter,
                                                         const core::window::NativeWindow& window) const {
    if (auto checked = checkAdapter(adapter); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto checked = checkWindow(window); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    return surfaceCapsFor(tearing_);
}

core::Result<std::vector<SwapchainFormat>> D3D12Instance::swapchainFormats(
    std::size_t adapter, const core::window::NativeWindow& window) const {
    if (auto checked = checkAdapter(adapter); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto checked = checkWindow(window); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    const auto formats = d3d12::swapchainFormats();
    return std::vector<SwapchainFormat>(formats.begin(), formats.end());
}

} // namespace bfr::rhi::d3d12
