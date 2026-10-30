#pragma once

#include "bfr/core/error.h"

#include <dxgi1_6.h>
#include <dxgitype.h>
#include <windows.h>

#include <cstdint>
#include <expected>
#include <format>
#include <string>
#include <string_view>

namespace bfr::rhi::d3d12 {

/// A D3D12 or DXGI call that failed, and what it returned.
struct D3D12Failure {
    std::string call;
    HRESULT result = S_OK;

    /// "call: E_... (0x...)", the HRESULT's name when it has one.
    [[nodiscard]] std::string message() const {
        return std::format("{}: {} (0x{:08X})", call, nameOf(result), static_cast<std::uint32_t>(result));
    }

    /// The name of result as its header spells it, "error" for one not listed.
    [[nodiscard]] static std::string_view nameOf(HRESULT result) {
        switch (result) {
            case E_OUTOFMEMORY:
                return "E_OUTOFMEMORY";
            case E_INVALIDARG:
                return "E_INVALIDARG";
            case E_FAIL:
                return "E_FAIL";
            case E_NOTIMPL:
                return "E_NOTIMPL";
            case E_NOINTERFACE:
                return "E_NOINTERFACE";
            case DXGI_ERROR_INVALID_CALL:
                return "DXGI_ERROR_INVALID_CALL";
            case DXGI_ERROR_UNSUPPORTED:
                return "DXGI_ERROR_UNSUPPORTED";
            case DXGI_ERROR_DEVICE_REMOVED:
                return "DXGI_ERROR_DEVICE_REMOVED";
            case DXGI_ERROR_DEVICE_HUNG:
                return "DXGI_ERROR_DEVICE_HUNG";
            case DXGI_ERROR_DEVICE_RESET:
                return "DXGI_ERROR_DEVICE_RESET";
            case DXGI_ERROR_DRIVER_INTERNAL_ERROR:
                return "DXGI_ERROR_DRIVER_INTERNAL_ERROR";
            case DXGI_ERROR_NOT_CURRENTLY_AVAILABLE:
                return "DXGI_ERROR_NOT_CURRENTLY_AVAILABLE";
            case DXGI_ERROR_SDK_COMPONENT_MISSING:
                return "DXGI_ERROR_SDK_COMPONENT_MISSING";
            default:
                return "error";
        }
    }
};

/// The code for a call that failed with result: running out of memory and a
/// lost device keep codes of their own, since the caller deals with each
/// differently; anything else is code.
[[nodiscard]] inline core::ErrorCode codeFor(core::ErrorCode code, HRESULT result) {
    switch (result) {
        case E_OUTOFMEMORY:
            return core::ErrorCode::RhiOutOfMemory;
        case DXGI_ERROR_DEVICE_REMOVED:
        case DXGI_ERROR_DEVICE_HUNG:
        case DXGI_ERROR_DEVICE_RESET:
        case DXGI_ERROR_DRIVER_INTERNAL_ERROR:
            return core::ErrorCode::RhiDeviceLost;
        default:
            return code;
    }
}

/// The error for failed, coded as codeFor says.
[[nodiscard]] inline std::unexpected<core::Error> failure(core::ErrorCode code, const D3D12Failure& failed) {
    return core::failure(codeFor(code, failed.result), failed.message());
}

} // namespace bfr::rhi::d3d12
