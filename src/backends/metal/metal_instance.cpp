#include "metal/metal_instance.h"

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/instance.h"
#include "bfr/rhi/rhi_config.h"
#include "metal/metal_caps.h"
#include "metal/metal_device.h"
#include "metal/metal_formats.h"
#include "metal/metal_swapchain.h"
#include "metal/metal_util.h"
#include "rhi/device_limits.h"
#include "rhi/format_check.h"

#include <Foundation/NSArray.hpp>
#include <Foundation/NSTypes.hpp>
#include <Metal/MTLDevice.hpp>
#include <TargetConditionals.h>

#include <cstddef>
#include <cstdlib>
#include <expected>
#include <format>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace bfr::rhi::metal {

core::Result<std::unique_ptr<MetalInstance>> MetalInstance::create(const InstanceDesc& desc, const RhiConfig& config) {
    auto rhi = std::make_unique<MetalInstance>(config, desc.logger);
    if (config.validation) {
        // Metal's validation layer is switched on by the environment, before
        // the first device exists, and it reports to the system log.
        setenv("MTL_DEBUG_LAYER", "1", 0);
    }
    if (auto listed = rhi->listAdapters(); !listed) {
        return std::unexpected(std::move(listed.error()));
    }
    return rhi;
}

MetalInstance::MetalInstance(const RhiConfig& config, core::Logger logger)
    : Instance(config), logger_(std::move(logger)) {}

core::Result<void> MetalInstance::listAdapters() {
    const AutoreleasePool pool;
    std::vector<Ref<MTL::Device>> found;
#if TARGET_OS_OSX
    NS::Array* all = MTL::CopyAllDevices();
    if (all != nullptr) {
        for (NS::UInteger i = 0; i < all->count(); ++i) {
            found.push_back(share(all->object<MTL::Device>(i)));
        }
        all->release();
    }
#else
    if (MTL::Device* device = MTL::CreateSystemDefaultDevice(); device != nullptr) {
        found.push_back(adopt(device));
    }
#endif

    // Each device not listed, and why.
    std::vector<std::string> rejected;
    for (const Ref<MTL::Device>& device : found) {
        const std::string name = toString(device->name());
        const std::string reason = belowMinimum(device.get());
        BFR_LOG_INFO(logger_, "adapter {}: {}", name,
                     reason.empty() ? std::string{"listed"} : "below BFR's minimum, " + reason);
        if (!reason.empty()) {
            rejected.push_back(std::format("{}: below BFR's minimum, {}", name, reason));
            continue;
        }
        adapters_.push_back({.name = name,
                             .type = device->hasUnifiedMemory() ? AdapterType::Integrated : AdapterType::Discrete,
                             .caps = readCaps(device.get())});
        devices_.push_back(device);
    }
    if (adapters_.empty()) {
        return core::failure(core::ErrorCode::RhiNoAdapter, core::error::withReasons("no usable adapter", rejected));
    }
    return {};
}

core::Result<void> MetalInstance::checkAdapter(std::size_t adapter) const {
    if (adapter >= adapters_.size()) {
        return core::failure(core::ErrorCode::RhiAdapterOutOfRange, "adapter {} is not one of the {} listed", adapter,
                             adapters_.size());
    }
    return {};
}

core::Result<std::unique_ptr<Device>> MetalInstance::createDevice(std::size_t adapter,
                                                                  const core::window::NativeWindow& presentTo,
                                                                  const DeviceDesc& desc) {
    if (auto checked = checkAdapter(adapter); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto checked = checkLimits(desc); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // Any queue's command buffer can present a drawable.
    if (auto checked = checkPresentQueue(presentTo, queueCount(desc)); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto checked = checkUnranked(desc, "Metal"); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto checked = checkNormalLevels(desc, "Metal has no queue levels"); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // Every device presents to a CAMetalLayer, so a window only has to be one.
    if (!std::holds_alternative<std::monostate>(presentTo)) {
        if (auto checked = checkWindow(presentTo); !checked) {
            return std::unexpected(std::move(checked.error()));
        }
    }
    const Caps& caps = adapters_[adapter].caps;
    return MetalDevice::create(*this, devices_[adapter], caps.features, caps.resolve, desc);
}

core::Result<FormatCaps> MetalInstance::formatCaps(std::size_t adapter, Format format) const {
    if (auto checked = checkAdapter(adapter); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto checked = checkFormat(format); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    return formatCapsFor(devices_[adapter].get(), format);
}

core::Result<SwapchainCaps> MetalInstance::swapchainCaps(std::size_t adapter,
                                                         const core::window::NativeWindow& window) const {
    if (auto checked = checkAdapter(adapter); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto checked = checkWindow(window); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    return surfaceCapsFor();
}

core::Result<std::vector<SwapchainFormat>> MetalInstance::swapchainFormats(
    std::size_t adapter, const core::window::NativeWindow& window) const {
    if (auto checked = checkAdapter(adapter); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto checked = checkWindow(window); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    const auto formats = metal::swapchainFormats();
    return std::vector<SwapchainFormat>(formats.begin(), formats.end());
}

} // namespace bfr::rhi::metal
