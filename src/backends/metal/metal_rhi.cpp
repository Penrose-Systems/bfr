#include "metal/metal_rhi.h"

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/rhi.h"
#include "bfr/rhi/rhi_config.h"
#include "metal/metal_caps.h"
#include "metal/metal_device.h"
#include "metal/metal_formats.h"
#include "metal/metal_swapchain.h"
#include "metal/metal_util.h"

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

core::Result<std::unique_ptr<MetalRhi>> MetalRhi::create(const RhiDesc& desc, const RhiConfig& config) {
    auto rhi = std::make_unique<MetalRhi>(config, desc.logger);
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

MetalRhi::MetalRhi(const RhiConfig& config, core::Logger logger) : Rhi(config), logger_(std::move(logger)) {}

core::Result<void> MetalRhi::listAdapters() {
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
        const std::string reason = belowFloor(device.get());
        BFR_LOG_INFO(logger_, "adapter {}: {}", name,
                     reason.empty() ? std::string{"listed"} : "below the floor, " + reason);
        if (!reason.empty()) {
            rejected.push_back(std::format("{}: below the floor, {}", name, reason));
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

core::Result<void> MetalRhi::checkAdapter(std::size_t adapter) const {
    if (adapter >= adapters_.size()) {
        return core::failure(core::ErrorCode::RhiAdapterOutOfRange,
                             std::format("adapter {} is not one of the {} listed", adapter, adapters_.size()));
    }
    return {};
}

core::Result<std::unique_ptr<Device>> MetalRhi::createDevice(std::size_t adapter,
                                                             const core::window::NativeWindow& presentTo) {
    if (auto checked = checkAdapter(adapter); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // Every device presents to a CAMetalLayer, so a window only has to be one.
    if (!std::holds_alternative<std::monostate>(presentTo)) {
        if (auto checked = checkWindow(presentTo); !checked) {
            return std::unexpected(std::move(checked.error()));
        }
    }
    return MetalDevice::create(*this, devices_[adapter]);
}

core::Result<FormatCaps> MetalRhi::formatCaps(std::size_t adapter, Format format) const {
    if (auto checked = checkAdapter(adapter); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // Metal has no call for what a format does, so every field is false.
    return FormatCaps{.format = format};
}

core::Result<SurfaceCaps> MetalRhi::surfaceCaps(std::size_t adapter, const core::window::NativeWindow& window) const {
    if (auto checked = checkAdapter(adapter); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto checked = checkWindow(window); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    return surfaceCapsFor();
}

core::Result<std::vector<SurfaceFormat>> MetalRhi::surfaceFormats(std::size_t adapter,
                                                                  const core::window::NativeWindow& window) const {
    if (auto checked = checkAdapter(adapter); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (auto checked = checkWindow(window); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    const auto formats = metal::surfaceFormats();
    return std::vector<SurfaceFormat>(formats.begin(), formats.end());
}

} // namespace bfr::rhi::metal
