#pragma once

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/rhi.h"
#include "bfr/rhi/rhi_config.h"
#include "metal/metal_util.h"

#include <CoreGraphics/CGColorSpace.h>
#include <CoreGraphics/CGGeometry.h>
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <objc/message.h>
#include <objc/runtime.h>
#include <QuartzCore/QuartzCore.hpp>
#include <TargetConditionals.h>

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace bfr::rhi::metal {

class MetalRhi final : public Rhi {
public:
    /// An Rhi over the Metal devices the system lists. With validation on,
    /// Metal's own validation layer is switched on, which reports to the
    /// system and not to the logger. Fails with RhiNoAdapter when no device
    /// meets the floor, naming each and why.
    [[nodiscard]] static core::Result<std::unique_ptr<MetalRhi>> create(const RhiDesc& desc, const RhiConfig& config);

    MetalRhi(const RhiConfig& config, core::Logger logger);
    ~MetalRhi() override = default;
    MetalRhi(const MetalRhi&) = delete;
    MetalRhi& operator=(const MetalRhi&) = delete;
    MetalRhi(MetalRhi&&) = delete;
    MetalRhi& operator=(MetalRhi&&) = delete;

    [[nodiscard]] std::span<const Adapter> adapters() const override { return adapters_; }
    [[nodiscard]] core::Result<std::unique_ptr<Device>> createDevice(std::size_t adapter,
                                                                     const core::window::NativeWindow& presentTo,
                                                                     const DeviceQueues& queues) override;
    [[nodiscard]] core::Result<FormatCaps> formatCaps(std::size_t adapter, Format format) const override;
    [[nodiscard]] core::Result<SwapchainCaps> swapchainCaps(std::size_t adapter,
                                                            const core::window::NativeWindow& window) const override;
    [[nodiscard]] core::Result<std::vector<SwapchainFormat>> swapchainFormats(
        std::size_t adapter, const core::window::NativeWindow& window) const override;

    using Rhi::config;

    /// Where the Rhi and its devices send their messages.
    [[nodiscard]] const core::Logger& logger() const { return logger_; }

private:
    [[nodiscard]] core::Result<void> listAdapters();
    // Fails with RhiAdapterOutOfRange when adapter is not an index into adapters_.
    [[nodiscard]] core::Result<void> checkAdapter(std::size_t adapter) const;

    core::Logger logger_;
    std::vector<Adapter> adapters_;
    // Index-aligned with adapters_.
    std::vector<Ref<MTL::Device>> devices_;
};

} // namespace bfr::rhi::metal
