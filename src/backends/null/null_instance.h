#pragma once

// The null backend: no GPU. One adapter that reports nothing, a device that accepts
// every call and draws nothing. In every build, so that every agnostic test
// runs on two backends on every machine, and anything shaped like one API
// fails against the other.

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/instance.h"
#include "bfr/rhi/rhi_config.h"

#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace bfr::rhi::null {

/// What the null adapter says a swapchain on any window may be.
[[nodiscard]] SwapchainCaps nullSurfaceCaps();

/// The formats and color spaces the null adapter presents in.
[[nodiscard]] std::span<const SwapchainFormat> nullSurfaceFormats();

class NullInstance final : public Instance {
public:
    [[nodiscard]] static core::Result<std::unique_ptr<NullInstance>> create(const InstanceDesc& desc,
                                                                            const RhiConfig& config);

    NullInstance(const RhiConfig& config, core::Logger logger);

    [[nodiscard]] std::span<const Adapter> adapters() const override { return adapters_; }
    [[nodiscard]] core::Result<std::unique_ptr<Device>> createDevice(std::size_t adapter,
                                                                     const core::window::NativeWindow& presentTo,
                                                                     const DeviceQueues& queues) override;
    [[nodiscard]] core::Result<FormatCaps> formatCaps(std::size_t adapter, Format format) const override;
    [[nodiscard]] core::Result<SwapchainCaps> swapchainCaps(std::size_t adapter,
                                                            const core::window::NativeWindow& window) const override;
    [[nodiscard]] core::Result<std::vector<SwapchainFormat>> swapchainFormats(
        std::size_t adapter, const core::window::NativeWindow& window) const override;

    /// Where the Instance and its devices send their messages.
    [[nodiscard]] const core::Logger& logger() const { return logger_; }

private:
    // Fails with RhiAdapterOutOfRange when adapter is not an index into adapters_.
    [[nodiscard]] core::Result<void> checkAdapter(std::size_t adapter) const;

    core::Logger logger_;
    std::array<Adapter, 1> adapters_;
};

} // namespace bfr::rhi::null
