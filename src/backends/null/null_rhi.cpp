#include "null/null_rhi.h"

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/rhi.h"
#include "bfr/rhi/rhi_config.h"
#include "null/null_device.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <memory>
#include <span>
#include <utility>
#include <variant>
#include <vector>

namespace bfr::rhi::null {

namespace {

// The least images a swapchain on the null adapter may have.
constexpr std::uint32_t kMinImages = 2;

// SurfaceCaps::width and height when the swapchain's size sets the window's.
constexpr std::uint32_t kSizeFromSwapchain = 0xFFFFFFFF;

// The null adapter draws nothing, so a window on it has no largest size.
constexpr std::uint32_t kNoLargest = std::numeric_limits<std::uint32_t>::max();

constexpr std::array kSurfaceFormats{
    SurfaceFormat{.format = Format::Bgra8Unorm, .colorSpace = ColorSpace::Srgb},
    SurfaceFormat{.format = Format::Bgra8Srgb, .colorSpace = ColorSpace::Srgb},
    SurfaceFormat{.format = Format::Rgba8Unorm, .colorSpace = ColorSpace::Srgb},
    SurfaceFormat{.format = Format::Rgba8Srgb, .colorSpace = ColorSpace::Srgb},
};

} // namespace

SurfaceCaps nullSurfaceCaps() {
    return {
        .minImageCount = kMinImages,
        .maxImageCount = 0,
        .width = kSizeFromSwapchain,
        .height = kSizeFromSwapchain,
        .minWidth = 1,
        .minHeight = 1,
        .maxWidth = kNoLargest,
        .maxHeight = kNoLargest,
        .rotation = SurfaceRotation::None,
        .fifo = true,
        .fifoRelaxed = false,
        .mailbox = false,
        .immediate = false,
        .alphaOpaque = true,
        .alphaPremultiplied = false,
        .alphaPostmultiplied = false,
        .alphaInherit = false,
    };
}

std::span<const SurfaceFormat> nullSurfaceFormats() {
    return kSurfaceFormats;
}

core::Result<std::unique_ptr<NullRhi>> NullRhi::create(const RhiDesc& desc, const RhiConfig& config) {
    return std::make_unique<NullRhi>(config, desc.logger);
}

NullRhi::NullRhi(const RhiConfig& config, core::Logger logger)
    : Rhi(config),
      logger_(std::move(logger)),
      // No GPU, so nothing to report: every limit 0, no heaps or queues, no
      // optional feature. The device accepts every call anyway.
      adapters_{Adapter{.name = "null", .type = AdapterType::Other, .caps = {}}} {
    BFR_LOG_INFO(logger_, "null rhi, one adapter with nothing behind it, validation {}",
                 config.validation ? "on" : "off");
}

core::Result<void> NullRhi::checkAdapter(std::size_t adapter) const {
    if (adapter >= adapters_.size()) {
        return core::failure(core::ErrorCode::RhiAdapterOutOfRange,
                             std::format("adapter {} is not one of the {} listed", adapter, adapters_.size()));
    }
    return {};
}

core::Result<std::unique_ptr<Device>> NullRhi::createDevice(std::size_t adapter,
                                                            const core::window::NativeWindow& /*presentTo*/) {
    if (auto checked = checkAdapter(adapter); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // The null device presents to every window, by presenting nowhere.
    return std::make_unique<NullDevice>(*this);
}

core::Result<FormatCaps> NullRhi::formatCaps(std::size_t adapter, Format format) const {
    if (auto checked = checkAdapter(adapter); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    // No GPU, so no format has a use.
    return FormatCaps{.format = format};
}

core::Result<SurfaceCaps> NullRhi::surfaceCaps(std::size_t adapter, const core::window::NativeWindow& window) const {
    if (auto checked = checkAdapter(adapter); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (std::holds_alternative<std::monostate>(window)) {
        return core::failure(core::ErrorCode::RhiSurfaceFailed, "there is no window");
    }
    return nullSurfaceCaps();
}

core::Result<std::vector<SurfaceFormat>> NullRhi::surfaceFormats(std::size_t adapter,
                                                                 const core::window::NativeWindow& window) const {
    if (auto checked = checkAdapter(adapter); !checked) {
        return std::unexpected(std::move(checked.error()));
    }
    if (std::holds_alternative<std::monostate>(window)) {
        return core::failure(core::ErrorCode::RhiSurfaceFailed, "there is no window");
    }
    const auto formats = nullSurfaceFormats();
    return std::vector<SurfaceFormat>(formats.begin(), formats.end());
}

} // namespace bfr::rhi::null
