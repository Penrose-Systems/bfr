#pragma once

#include "bfr/core/error.h"
#include "bfr/core/log.h"
#include "bfr/core/window.h"
#include "bfr/rhi/caps.h"
#include "bfr/rhi/device.h"
#include "bfr/rhi/format.h"
#include "bfr/rhi/rhi_config.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace bfr::rhi {

enum class AdapterType : std::uint8_t {
    Discrete,
    Integrated,
    Virtual,
    Cpu,
    Other,
};

/// A GPU a device can be created on.
struct Adapter {
    std::string name;
    AdapterType type = AdapterType::Other;
    Caps caps;
};

/// The native graphics API an Rhi runs on. The values are bfr_backend's, so
/// that zero, which a default-initialised RhiDesc holds, names no backend.
enum class Backend : std::uint8_t {
    Vulkan = 1,
    D3D12 = 2,
    Metal = 3,
    /// No GPU: one adapter that reports nothing, a device that accepts every
    /// call and draws nothing. In every build, inside the library only: the
    /// agnostic tests run on it, and the C API neither lists nor accepts it.
    Null = 4,
};

[[nodiscard]] constexpr std::string_view name(Backend backend) noexcept {
    switch (backend) {
        case Backend::Vulkan:
            return "vulkan";
        case Backend::D3D12:
            return "d3d12";
        case Backend::Metal:
            return "metal";
        case Backend::Null:
            return "null";
    }
    return "unknown";
}

/// The backends this build has, in Backend order. Rhi::create accepts these
/// and refuses every other.
[[nodiscard]] std::span<const Backend> backends() noexcept;

struct RhiDesc {
    /// The API the Rhi runs on. The caller names one; there is no default.
    Backend backend{};

    /// Where the Rhi, its devices and, when RhiConfig turns it on, validation
    /// send their messages. Nowhere, until the caller says.
    core::Logger logger = core::Logger::none();
};

/// The graphics API's entry point and the adapters it found.
class Rhi {
public:
    /// An Rhi on desc.backend. Fails with RhiBackendUnavailable when
    /// desc.backend is not in backends(), RhiLoaderFailed when the API's loader
    /// won't load, giving the OS's reason, RhiValidationUnavailable when
    /// validation is asked for and the system can't provide it,
    /// RhiInstanceFailed when the API refuses the instance, and RhiNoAdapter
    /// when no adapter meets the floor, naming each and why. With validation
    /// on, a refused instance's error carries the messages the API sent
    /// meanwhile.
    [[nodiscard]] static core::Result<std::unique_ptr<Rhi>> create(const RhiDesc& desc, const RhiConfig& config);

    virtual ~Rhi() = default;
    Rhi(const Rhi&) = delete;
    Rhi& operator=(const Rhi&) = delete;
    Rhi(Rhi&&) = delete;
    Rhi& operator=(Rhi&&) = delete;

    [[nodiscard]] virtual std::span<const Adapter> adapters() const = 0;

    /// A device on adapters()[adapter], destroyed before this Rhi. Its
    /// graphics queue presents to presentTo where a queue family can, else a
    /// queue of its own does; with no window, graphics presents. Fails with
    /// RhiAdapterOutOfRange when adapter is not an index into adapters(),
    /// RhiSurfaceFailed as Device::createSwapchain describes,
    /// RhiPresentUnsupported when no queue family presents to presentTo, and
    /// RhiDeviceFailed when the adapter has no graphics queue or the API
    /// refuses the device.
    [[nodiscard]] virtual core::Result<std::unique_ptr<Device>> createDevice(
        std::size_t adapter, const core::window::NativeWindow& presentTo) = 0;

    // The queries below fail with RhiAdapterOutOfRange when adapter is not an
    // index into adapters(). The API answers each afresh.

    /// What adapters()[adapter] does with format. Fails with the API's error
    /// when the query fails for a reason other than the format being
    /// unsupported.
    [[nodiscard]] virtual core::Result<FormatCaps> formatCaps(std::size_t adapter, Format format) const = 0;

    /// What a swapchain for window may be on adapters()[adapter]. Fails with
    /// RhiSurfaceFailed as Device::createSwapchain describes, or when a query
    /// fails.
    [[nodiscard]] virtual core::Result<SurfaceCaps> surfaceCaps(std::size_t adapter,
                                                                const core::window::NativeWindow& window) const = 0;

    /// The formats and color spaces a swapchain for window may use on
    /// adapters()[adapter], those BFR has names for, in the API's order. Fails
    /// as surfaceCaps does.
    [[nodiscard]] virtual core::Result<std::vector<SurfaceFormat>> surfaceFormats(
        std::size_t adapter, const core::window::NativeWindow& window) const = 0;

protected:
    explicit Rhi(const RhiConfig& config) : config_(config) {}

    [[nodiscard]] const RhiConfig& config() const { return config_; }

private:
    // Every backend runs with these settings, so the interface holds them: the
    // one exception to interfaces holding no data.
    RhiConfig config_;
};

} // namespace bfr::rhi
