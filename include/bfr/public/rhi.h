#pragma once

#include "caps.h"
#include "device.h"
#include "error.h"
#include "format.h"
#include "log.h"
#include "rhi_config.h"
#include "window.h"

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

struct RhiDesc {
    /// Where the Rhi, its devices and, when RhiConfig turns it on, validation
    /// send their messages.
    core::Logger logger;
};

/// The graphics API's entry point and the adapters it found.
class Rhi {
public:
    /// The backend for this build. Fails with RhiLoaderFailed when the API's
    /// loader won't load, giving the OS's reason, RhiValidationUnavailable when
    /// validation is asked for and the system can't provide it, RhiInstanceFailed
    /// when the API refuses the instance, and RhiNoAdapter when no adapter meets
    /// the floor, naming each and why. With validation on, a refused instance's
    /// error carries the messages the API sent meanwhile.
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

    /// The API's own object for adapters()[adapter], valid while this Rhi
    /// lives: a VkPhysicalDevice on Vulkan.
    [[nodiscard]] virtual core::Result<void*> native(std::size_t adapter) const = 0;

protected:
    explicit Rhi(const RhiConfig& config) : config_(config) {}

    [[nodiscard]] const RhiConfig& config() const { return config_; }

private:
    // Every backend runs with these settings, so the interface holds them: the
    // one exception to interfaces holding no data.
    RhiConfig config_;
};

} // namespace bfr::rhi
