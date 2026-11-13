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

/// The native graphics API an Instance runs on, and what BFR needs of it. The
/// values are bfr_backend's, so that zero, which a default-initialised
/// InstanceDesc holds, names no backend.
enum class Backend : std::uint8_t {
    /// Vulkan 1.2 or later with dynamic rendering and synchronization2
    /// (extensions before 1.3), descriptor indexing with update-after-bind and
    /// partially bound tables, and timelines.
    Vulkan = 1,
    D3D12 = 2,
    Metal = 3,
};

[[nodiscard]] constexpr std::string_view name(Backend backend) noexcept {
    switch (backend) {
        case Backend::Vulkan:
            return "vulkan";
        case Backend::D3D12:
            return "d3d12";
        case Backend::Metal:
            return "metal";
    }
    return "unknown";
}

/// The backends this build has, in Backend order. Instance::create accepts these
/// and refuses every other.
[[nodiscard]] std::span<const Backend> backends() noexcept;

struct InstanceDesc {
    /// The API the Instance runs on. The caller names one; there is no default.
    Backend backend{};

    /// The API version the caller uses, in AdapterInfo's terms (1, 2 is
    /// Vulkan 1.2), passed to the backend as it is. Below what BFR needs,
    /// which the backend's Backend value states, create fails with
    /// RhiUnsupported.
    std::uint32_t apiMajor = 0;
    std::uint32_t apiMinor = 0;

    /// Where the Instance, its devices and, when RhiConfig turns it on, validation
    /// send their messages. Nowhere, until the caller says.
    core::Logger logger = core::Logger::none();

    /// The application and the engine it runs on, passed to the backend as
    /// given where it takes them; read during create only. An empty name is
    /// none.
    std::string_view applicationName{};
    std::uint32_t applicationVersion = 0;
    std::string_view engineName{};
    std::uint32_t engineVersion = 0;

    /// Time the CPU spends in each call of the Instance and its devices, which
    /// Instance::callTimings and Device::callTimings give. Off, it costs
    /// nothing; on, each call reads the clock twice.
    bool callTimings = false;
};

/// The graphics API's entry point and the adapters it found.
class Instance {
public:
    /// An Instance on desc.backend. Fails with RhiBackendUnavailable when
    /// desc.backend is not in backends(), RhiLoaderFailed when the API's loader
    /// won't load, giving the OS's reason, RhiValidationUnavailable when
    /// validation is asked for and the system can't provide it,
    /// RhiInstanceFailed when the API refuses the instance, RhiUnsupported
    /// when desc's version is below BFR's minimum, RhiInvalidCall when the
    /// backend has no form for it, and RhiNoAdapter when no adapter meets
    /// BFR's minimum, naming each and why. With validation
    /// on, a refused instance's error carries the messages the API sent
    /// meanwhile.
    [[nodiscard]] static core::Result<std::unique_ptr<Instance>> create(const InstanceDesc& desc,
                                                                        const RhiConfig& config);

    virtual ~Instance() = default;
    Instance(const Instance&) = delete;
    Instance& operator=(const Instance&) = delete;
    Instance(Instance&&) = delete;
    Instance& operator=(Instance&&) = delete;

    [[nodiscard]] virtual std::span<const Adapter> adapters() const = 0;

    /// A device on adapters()[adapter] with the queues desc asks for,
    /// destroyed before this Instance. With a window in presentTo, a backend
    /// that picks queue families picks ones that present to it where it can.
    /// Fails with
    /// RhiAdapterOutOfRange when adapter is not an index into adapters(),
    /// RhiInvalidCall when a maximum in desc is more than a handle names or a
    /// queue's level is not one BFR has, RhiWindowFailed as
    /// Device::createSwapchain describes, RhiUnsupported when the backend
    /// can't set the ranks or levels asked for, and RhiDeviceFailed when the
    /// API refuses the device.
    [[nodiscard]] virtual core::Result<std::unique_ptr<Device>> createDevice(
        std::size_t adapter, const core::window::NativeWindow& presentTo, const DeviceDesc& desc) = 0;

    // The queries below fail with RhiAdapterOutOfRange when adapter is not an
    // index into adapters(). The API answers each afresh.

    /// What adapters()[adapter] does with format. Fails with the API's error
    /// when the query fails for a reason other than the format being
    /// unsupported.
    [[nodiscard]] virtual core::Result<FormatCaps> formatCaps(std::size_t adapter, Format format) const = 0;

    /// What a swapchain for window may be on adapters()[adapter]. Fails with
    /// RhiWindowFailed as Device::createSwapchain describes, or when a query
    /// fails.
    [[nodiscard]] virtual core::Result<SwapchainCaps> swapchainCaps(std::size_t adapter,
                                                                    const core::window::NativeWindow& window) const = 0;

    /// The formats and color spaces a swapchain for window may use on
    /// adapters()[adapter], those BFR has names for, in the API's order. Fails
    /// as swapchainCaps does.
    [[nodiscard]] virtual core::Result<std::vector<SwapchainFormat>> swapchainFormats(
        std::size_t adapter, const core::window::NativeWindow& window) const = 0;

    /// The CPU time spent in each of the Instance's own calls since it was
    /// made, written to timings by Call, as many as it holds; its devices
    /// count theirs. Fails with RhiInvalidCall when InstanceDesc::callTimings
    /// was not set.
    [[nodiscard]] virtual core::Result<void> callTimings(std::span<CallTiming> timings) const;

protected:
    explicit Instance(const RhiConfig& config) : config_(config) {}

    [[nodiscard]] const RhiConfig& config() const { return config_; }

private:
    // Every backend runs with these settings, so the interface holds them: the
    // one exception to interfaces holding no data.
    RhiConfig config_;
};

} // namespace bfr::rhi
