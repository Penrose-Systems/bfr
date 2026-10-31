#include "bfr/rhi/instance.h"

#include "bfr/core/error.h"
#include "bfr/rhi/rhi_config.h"
#if BFR_HAS_D3D12
    #include "d3d12/d3d12_instance.h"
#endif
#if BFR_HAS_METAL
    #include "metal/metal_instance.h"
#endif
#if BFR_HAS_VULKAN
    #include "vulkan/vulkan_instance.h"
#endif

#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <memory>
#include <span>

namespace bfr::rhi {

namespace {

// Each backend this build has, in Backend order. The only list of backends
// outside the build files: every other file speaks to one backend or to the
// interface.
constexpr std::size_t kBackendCount = BFR_HAS_VULKAN + BFR_HAS_D3D12 + BFR_HAS_METAL;
constexpr std::array<Backend, kBackendCount> kBackends{
#if BFR_HAS_VULKAN
    Backend::Vulkan,
#endif
#if BFR_HAS_D3D12
    Backend::D3D12,
#endif
#if BFR_HAS_METAL
    Backend::Metal,
#endif
};

} // namespace

std::span<const Backend> backends() noexcept {
    return kBackends;
}

core::Result<std::unique_ptr<Instance>> Instance::create(const InstanceDesc& desc, const RhiConfig& config) {
    switch (desc.backend) {
#if BFR_HAS_VULKAN
        case Backend::Vulkan:
            return vulkan::VulkanInstance::create(desc, config);
#else
        case Backend::Vulkan:
#endif
#if BFR_HAS_D3D12
        case Backend::D3D12:
            return d3d12::D3D12Instance::create(desc, config);
#else
        case Backend::D3D12:
#endif
#if BFR_HAS_METAL
        case Backend::Metal:
            return metal::MetalInstance::create(desc, config);
#else
        case Backend::Metal:
#endif
            return core::failure(core::ErrorCode::RhiBackendUnavailable,
                                 std::format("backend {} is not in this build", name(desc.backend)));
    }
    return core::failure(core::ErrorCode::RhiBackendUnavailable,
                         std::format("backend {} is not one BFR has", static_cast<std::uint32_t>(desc.backend)));
}

} // namespace bfr::rhi
