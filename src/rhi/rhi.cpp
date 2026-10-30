#include "bfr/rhi/rhi.h"

#include "bfr/core/error.h"
#include "bfr/rhi/rhi_config.h"
#if BFR_HAS_METAL
    #include "metal/metal_rhi.h"
#endif
#if BFR_HAS_VULKAN
    #include "vulkan/vulkan_rhi.h"
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
constexpr std::size_t kBackendCount = BFR_HAS_VULKAN + BFR_HAS_METAL;
constexpr std::array<Backend, kBackendCount> kBackends{
#if BFR_HAS_VULKAN
    Backend::Vulkan,
#endif
#if BFR_HAS_METAL
    Backend::Metal,
#endif
};

} // namespace

std::span<const Backend> backends() noexcept {
    return kBackends;
}

core::Result<std::unique_ptr<Rhi>> Rhi::create(const RhiDesc& desc, const RhiConfig& config) {
    switch (desc.backend) {
#if BFR_HAS_VULKAN
        case Backend::Vulkan:
            return vulkan::VulkanRhi::create(desc, config);
#else
        case Backend::Vulkan:
#endif
#if BFR_HAS_METAL
        case Backend::Metal:
            return metal::MetalRhi::create(desc, config);
#else
        case Backend::Metal:
#endif
        case Backend::D3D12:
            return core::failure(core::ErrorCode::RhiBackendUnavailable,
                                 std::format("backend {} is not in this build", name(desc.backend)));
    }
    return core::failure(core::ErrorCode::RhiBackendUnavailable,
                         std::format("backend {} is not one BFR has", static_cast<std::uint32_t>(desc.backend)));
}

} // namespace bfr::rhi
