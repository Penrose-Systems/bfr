#include "rhi_config.h"

#include "config.h"

namespace bfr::rhi {

void RhiConfig::set(const config::Config& config) {
    const auto rhi = config.section("rhi", kVersion);
    validation = rhi.value("validation", validation);
    swapchainImages = rhi.value("swapchainImages", swapchainImages);
}

} // namespace bfr::rhi
