#pragma once

#include "metal/metal_util.h"

#include <Metal/MTL4Counters.hpp>

namespace bfr::rhi::metal {

/// A counter heap of timestamps, released when this goes.
struct MetalTimestampPool {
    Ref<MTL4::CounterHeap> heap;
};

} // namespace bfr::rhi::metal
