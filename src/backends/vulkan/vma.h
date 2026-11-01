#pragma once

// VMA over volk's headers, which declare no Vulkan prototypes. volk must come
// first, so the order is kept by hand.

// clang-format off
#include <volk.h> // IWYU pragma: keep
#include <vk_mem_alloc.h> // IWYU pragma: export
// clang-format on
