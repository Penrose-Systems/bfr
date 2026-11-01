// VMA's implementation, compiled once, without BFR's warnings. It fetches every
// Vulkan function itself from the two getters each allocator is given, since
// each device calls through its own table, not volk's global pointers.
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_IMPLEMENTATION
#include "vulkan/vma.h" // IWYU pragma: keep
