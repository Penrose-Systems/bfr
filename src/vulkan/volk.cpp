// volk's loader, built with the window systems' platform defines so it loads
// their surface functions. vcpkg builds its volk library without them.
#define VOLK_IMPLEMENTATION
#include <volk.h> // IWYU pragma: keep
