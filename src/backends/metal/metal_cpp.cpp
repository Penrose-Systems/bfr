// metal-cpp, Apple's header-only C++ wrapper of Metal, wants exactly one
// source to define its implementation. It is built without BFR's warnings.
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
