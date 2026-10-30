#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <objc/message.h>
#include <objc/runtime.h>
#include <QuartzCore/QuartzCore.hpp>

// Small helpers over metal-cpp for the Metal backend. metal-cpp is built with
// manual reference counting: whatever new*, alloc or copy returns is owned by
// the caller, and what Metal hands out otherwise (nextDrawable, a device
// listing) is autoreleased, so those calls run inside an AutoreleasePool.

#include <bit>
#include <string>
#include <utility>

namespace bfr::rhi::metal {

/// Drains what Metal autoreleased while it lives.
class AutoreleasePool {
public:
    AutoreleasePool() : pool_(NS::AutoreleasePool::alloc()->init()) {}
    ~AutoreleasePool() { pool_->release(); }
    AutoreleasePool(const AutoreleasePool&) = delete;
    AutoreleasePool& operator=(const AutoreleasePool&) = delete;
    AutoreleasePool(AutoreleasePool&&) = delete;
    AutoreleasePool& operator=(AutoreleasePool&&) = delete;

private:
    NS::AutoreleasePool* pool_;
};

/// A reference to a metal-cpp object, released when the last Ref to it goes. A
/// Ref that holds nothing releases nothing.
template <class T>
class Ref {
public:
    Ref() = default;
    ~Ref() { reset(); }
    Ref(const Ref& other) : object_(other.object_) {
        if (object_ != nullptr) {
            object_->retain();
        }
    }
    Ref(Ref&& other) noexcept : object_(std::exchange(other.object_, nullptr)) {}
    Ref& operator=(const Ref& other) {
        if (this != &other) {
            Ref copy{other};
            std::swap(object_, copy.object_);
        }
        return *this;
    }
    Ref& operator=(Ref&& other) noexcept {
        std::swap(object_, other.object_);
        return *this;
    }

    /// Takes over the one reference the caller owns, such as new* returns.
    [[nodiscard]] static Ref adopt(T* object) {
        Ref ref;
        ref.object_ = object;
        return ref;
    }

    /// Shares object, which the caller does not own, by retaining it.
    [[nodiscard]] static Ref share(T* object) {
        if (object != nullptr) {
            object->retain();
        }
        return adopt(object);
    }

    void reset() {
        if (object_ != nullptr) {
            object_->release();
            object_ = nullptr;
        }
    }

    [[nodiscard]] T* get() const { return object_; }
    [[nodiscard]] T* operator->() const { return object_; }
    [[nodiscard]] explicit operator bool() const { return object_ != nullptr; }

private:
    T* object_ = nullptr;
};

/// A Ref that takes over the reference the caller owns.
template <class T>
[[nodiscard]] Ref<T> adopt(T* object) {
    return Ref<T>::adopt(object);
}

/// A Ref that shares object, which the caller does not own.
template <class T>
[[nodiscard]] Ref<T> share(T* object) {
    return Ref<T>::share(object);
}

/// text as a std::string; empty for a null string.
[[nodiscard]] inline std::string toString(const NS::String* text) {
    return text != nullptr ? std::string{text->utf8String()} : std::string{};
}

/// Calls the BOOL setter on layer, such as "setOpaque:". metal-cpp has no
/// wrapper for the CALayer properties BFR sets with this.
inline void setLayerFlag(CA::MetalLayer* layer, const char* setter, bool value) {
    using Send = void (*)(void*, SEL, BOOL);
    std::bit_cast<Send> (&objc_msgSend)(layer, sel_registerName(setter), value ? YES : NO);
}

} // namespace bfr::rhi::metal
