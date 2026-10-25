#pragma once

#include <cstdlib>
#include <optional>
#include <string>
#include <utility>

// Sets an environment variable for one scope and restores the old value, or
// its absence, when the scope ends.
class ScopedEnv {
public:
    ScopedEnv(std::string name, const std::string& value) : name_(std::move(name)) {
        if (const char* previous = std::getenv(name_.c_str()); previous != nullptr) {
            previous_ = previous;
        }
        constexpr int kOverwrite = 1;
        setenv(name_.c_str(), value.c_str(), kOverwrite);
    }

    ~ScopedEnv() {
        if (previous_) {
            constexpr int kOverwrite = 1;
            setenv(name_.c_str(), previous_->c_str(), kOverwrite);
        }
        else {
            unsetenv(name_.c_str());
        }
    }

    ScopedEnv(const ScopedEnv&) = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;
    ScopedEnv(ScopedEnv&&) = delete;
    ScopedEnv& operator=(ScopedEnv&&) = delete;

private:
    std::string name_;
    std::optional<std::string> previous_;
};
