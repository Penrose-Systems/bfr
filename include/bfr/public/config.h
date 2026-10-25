#pragma once

#include "error.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string_view>

namespace bfr::config {

/// Settings read from a JSON file. Each module reads its own top-level section,
/// and every value it asks for comes with the default to use when the file
/// doesn't set it.
class Config {
public:
    /// Fails with ConfigLoadFailed when the file can't be read, naming the file
    /// and the reason.
    [[nodiscard]] static core::Result<Config> load(const std::filesystem::path& path);

    /// The top-level section called name, copied out of this config. version is
    /// the layout the caller reads; a section the file gives another version is
    /// still read, and the difference is logged as a warning.
    [[nodiscard]] Config section(std::string_view name, std::uint32_t version) const;

    /// The value at key, converted to T, or fallback.
    template <class T>
    [[nodiscard]] T value(std::string_view key, T fallback) const;

    ~Config();
    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;
    Config(Config&&) noexcept;
    Config& operator=(Config&&) noexcept;

private:
    struct Document;

    explicit Config(std::unique_ptr<const Document> document);

    std::unique_ptr<const Document> document_;
};

extern template bool Config::value<bool>(std::string_view key, bool fallback) const;
extern template std::uint32_t Config::value<std::uint32_t>(std::string_view key, std::uint32_t fallback) const;

} // namespace bfr::config
