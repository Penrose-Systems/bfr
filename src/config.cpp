#include "config.h"

#include "error.h"
#include "log.h"

#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>

#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>

namespace bfr::config {

struct Config::Document {
    nlohmann::json root;
    // Where root sits in the file, for warnings; empty for the whole file.
    std::string path;
};

Config::Config(std::unique_ptr<const Document> document) : document_(std::move(document)) {}

Config::~Config() = default;
Config::Config(Config&&) noexcept = default;
Config& Config::operator=(Config&&) noexcept = default;

core::Result<Config> Config::load(const std::filesystem::path& path) {
    std::ifstream file{path};
    if (!file) {
        // The stream keeps no reason of its own; errno has the OS's.
        return core::failure(core::ErrorCode::ConfigLoadFailed,
                             std::format("{}: {}", path.string(), std::generic_category().message(errno)));
    }
    try {
        return Config{std::make_unique<const Document>(nlohmann::json::parse(file))};
    } catch (const nlohmann::json::parse_error& error) {
        return core::failure(core::ErrorCode::ConfigLoadFailed, std::format("{}: {}", path.string(), error.what()));
    }
}

Config Config::section(std::string_view name, std::uint32_t version) const {
    const auto& root = document_->root;
    if (!root.contains(name)) {
        return Config{std::make_unique<const Document>()};
    }
    const auto& section = root.at(name);
    // A drifted section is still read; the warning only says it drifted.
    if (!section.contains("version")) {
        BFR_LOG_WARN("Config section {} has no version; the code reads version {}", name, version);
    }
    else if (section.at("version") != version) {
        BFR_LOG_WARN("Config section {} is version {}; the code reads version {}", name, section.at("version").dump(),
                     version);
    }
    return Config{std::make_unique<const Document>(section, std::string{name})};
}

template <class T>
T Config::value(std::string_view key, T fallback) const {
    const auto& root = document_->root;
    if (!root.contains(key)) {
        return fallback;
    }
    const auto& found = root.at(key);
    // The JSON library would wrap a negative or too large number into range
    // and truncate a fraction.
    if constexpr (std::is_unsigned_v<T> && !std::is_same_v<T, bool>) {
        if (!found.is_number_unsigned() || found.template get<std::uint64_t>() > std::numeric_limits<T>::max()) {
            BFR_LOG_WARN("Config value {}.{}: {} is not a whole number from 0 to {}; using the default",
                         document_->path, key, found.dump(), std::numeric_limits<T>::max());
            return fallback;
        }
    }
    try {
        return found.template get<T>();
    } catch (const nlohmann::json::type_error& error) {
        BFR_LOG_WARN("Config value {}.{}: {}; using the default", document_->path, key, error.what());
        return fallback;
    }
}

template bool Config::value<bool>(std::string_view key, bool fallback) const;
template std::uint32_t Config::value<std::uint32_t>(std::string_view key, std::uint32_t fallback) const;

} // namespace bfr::config
