#pragma once

#include <filesystem>
#include <fstream>
#include <string_view>

// Writes text to a file named name in the temp directory and returns its path.
inline std::filesystem::path writeConfig(std::string_view name, std::string_view text) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream{path} << text;
    return path;
}
