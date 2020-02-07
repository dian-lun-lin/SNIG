#pragma once

#include <iostream>
#include <experimental/filesystem>
#include <fstream>
#include <string>
#include <assert.h>

namespace std {
	namespace fs = experimental::filesystem;
}

namespace sparse_dnn {

// TODO: finish the implementation and include a unittest ...
// https://github.com/onqtam/doctest
inline std::string read_file_to_string(const std::fs::path& path) {
	assert(std::fs::exists(path));
	std::ifstream f{ path };
	const auto fsize = std::fs::file_size(path);
   std::string result;
	result.reserve(fsize);
	f.read(&result[0], fsize);
	return result;
}

inline void write_file_from_string(const std::fs::path& path, const std::string& s) {
	assert(std::fs::exists(path));
	std::ofstream f{ path };
	f.write(&s[0], std::fs::file_size(path));
}

}
