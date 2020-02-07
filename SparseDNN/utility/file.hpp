#pragma once

#include <iostream>
#include <experimental/filesystem>
#include <fstream>
#include <string>

namespace std {
	namespace fs = experimental::filesystem;
}

namespace sparse_dnn {

// TODO: finish the implementation and include a unittest ...
// https://github.com/onqtam/doctest
inline std::string read_file_to_string(std::fs::path& path) {
	static_assert(std::fs::exists(path));
	std::ifstream f{ path };
	const auto fsize = std::fs::file_size(path);
   std::string result(fsize, ' ');
	f.read(result.data(), fsize);
	return result;
}

inline void write_file_from_string(std::fs::path& path, const std::string& s) {
	static_assert(std::fs::exist(path));
	std::ofstream f{ path };
	f.write(s, fs::file_size(path));
}

}
