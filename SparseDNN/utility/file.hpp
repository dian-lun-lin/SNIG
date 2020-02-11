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
  
  using namespace std::literals::string_literals;

	std::ifstream f{ path };
  if(!f){
    throw std::runtime_error("cannot open the file"s + path.c_str());
  }

	const auto fsize = std::fs::file_size(path);
  std::string result;
	result.reserve(fsize);
	f.read(&result[0], fsize);
	return result;
}

inline void write_file_from_string(const std::fs::path& path, const std::string& s) {
  // TODO
  using namespace std::literals::string_literals;

	std::ofstream f{ path };
  if(!f){
    throw std::runtime_error("cannot open the file"s + path.c_str());
  }
	f.write(&s[0], std::fs::file_size(path));
}

}
