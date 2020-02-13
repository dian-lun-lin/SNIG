#pragma once

#include <iostream>
#include <experimental/filesystem>
#include <fstream>
#include <Eigen/Sparse>
#include <vector>
#include <string>

namespace std {
	namespace fs = experimental::filesystem;
}

namespace sparse_dnn {
    
//issue
//T is the floating posize_t type, either float or double
//static_assert(
    //std::is_same<T, float>::value || std::is_same<T, double>::value,
    //"data type must be either float or double"
    //);

// C++17 if constexpr - compile-time switch
// SFINAE
template <typename T>
std::enable_if_t<std::is_same<T, float>::value, float> 
to_numeric(const std::string& str) {
  return std::stof(str);
}

template <typename T>
std::enable_if_t<std::is_same<T, double>::value, double> 
to_numeric(const std::string& str) {
  return std::stod(str);
}

template <typename T>
Eigen::SparseMatrix<T> tsv_string_to_matrix(
  const std::string& s,
  const size_t rows,
  const size_t cols
);

inline
std::string read_file_to_string(const std::fs::path& path);

inline
void write_file_from_string(
    const std::fs::path& path,
    const std::string& s
);
    
template <typename T>
std::vector<Eigen::SparseMatrix<T> > read_weight(
    const std::fs::path& weight_path,
    const size_t num_neurons_per_layer,
    const size_t num_layers=1024
);

template <typename T>
Eigen::SparseMatrix<T> read_input(
    const std::fs::path& input_path,
    const size_t num_inputs,
    const size_t num_features
);

template <typename T>
Eigen::SparseVector<T> read_golden(
    const std::fs::path& golden_path,
    const size_t num_inputs
);


template <typename T>
bool is_passed(
    const Eigen::SparseMatrix<T>& output,
    const Eigen::SparseMatrix<T>& golden
);

//-----------------------------------------------------------------------------
//Definition of reader function
//-----------------------------------------------------------------------------

template <typename T>
Eigen::SparseMatrix<T> tsv_string_to_matrix(
    const std::string& s,
    const size_t rows,
    const size_t cols
) {
  //T is the floating posize_t type, either float or double
  static_assert(
      std::is_same<T, float>::value || std::is_same<T, double>::value,
      "data type must be either float or double"
      );

  typedef Eigen::Triplet<T> E;
  std::string line;
  std::vector<E> triplet_list;
  triplet_list.reserve(rows/200);
  std::istringstream read_s(s);

  std::vector<T> numerics;

  while(std::getline(read_s, line)){
    std::istringstream lineStream(line);
    std::string token;
    numerics.clear();
    while(std::getline(lineStream, token, '\t')) {
       numerics.push_back(to_numeric<T>(token));
    }
    triplet_list.push_back(E(numerics[0] - 1,numerics[1] - 1, numerics[2]));
  }

  Eigen::SparseMatrix<T> mat(rows, cols);
  mat.reserve(triplet_list.size());
  mat.setFromTriplets(triplet_list.begin(), triplet_list.end());
  mat.makeCompressed();
  return mat;
}


template <typename T>
std::vector<Eigen::SparseMatrix<T> > read_weight(
    const std::fs::path& weight_path,
    const size_t num_neurons_per_layer,
    const size_t num_layers
) {

  //T is the floating posize_t type, either float or double
  static_assert(
      std::is_same<T, float>::value || std::is_same<T, double>::value,
      "data type must be either float or double"
      );

  std::vector<Eigen::SparseMatrix<T> > mats;
  mats.reserve(num_layers);

  //read weights
  for(size_t i = 0; i < num_layers; ++i){
    std::fs::path p = weight_path;
    p /= "n" + std::to_string(num_neurons_per_layer) + "-l"
      + std::to_string(i + 1) + ".tsv";
    std::string data_str = read_file_to_string(p);
    mats.push_back(tsv_string_to_matrix<T>(
          data_str,
          num_neurons_per_layer,
          num_neurons_per_layer)
    );
  }
  return mats;
}

template<typename T>
Eigen::SparseMatrix<T> read_input(
    const std::fs::path& input_path,
    const size_t num_inputs,
    const size_t num_features
) {

  //T is the floating posize_t type, either float or double
  static_assert(
      std::is_same<T, float>::value || std::is_same<T, double>::value,
      "data type must be either float or double"
      );

  std::string input_str = read_file_to_string(input_path);
  return tsv_string_to_matrix<T>(input_str, num_inputs, num_features);
}

template <typename T>
Eigen::SparseVector<T> read_golden(
    const std::fs::path& golden_path,
    const size_t num_inputs
) {

  //T is the floating posize_t type, either float or double
  static_assert(
      std::is_same<T, float>::value || std::is_same<T, double>::value,
      "data type must be either float or double"
      );

  std::string line;
  std::istringstream read_s(read_file_to_string(golden_path));
  Eigen::SparseVector<T> mat(num_inputs, 1);
  mat.reserve(num_inputs / 200);
  while(std::getline(read_s, line)) {
    mat.insert(std::stoi(line) - 1) = 1;
  }   
  return mat;
}

inline
std::string read_file_to_string(const std::fs::path& path) {
  
  using namespace std::literals::string_literals;

	std::ifstream f{ path };
  if(!f){
    throw std::runtime_error("cannot open the file"s + path.c_str());
  }

	const auto fsize = std::fs::file_size(path);
  std::string result(fsize, ' ');
  f.read(&result[0], fsize);
	return result;
}

inline
void write_file_from_string(
    const std::fs::path& path,
    const std::string& s
) {
  using namespace std::literals::string_literals;

	std::ofstream f{ path };
  if(!f){
    throw std::runtime_error("cannot open the file"s + path.c_str());
  }
	f.write(&s[0], std::fs::file_size(path));
}

template <typename T>
bool is_passed(
    const Eigen::SparseMatrix<T>& output,
    const Eigen::SparseMatrix<T>& golden
){
  Eigen::SparseMatrix<T> diff_mat = golden - output;
  diff_mat = diff_mat.pruned();
  if(!diff_mat.nonZeros())
    return true;
  else
    return false;
}

} // end of namespace sparse_dnn-----------------------------------------------
