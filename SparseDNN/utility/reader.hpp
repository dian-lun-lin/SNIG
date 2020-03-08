#pragma once

#include <iostream>
#include <experimental/filesystem>
#include <fstream>
#include <Eigen/Sparse>
#include <vector>
#include <string>
#include <SparseDNN/utility/matrix_format.h>
#include <numeric>
#include <SparseDNN/utility/matrix_operation.hpp>

namespace std {
	namespace fs = experimental::filesystem;
}

namespace sparse_dnn {
    
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
  const size_t cols,
  const size_t nnz
);

template <typename T>
void tsv_string_to_CSC_matrix(
    const std::string& s,
    const size_t cols,
    CSCMatrix<T>& mat
);

template <typename T>
void tsv_string_to_CSR_matrix(
    const std::string& s,
    const size_t rows,
    const size_t nnz,
    CSRMatrix<T>& mat
);

template <typename T>
void tsv_string_to_2D_array(
    const std::string& s,
    const size_t cols,
    T* arr
);

template <typename T>
void tsv_string_to_CSR_packed_array(
    const std::string& s,
    const size_t rows,
    const size_t cols,
    const size_t nnz,
    const int COL_BLK,
    const int N_SLAB,
    int* arr
);

inline
size_t count_nnz(const std::string& s);

inline
std::string read_file_to_string(const std::fs::path& path);

inline
void write_file_from_string(
    const std::fs::path& path,
    const std::string& s
);
    
template <typename T>
void read_weight(
    const std::string& s,
    const size_t num_neurons_per_layer,
    const size_t nnz,
    CSRMatrix<T>& mat
);

template <typename T>
void read_weight(
    const size_t num_neurons_per_layer,
    const size_t nnz_per_layer,
    const size_t num_layers,
    const int COL_BLK,
    const int N_SLAB,
    int* arr
);

template <typename T>
std::vector<Eigen::SparseMatrix<T> > read_weight(
    const std::fs::path& weight_path,
    const size_t num_neurons_per_layer,
    const size_t num_layers
);

template <typename T>
Eigen::SparseMatrix<T> read_input(
    const std::fs::path& input_path,
    const size_t num_inputs,
    const size_t num_features
);

template <typename T>
Eigen::SparseMatrix<T> read_input(
    const std::string& s,
    const size_t num_inputs,
    const size_t num_features,
    const size_t nnz,
    CSRMatrix<T>& mat
);

template<typename T>
void read_input(
    const std::fs::path& input_path,
    const size_t num_inputs,
    const size_t num_neurons_per_layer,
    T* arr,
    int* non_empty_rows,
    int& num_non_empty_rows
);

template <typename T>
Eigen::SparseVector<T> read_golden(
    const std::fs::path& golden_path,
    const size_t num_inputs
);



//-----------------------------------------------------------------------------
//Definition of reader function
//-----------------------------------------------------------------------------

template <typename T>
Eigen::SparseMatrix<T> tsv_string_to_matrix(
    const std::string& s,
    const size_t rows,
    const size_t cols,
    const size_t nnz
) {
  //T is the floating posize_t type, either float or double
  static_assert(
      std::is_same<T, float>::value || std::is_same<T, double>::value,
      "data type must be either float or double"
      );

  typedef Eigen::Triplet<T> E;
  std::string line;
  std::vector<E> triplet_list;
  triplet_list.reserve(nnz);
  std::istringstream read_s(s);
  std::vector<std::string> tokens;

  while(std::getline(read_s, line)){
    std::istringstream lineStream(line);
    std::string token;
    tokens.clear();
    while(std::getline(lineStream, token, '\t')) {
       tokens.push_back(token);
    }
    triplet_list.push_back(E(
      std::stoi(tokens[0]) - 1,
      std::stoi(tokens[1]) - 1,
      to_numeric<T>(tokens[2])
    ));
  }

  Eigen::SparseMatrix<T> mat(rows, cols);
  mat.reserve(triplet_list.size());
  mat.setFromTriplets(triplet_list.begin(), triplet_list.end());
  return mat;
}

//issue:doesnt't check boundary
template <typename T>
void tsv_string_to_2D_array(
    const std::string& s,
    const size_t cols,
    T* arr
){
  //T is the floating posize_t type, either float or double
  static_assert(
      std::is_same<T, float>::value || std::is_same<T, double>::value,
      "data type must be either float or double"
      );

  std::string line;
  std::istringstream read_s(s);

  std::vector<std::string> tokens;

  while(std::getline(read_s, line)){
    std::istringstream lineStream(line);
    std::string token;
    tokens.clear();
    while(std::getline(lineStream, token, '\t')) {
       tokens.push_back(token);
    }
    arr[(std::stoi(tokens[0]) - 1) * cols + std::stoi(tokens[1]) - 1] = to_numeric<T>(tokens[2]);
  }

}

template <typename T>
void tsv_string_to_CSR_matrix(
    const std::string& s,
    const size_t rows,
    const size_t cols,
    const size_t nnz,
    CSRMatrix<T>& mat
) {
  //T is the floating posize_t type, either float or double
  static_assert(
      std::is_same<T, float>::value || std::is_same<T, double>::value,
      "data type must be either float or double"
      );
  Eigen::SparseMatrix<T, Eigen::RowMajor> eigen_mat = tsv_string_to_matrix<T>(s, rows, cols, nnz);

  std::copy(eigen_mat.outerIndexPtr(), eigen_mat.outerIndexPtr() + rows + 1, mat.row_array);
  std::copy(eigen_mat.innerIndexPtr(), eigen_mat.innerIndexPtr() + nnz, mat.col_array);
  std::copy(eigen_mat.valuePtr(), eigen_mat.valuePtr() + nnz, mat.data_array);
}

template <typename T>
void tsv_string_to_CSR_packed_array(
    const std::string& s,
    const size_t rows,
    const size_t cols,
    const size_t nnz,
    const int COL_BLK,
    const int N_SLAB,
    int* arr
) {
  //T is the floating posize_t type, either float or double
  static_assert(
      std::is_same<T, float>::value || std::is_same<T, double>::value,
      "data type must be either float or double"
      );
  typedef Eigen::Triplet<T> E;
  std::string line;
  std::vector<E> triplet_list;
  triplet_list.reserve(nnz);
  std::istringstream read_s(s);
  std::vector<std::string> tokens;

  while(std::getline(read_s, line)){
    std::istringstream lineStream(line);
    std::string token;
    tokens.clear();
    while(std::getline(lineStream, token, '\t')) {
       tokens.push_back(token);
    }
    triplet_list.push_back(E(
      std::stoi(tokens[0]) - 1 + 
      rows * ((std::stoi(tokens[1]) - 1) / COL_BLK),
      std::stoi(tokens[1]) - 1,
      to_numeric<T>(tokens[2])
    ));
  }

  Eigen::SparseMatrix<T> eigen_mat(rows * N_SLAB, cols);
  eigen_mat.reserve(triplet_list.size());
  eigen_mat.setFromTriplets(triplet_list.begin(), triplet_list.end());

  std::copy(eigen_mat.outerIndexPtr(), eigen_mat.outerIndexPtr() + rows * N_SLAB + 1, arr);
  std::copy(eigen_mat.innerIndexPtr(), eigen_mat.innerIndexPtr() + nnz, arr + rows * N_SLAB + 1);
  T* tmp = reinterpret_cast<T*>(arr + rows * N_SLAB + 1 + nnz);
  for(int i = 0; i < nnz; ++i){
    tmp[i] = *(eigen_mat.valuePtr() + i);
  }
}

inline
size_t count_nnz(const std::string& s){
  return std::count(s.begin(), s.end(), '\n');
}

template <typename T>
void tsv_string_to_CSC_matrix(
    const std::string& s,
    const size_t cols,
    CSCMatrix<T>& mat
) {

  //T is the floating posize_t type, either float or double
  static_assert(
      std::is_same<T, float>::value || std::is_same<T, double>::value,
      "data type must be either float or double"
      );
  
  std::string line;
  std::istringstream read_s(s);

  std::vector<std::string> tokens;

  size_t tmp = 0;
  size_t count_per_col = 0;
  size_t count_nnz = 0;
  while(std::getline(read_s, line)){
    std::istringstream lineStream(line);
    std::string token;
    tokens.clear();
    while(std::getline(lineStream, token, '\t')) {
       tokens.push_back(token);
    }
    ++mat.col_array[std::stoi(tokens[1])];
    mat.row_array[count_nnz] = std::stoi(tokens[0]) - 1;
    mat.data_array[count_nnz++] = to_numeric<T>(tokens[2]);
  }
  std::partial_sum(mat.col_array, mat.col_array + cols + 1, mat.col_array);
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
    mats.push_back(
      tsv_string_to_matrix<T>(
          data_str,
          num_neurons_per_layer,
          num_neurons_per_layer,
          count_nnz(data_str)
      )
    );
  }
  return mats;
}
template <typename T>
void read_weight(
    const std::string& s,
    const size_t num_neurons_per_layer,
    const size_t nnz,
    CSRMatrix<T>& mat
) {
 tsv_string_to_CSR_matrix(s, num_neurons_per_layer, num_neurons_per_layer, nnz, mat);
}

template <typename T>
void read_weight(
    const std::fs::path& weight_path,
    const size_t num_neurons_per_layer,
    const size_t nnz_per_layer,
    const size_t num_layers,
    const int COL_BLK,
    const int N_SLAB,
    int* arr
) {
  for(int i = 0; i < num_layers; ++i){
    std::fs::path p = weight_path;
    p /= "n" + std::to_string(num_neurons_per_layer) + "-l"
      + std::to_string(i + 1) + ".tsv";
    std::string data_str = read_file_to_string(p);
//issue: double right?
    tsv_string_to_CSR_packed_array<T>(data_str, num_neurons_per_layer, num_neurons_per_layer, nnz_per_layer, COL_BLK, N_SLAB, arr + i * (num_neurons_per_layer * N_SLAB + 1 + nnz_per_layer + ((sizeof(T) / sizeof(int)) * nnz_per_layer)));
  }
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
  return tsv_string_to_matrix<T>(input_str, num_inputs, num_features, count_nnz(input_str));
}

template<typename T>
void read_input(
    const std::fs::path& input_path,
    const size_t num_inputs,
    const size_t num_neurons_per_layer,
    T* arr,
    int* non_empty_rows,
    int& num_non_empty_rows
) {
  std::string input_str = read_file_to_string(input_path);
  tsv_string_to_2D_array<T>(input_str, num_neurons_per_layer, arr);
  int counter = 0;
  T* it;
  for(int i = 0; i < num_inputs; ++i){
    it = std::find_if(arr + i * num_neurons_per_layer, arr + (i + 1) * num_neurons_per_layer, [](T v){ return v != 0;});
    if(it != (arr + (i + 1) * num_neurons_per_layer)){
      non_empty_rows[counter++] = i;
    }
  }
  num_non_empty_rows = counter;
}

template <typename T>
Eigen::SparseMatrix<T> read_input(
    const std::string& s,
    const size_t num_inputs,
    const size_t num_features,
    const size_t nnz,
    CSRMatrix<T>& mat
) {
  tsv_string_to_CSR_matrix<T>(s, num_inputs, num_features, nnz, mat);
}

inline
Eigen::Matrix<int, Eigen::Dynamic, 1> read_golden(
    const std::fs::path& golden_path,
    const size_t num_inputs
) {

  std::string line;
  std::istringstream read_s(read_file_to_string(golden_path));
  Eigen::Matrix<int, Eigen::Dynamic, 1> golden(num_inputs, 1);
  while(std::getline(read_s, line)) {
    golden(std::stoi(line) - 1, 0) = 1;
  }   
  return golden;
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


} // end of namespace sparse_dnn-----------------------------------------------
