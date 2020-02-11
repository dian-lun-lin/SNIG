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
template <typename T>
class Reader {
	
	// T is the floating posize_t type, either float or double
	static_assert(
			std::is_same<T, float>::value || std::is_same<T, double>::value,
			"data type must be either float or double"
			);

	private:

		const std::fs::path _weight_path;
		const std::fs::path _input_path;
		const std::fs::path _golden_path;
    Eigen::SparseMatrix<T> _tsv_string_to_matrix(
      const std::string& s,
      const size_t rows,
      const size_t cols
    ) const;
    std::string _read_file_to_string(const std::fs::path& path) const;
    void _write_file_from_string(
        const std::fs::path& path,
        const std::string& s
    ) const;
	
	public:

    Reader(
      const std::fs::path& weight_path,
      const std::fs::path& input_path,
      const std::fs::path& golden_path
    );

    std::vector<Eigen::SparseMatrix<T> > read_weight_and_input(
        size_t num_inputs,
        size_t num_neurons,
        size_t num_layers=1024
    ) const;
    Eigen::SparseMatrix<T> read_golden(size_t num_inputs) const;

};

//-----------------------------------------------------------------------------
//Definition of Reader
//-----------------------------------------------------------------------------

template <typename T>
Reader<T>::Reader(
  const std::fs::path& weight_path,
  const std::fs::path& input_path,
  const std::fs::path& golden_path
):
  _weight_path(weight_path),
  _input_path(input_path),
  _golden_path(golden_path)
{
  
}

template <typename T>
Eigen::SparseMatrix<T> Reader<T>::
_tsv_string_to_matrix(
    const std::string& s,
    const size_t rows,
    const size_t cols
) const {

  typedef Eigen::Triplet<T> E;
  std::string line;
  std::vector<E> triplet_list;
  triplet_list.reserve(rows/200);
  std::istringstream read_s(s);

  while(std::getline(read_s, line)){
    std::istringstream lineStream(line);
    std::string token;
    std::vector<T> tokens;
    //issue
    while(std::getline(lineStream, token, '\t'))  tokens.push_back(std::stof(token));
    triplet_list.push_back(E(tokens[0] - 1,tokens[1] - 1, tokens[2]));
  }
  //issue
  Eigen::SparseMatrix<T> mat(rows, cols);
  mat.reserve(triplet_list.size());
  mat.setFromTriplets(triplet_list.begin(), triplet_list.end());
  mat.makeCompressed();
  return mat;
}

template <typename T>
Eigen::SparseMatrix<T> Reader<T>::
read_golden(size_t num_inputs) const {

  std::string line;
  std::istringstream read_s(_read_file_to_string(_golden_path));
  Eigen::SparseMatrix<T> mat(num_inputs, 1);
  mat.reserve(num_inputs / 200);
  while(std::getline(read_s, line)) mat.insert(std::stoi(line) - 1, 1) = 1;
  return mat;
}

template <typename T>
std::vector<Eigen::SparseMatrix<T> > Reader<T>::
read_weight_and_input(
    size_t num_inputs, 
    size_t num_neurons,
    size_t num_layers
) const {

  std::vector<Eigen::SparseMatrix<T> > mats;
  mats.reserve(num_layers + 1);
  std::cout << "Loading the weight and input..............";

  //read input
  std::string input_str = _read_file_to_string(_input_path);
  mats.push_back(_tsv_string_to_matrix(input_str, num_inputs, num_neurons));

  //read weights
  for(size_t i = 0; i < num_layers; ++i){
    std::fs::path p = _weight_path;
    p /= "n" + std::to_string(num_neurons) + "-l"
      + std::to_string(i + 1) + ".tsv";
    std::string data_str = _read_file_to_string(p);
    mats.push_back(_tsv_string_to_matrix(data_str, num_neurons, num_neurons));
  }
  std::cout << "Done\n";
  return mats;
}

template<typename T>
std::string Reader<T>:: _read_file_to_string(const std::fs::path& path) const {
  
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

template<typename T>
void Reader<T>::_write_file_from_string(
    const std::fs::path& path,
    const std::string& s
) const {
  using namespace std::literals::string_literals;

	std::ofstream f{ path };
  if(!f){
    throw std::runtime_error("cannot open the file"s + path.c_str());
  }
	f.write(&s[0], std::fs::file_size(path));
}

} // end of namespace sparse_dnn-----------------------------------------------
