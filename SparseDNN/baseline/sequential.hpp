#pragma once

#include <iostream>
#include <SparseDNN/util/file.hpp>
#include <Eigen/Sparse>

namespace std {
	namespace fs = experimental::filesystem;
}
namespace sparse_dnn {
template <typename T>
class Sequential {

  // T is the floating point type, either float or double
  static_assert(
    std::is_same<T, float>::value || std::is_same<T, double>::value,
    "data type must be either float or double"
  );

  private:
	 const std::fs::path& _weight_path;
	 const size_t _num_inputs;
    const size_t _num_neurons;
    const size_t _num_layers;
    const T _bias;

	 Eigen::SparseMatrix<T> read_weight_or_input const  (const std::fs::path& path, bool is_input);
	 Eigen::SpaseVector<T> read_golden const (const std::fs::path& path);

  public:

    Sequential(
      const std::fs::path& weight_path,
		const size_t num_inputs,
      const size_t num_neurons,
      const size_t num_layers,
      T bias
    );

    ~Sequential();

    size_t num_neurons() const;
    size_t num_layers() const;
    T bias() const;

    void infer() const;
	 void check_passed(const Eigen::SparseMatrix& output_mat, const std::fs::path& golden) const;
};

// ----------------------------------------------------------------------------
// Definition of Sequential
// ----------------------------------------------------------------------------

template <typename T>
Sequential<T>::Sequential(
  const std::fs::path& weight_path,
  const size_t num_inputs,
  const size_t num_neurons,
  const size_t num_layers,
  T bias
) :
  _weight_path {weight_path},
  _num_inputs {num_inputs},
  _num_neurons {num_neurons},
  _num_layers {num_layers},
  _bias {bias}
{
  std::cout << "Constructing a sequential baseline.\n";
}

template <typename T>
Sequential<T>::~Sequential() {
}

template <typename T>
inline Sequential<T>:: Eigen::SparseMatrix<T> read_weight_or_input const  (const std::fs::path& path, bool is_input) {
		std::string line;
		std::vector<T> tripletList;
		Eigen::SParseMatrix<T> mat;
		if(is_input){
			mat.conservativeResize(_num_inputs, _num_neurons);
			tripletList.reserve((_num_inputs*_num_neurons)/200);
		}
		else{
			mat.conservativeResize(_num_neurons, _num_neurons);
			tripletList.reserve((_num_neurons^2)/200);
		}
		std::istringstream read_s(read_file_to_string(path));
		while(std::getline(read_s, line)){
			std::istringstream lineStream(line);
			std::string token;
			std::vector<T> tokens;
			while(std::getline(lineStream, token, '\t'))	tokens.push_back(std::stof(token));
			tripletList.push_back(T(tokens[0] - 1, tokens[1] - 1, tokens[2]));
		}
		mat.reserve(tripletList.size());
		mat.setFromTriplets(tripletList.begin(), tripletList.end());
		mat.makeCompressed();
		return mat;
}

template <typename T>
inline Sequential<T>:: Eigen::SpaseVector<T> read_golden(const std::fs::path& path) const {
	std::string line;
	std::istringstream read_s(read_file_to_string(path));
	Eigen::SparseVector<T> v(_num_inputs);
	v.reserve(_num_inputs / 200);
	while(std::getline(f, line)) v.insert(std::stoi(line) - 1) = 1;
	return v;
}


template <typename T>
inline Sequential::void infer(
      const std::fs::path& input, 
      const std::fs::path& golden)
     const {
   std::cout << "Loading the weight and input..............\n";
	Eigen::SparseMatrix<T> input_mat(_num_inputs, _num_neurons);
	Eigen::SparseMatrix<T> weight_mat(_num_neurons, _num_neurons);
	read_weight_and_input(input, input_mat, weight_mat);
	Eigen::SparseMatrix<T> Y(input_mat);
	Eigen::SparseMatrix<T> Z(Y.rows(), Y.cols());
	Z.reserve(Y.nonZeros());
	std::cout << "Start inference............................\n";
	for(size_t i = 0; i < W.size(); ++i){
		Z = (Y * W[i]).pruned();
		Z.coeffs() += _bias;
		Y = Z.unaryExpr([] (T a) {
				if(a < 0) return 0.0f;
				else if(a > YMAX) return YMAX;
				return a;
				});
	}
	Y.makeCompressed();
	std::cout << "Done.\n";
	check_passed(Y, golden);
}
template <typename T>
inline Sequential::void check_passed(const Eigen::SparseMatrix<T>& output_mat, const std::fs::path& golden) const {
	std::cout << "Checking correctness........................\n";
	Eigen::SpaseVector<T> golden_mat(_num_inputs);
	golden_mat = read_golden(golden);
	Eigen::SparseVector<T> output_mat = Eigen::MatrixXf(scores).rowwise().sum().sparseView();
	output_mat = output_mat.unaryExpr([] (T a) {
			if(a > 0) return 1;
			else return 0;
			});
	Eigen::SparseVector<T> diff_mat = gloden_mat - output_mat;
	diff_mat = diff_mat.pruned();
	if(diff_mat.nonZeros())
		std::cout << "Challenge FAILED\n";
	else
		std::cout << "Challenge PASSED\n" ;
}

}
}  // end of namespace sparse_dnn ---------------------------------------------



