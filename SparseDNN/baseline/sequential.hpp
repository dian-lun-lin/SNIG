#pragma once

#include <iostream>
#include <Eigen/Sparse>
#include <SparseDNN/utility/reader.hpp>
#include <vector>

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

    const Reader<T> _reader;
    const size_t _num_inputs;
    const size_t _num_neurons;
    const size_t _num_layers;
    const T _bias;

    void _check_passed(const Eigen::SparseMatrix<T>& output_mat) const;

  public:
    
    Sequential(
      const std::fs::path& weight_path,
      const std::fs::path& input_path,
      const std::fs::path& golden_path,
		  const size_t num_inputs,
      const size_t num_neurons,
      const size_t num_layers,
      T bias
    );

    ~Sequential();

    size_t num_neurons() const { return _num_neurons; };
    size_t num_layers() const { return _num_layers; };
    T bias() const { return _bias; };

    void infer() const;
};

// ----------------------------------------------------------------------------
// Definition of Sequential
// ----------------------------------------------------------------------------

template <typename T>
Sequential<T>::Sequential(
  const std::fs::path& weight_path,
  const std::fs::path& input_path,
  const std::fs::path& golden_path,
  const size_t num_inputs,
  const size_t num_neurons,
  const size_t num_layers,
  T bias
):
  _reader(weight_path, input_path, golden_path),
  _num_inputs (num_inputs),
  _num_neurons (num_neurons),
  _num_layers (num_layers),
  _bias (bias)
{
  std::cout << "Constructing a sequential baseline.\n";
}

template <typename T>
Sequential<T>::~Sequential() {
}

template <typename T>
void Sequential<T>::infer() const {

  auto mats = _reader.read_weight_and_input(_num_inputs, _num_neurons, _num_layers);

  std::cout << "Start inference............................";
  Eigen::SparseMatrix<T> Y(mats[0]);
  for(auto w = mats.begin() + 1; w < mats.end(); ++w){
    Eigen::SparseMatrix<T> Z = (Y * (*w)).pruned();
    Z.coeffs() += _bias;
    Y = Z.unaryExpr([] (T a) {
	   if(a < 0) return T(0);
	   else if(a > 32) return T(32);
	   return a;
	   });
  }
  Y.makeCompressed();
  std::cout << "Done\n";

  _check_passed(Y);
}

template <typename T>
void Sequential<T>::_check_passed(const Eigen::SparseMatrix<T>& output_mat) const {

  std::cout << "Checking correctness........................\n";
  auto golden_mat = _reader.read_golden(_num_inputs);
  Eigen::SparseMatrix<T> final_output_mat(_num_neurons, 1);
  final_output_mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> \
	  (output_mat).rowwise().sum().sparseView();
  final_output_mat = final_output_mat.unaryExpr([] (T a) {
    if(a > 0) return T(1);
    else return T(0);
    });

  Eigen::SparseMatrix<T> diff_mat = golden_mat - final_output_mat;
  diff_mat = diff_mat.pruned();
  if(diff_mat.nonZeros())
    std::cout << "Challenge FAILED\n";
  else
    std::cout << "Challenge PASSED\n" ;
}

}  // end of namespace sparse_dnn ----------------------------------------------



