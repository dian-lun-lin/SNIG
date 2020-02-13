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
    
    //const issue
    std::vector<Eigen::SparseMatrix<T> > _weights;
    const size_t _num_neurons_per_layer;
    const size_t _num_layers;
    const T _bias;

  public:
    
    Sequential(
      const std::fs::path& weight_path,
      T bias,
      const size_t num_neurons_per_layer=1024,
      const size_t num_layers=120
    );

    ~Sequential();


    size_t num_neurons_per_layer() const { return _num_neurons_per_layer; };
    size_t num_layers() const { return _num_layers; };
    T bias() const { return _bias; };

    Eigen::SparseMatrix<T> infer(
        const std::fs::path& input_path,
        const size_t num_input
    ) const;
};

// ----------------------------------------------------------------------------
// Definition of Sequential
// ----------------------------------------------------------------------------

template <typename T>
Sequential<T>::Sequential(
  const std::fs::path& weight_path,
  T bias,
  const size_t num_neurons_per_layer,
  const size_t num_layers
):
  _bias (bias),
  _num_neurons_per_layer (num_neurons_per_layer),
  _num_layers (num_layers)
{
  std::cout << "Constructing a sequential baseline.\n";

  std::cout << "Loading the weight..............";
  _weights = read_weight<T>(weight_path, num_neurons_per_layer, num_layers);
  std::cout << "Done\n";
}

template <typename T>
Sequential<T>::~Sequential() {
}

template <typename T>
Eigen::SparseMatrix<T> Sequential<T>::infer(
  const std::fs::path& input_path,
  const size_t num_inputs
) const {

  std::cout << "Reading input..............................";
  auto Y = read_input<T>(input_path, num_inputs, _num_neurons_per_layer);
  std::cout << "Done" << std::endl;

  std::cout << "Start inference............................";
  Eigen::SparseMatrix<T> Z(num_inputs, _num_neurons_per_layer);
  Z.reserve(num_inputs*_num_neurons_per_layer/200);
  for(auto w : _weights){
    Z = (Y * w).pruned();
    Z.coeffs() += _bias;
    Y = Z.unaryExpr([] (T a) {
	   if(a < 0) return T(0);
	   else if(a > 32) return T(32);
	   return a;
	   });
  }
  std::cout << "Done\n";

  //cout issue
  std::cout << "Start scoring..............................";
  //Eigen::SparseMatrix<T> score(num_inputs, 1);
  //score.reserve(num_inputs/2000);
  Eigen::SparseVector<T> score = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
    (Y).rowwise().sum().sparseView();
  score = score.unaryExpr([] (T a) {
    if(a > 0) return 1;
    else return 0;
  });
  std::cout << "Done\n";
  return score;
}
}  // end of namespace sparse_dnn ----------------------------------------------



