#pragma once
#include <iostream>
#include <Eigen/SparseCore>
#include <Eigen/Dense>
#include <SparseDNN/utility/reader.hpp>
#include <SparseDNN/utility/matrix_operation.hpp>
#include <SparseDNN/utility/scoring.hpp>
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
    const int _num_neurons_per_layer;
    const int _num_layers;
    const T _bias;

  public:
    
    Sequential(
      const std::fs::path& weight_path,
      const T bias,
      const int num_neurons_per_layer=1024,
      const int num_layers=120
    );

    ~Sequential();

    int num_neurons_per_layer() const;
    int num_layers() const;

    Eigen::Matrix<int, Eigen::Dynamic, 1> infer(
        const std::fs::path& input_path,
        const int num_input
    ) const;
};

// ----------------------------------------------------------------------------
// Definition of Sequential
// ----------------------------------------------------------------------------

template <typename T>
Sequential<T>::Sequential(
  const std::fs::path& weight_path,
  const T bias,
  const int num_neurons_per_layer,
  const int num_layers
):
  _bias {bias},
  _num_neurons_per_layer {num_neurons_per_layer},
  _num_layers {num_layers}
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
int Sequential<T>::num_neurons_per_layer() const { 
  return _num_neurons_per_layer; 
}

template <typename T>
int Sequential<T>::num_layers() const { 
  return _num_layers; 
}

template <typename T>
Eigen::Matrix<int, Eigen::Dynamic, 1> Sequential<T>::infer(
  const std::fs::path& input_path,
  const int num_inputs
) const {

  std::cout << "Reading input.............................." << std::flush;
  auto y = read_input<T>(input_path, num_inputs, _num_neurons_per_layer);
  std::cout << "Done" << std::endl;

  std::cout << "Start inference............................" << std::flush;

  for(const auto& w : _weights) {
    Eigen::SparseMatrix<T> z(num_inputs, _num_neurons_per_layer);
    z = (y * w).pruned();
    z.coeffs() += _bias;
    y = z.unaryExpr([] (T a) {
	   if(a < 0) return T(0);
	   else if(a > 32) return T(32);
	   return a;
	   });
  }
  std::cout << "Done\n";

  return get_score<T>(y);
}
}  // end of namespace sparse_dnn ----------------------------------------------

