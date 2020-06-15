#pragma once
#include <SNIG/utility/reader.hpp>
#include <xtensor/xarray.hpp>
#include <taskflow/taskflow.hpp>
#include <vector>


namespace std {
  namespace fs = experimental::filesystem;
}

namespace snig {

template <typename T>
class XtSequential {

  static_assert(
    std::is_same<T, float>::value || std::is_same<T, double>::value,
    "data type must be either float or double"
  );

  private:

    std::vector<xt::xarray<T> > _weights;
    size_t _num_neurons;
    size_t _num_layers;
    size_t _batch_size;
    T _bias;

  public:

    XtSequential(
      const std::fs::path& weight_path,
      size_t num_neurons, 
      size_t num_layers,
      size_t batch_size,
      T bias
    );

    ~XtSequential();

    xt::xarray<T> infer(
      const std::fs::path& input_path,
      const std::fs::path& golden_path
    );
};

// ----------------------------------------------------------------------------
// Definition of XtSequential
// ----------------------------------------------------------------------------

template <typename T>
XtSequential<T>::XtSequential (
  const std::fs::path& weight_path,
  size_t num_neurons, 
  size_t num_layers,
  size_t batch_size,
  T bias
) : _num_neurons{num_neurons}, _num_layers{num_layers} , _batch_size{batch_size}, _bias{bias} {
  _weights = read_weight_xt<T>(weight_path, num_neurons, num_layers);
}

template <typename T>
XtSequential<T>::~XtSequential () {
}

template <typename T>
xt::xarray<T> XtSequential<T>::infer (
  const std::fs::path& input_path,
  const std::fs::path& golden_path
) {
  std::cerr << "dffdsd";
}


} //end of snig----------------------------------------------------------------
