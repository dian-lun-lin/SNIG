#pragma once

#include <iostream>
#include <experimental/filesystem>

namespace std {
  namespace filesystem = experimental::filesystem;
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

    const size_t _num_neurons;
    const size_t _num_layers;
    const T _bias;

    // TODO:
    // _read_

    std::string _read_file_to_string();

  public:

    Sequential(
      const std::filesystem::path& path,
      size_t num_neurons,
      size_t num_layers,
      T bias
    );

    ~Sequential();

    size_t num_neurons() const;
    size_t num_layers() const;
    T bias() const;

    float infer(
      const std::filesystem::path& input, 
      const std::filesystem::path& golden
    ) const;

};

// ----------------------------------------------------------------------------
// Definition of Sequential
// ----------------------------------------------------------------------------

template <typename T>
Sequential<T>::Sequential(
  const std::filesystem::path& path,
  size_t num_neurons,
  size_t num_layers,
  T bias
) :
  _num_neurons {num_neurons},
  _num_layers {num_layers},
  _bias {bias}
{
  // TODO
  std::cout << "constructing a sequential baseline\n";
}

template <typename T>
Sequential<T>::~Sequential() {
  // TODO
}

}  // end of namespace sparse_dnn ---------------------------------------------



