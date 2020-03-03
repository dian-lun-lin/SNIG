#pragma once
#include <Eigen/Sparse>
#include <SparseDNN/utility/reader.hpp>
#include <SparseDNN/utility/matrix_operation.hpp>
#include <SparseDNN/utility/thread_pool.hpp>
#include <SparseDNN/utility/scoring.hpp>
#include <Eigen/Dense>

#include <vector>

#include<thread>

namespace std {
  namespace fs = experimental::filesystem;
}

namespace sparse_dnn {


template <typename T>
class CPUParallel {

  static_assert(
  std::is_same<T, float>::value||std::is_same<T, double>::value,
  "data type must be either float or double"
  );

  private:

    std::vector<Eigen::SparseMatrix<T> > _weights;
    const size_t _num_neurons_per_layer;
    const size_t _num_layers;
    const T _bias;

    Eigen::Matrix<int, Eigen::Dynamic, 1> _data_parallel_task(
        Eigen::SparseMatrix<T> y
    ) const;

  public:

    CPUParallel(
      const std::fs::path& wieght_path,
      const T bias,
      const size_t num_neurons_per_layer=1024,
      const size_t num_layers=120
      );

    ~CPUParallel();

    size_t num_neurons_per_layer() const { return _num_neurons_per_layer; };
    size_t num_layers() const { return _num_layers; };
    T bias() const { return _bias; };

    Eigen::Matrix<int, Eigen::Dynamic, 1> infer(
      const std::fs::path& input_path,
      const size_t num_inputs
      ) const;
};

// ----------------------------------------------------------------------------
// Definition of CPUParallel
// ----------------------------------------------------------------------------

template<typename T>
CPUParallel<T>::CPUParallel(
  const std::fs::path& weight_path,
  const T bias,
  const size_t num_neurons_per_layer,
  const size_t num_layers
):
  _bias{bias},
  _num_neurons_per_layer{num_neurons_per_layer},
  _num_layers{num_layers}
{
  std::cout << "Constructing a CPU parallel network.\n";

  std::cout << "Loading the weight.............." << std::flush;
  _weights = read_weight<T>(weight_path, num_neurons_per_layer, num_layers);
  std::cout << "Done\n";
}

template<typename T>
CPUParallel<T>::~CPUParallel(){

}

template<typename T>
Eigen::Matrix<int, Eigen::Dynamic, 1> CPUParallel<T>::infer(
  const std::fs::path& input_path,
  const size_t num_inputs
) const {

  std::cout << "Reading input.............................." << std::flush;
  auto input = read_input<T>(input_path, num_inputs, _num_neurons_per_layer);
  std::cout << "Done\n";

  std::cout << "Start inference............................" << std::flush;
  Eigen::SparseVector<T> result;

  size_t num_tasks = 128;
  size_t num_threads = std::thread::hardware_concurrency();
  ThreadPool pool(num_threads);

  auto slicing_inputs = slice_by_row<T>(input, num_tasks);

  std::vector<std::future<Eigen::Matrix<int, Eigen::Dynamic, 1> > > futures;
  futures.reserve(num_tasks + 1);

  for(const auto& each_input:slicing_inputs){
    futures.push_back(pool.enqueue(
        [this, each_input] () {
           return this->_data_parallel_task(each_input);
        }
      )
    );
  }
  for(auto& f:futures){
    f.wait();
  }

  std::vector<Eigen::Matrix<int, Eigen::Dynamic, 1> > get_results;
  get_results.reserve(futures.size());
  for(auto& f:futures){
    get_results.push_back(f.get());
  }

  return concatenate_by_row(get_results);
}

template<typename T>
Eigen::Matrix<int, Eigen::Dynamic, 1> CPUParallel<T>::_data_parallel_task(
    Eigen::SparseMatrix<T> y
) const {

  Eigen::SparseMatrix<T> z;
  for(const auto& w:_weights){
    z = (y * w).pruned();
    z.coeffs() += _bias;
    y = z.unaryExpr([] (T a) {
     if(a < 0) return T(0);
     else if(a > 32) return T(32);
     return a;
     });
  }

  return get_score<T>(y);
}

}//end of namespace sparse_dnn ----------------------------------------------

