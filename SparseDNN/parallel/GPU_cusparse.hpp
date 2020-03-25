#pragma once
#include <Eigen/SparseCore>
#include <Eigen/Dense>
#include <SparseDNN/utility/reader.hpp>
#include <SparseDNN/utility/matrix_operation.hpp>
#include <SparseDNN/utility/scoring.hpp>
#include <SparseDNN/parallel/task.hpp>

namespace std{
  namespace fs = experimental::filesystem;
}

namespace sparse_dnn {
template <typename T>
class GPUCusparse{
    
  static_assert(
    std::is_same<T, float>::value || std::is_same<T, double>::value,
    "data type must be either float or double"
  );
  
  private:

    std::vector<CSRMatrix<T> > _weights;
    int _num_neurons_per_layer;
    int _num_layers;
    T _bias;
    
  public:

    GPUCusparse(
      const std::fs::path& weight_path,
      const T bias,
      const int num_neurons_per_layer=1024,
      const int num_layers=120 
    );

    ~GPUCusparse();

    int num_neurons_per_layer() const;
    int num_layers() const;

    Eigen::Matrix<int, Eigen::Dynamic, 1> infer(
      const std::fs::path& input_path,
      const int num_inputs
    ) const;

};

// ----------------------------------------------------------------------------
// Definition of GPUCusparse
// ----------------------------------------------------------------------------

template<typename T>
GPUCusparse<T>::GPUCusparse(
  const std::fs::path& weight_path,
  const T bias,
  const int num_neurons_per_layer,
  const int num_layers
):
  _bias{bias},
  _num_neurons_per_layer{num_neurons_per_layer},
  _num_layers{num_layers}
{

  std::cout << "Constructing a GPU parallel network.\n";

  std::cout << "Loading the weight.............." << std::flush;
  _weights.reserve(num_layers);
  CSRMatrix<T> weight;
  for(int i = 0; i < num_layers; ++i) {
    std::fs::path p = weight_path;
    p /= "n" + std::to_string(num_neurons_per_layer) + "-l"
      + std::to_string(i + 1) + ".tsv";
    std::string data_str = read_file_to_string(p);
    int nnz = count_nnz(data_str);
    weight.row_array = new int[num_neurons_per_layer + 1];
    weight.col_array = new int[nnz];
    weight.data_array = new T[nnz];
    read_weight<T>(data_str, num_neurons_per_layer, nnz, weight);
    _weights.push_back(weight);
  }

  std::cout << "Done\n";
}

template<typename T>
GPUCusparse<T>::~GPUCusparse() {
  for(auto& w:_weights) {
    delete[] w.row_array;
    delete[] w.col_array;
    delete[] w.data_array;
  }
}

template<typename T>
int GPUCusparse<T>::num_neurons_per_layer() const {
   return _num_neurons_per_layer; 
}

template<typename T>
int GPUCusparse<T>::num_layers() const {
   return _num_layers; 
}

template<typename T>
Eigen::Matrix<int, Eigen::Dynamic, 1> GPUCusparse<T>::infer(
  const std::fs::path& input_path,
  const int num_inputs
) const {

  int num_threads_per_block = 1024;
  int num_blocks_per_grid = 512;

  std::cout << "Reading input.............................." << std::flush;
  std::string data_str = read_file_to_string(input_path);
  int y_nnz = count_nnz(data_str);
  CSRMatrix<T> y;
  y.row_array = new int[num_inputs + 1];
  y.col_array = new int[y_nnz];
  y.data_array = new T[y_nnz];
  read_input<T>(data_str, num_inputs, _num_neurons_per_layer, y_nnz, y);
  std::cout << "Done\n";

  std::cout << "Start inference............................" << std::flush;
  CSRMatrix<T> d_y;
  cudaMalloc(&d_y.row_array, sizeof(int) * (num_inputs + 1));

  CSRMatrix<T> d_z;
  cudaMalloc(&d_z.row_array, sizeof(int) * (num_inputs + 1));

  CSRMatrix<T> d_w;
  cudaMalloc(&d_w.row_array, sizeof(int) * (_num_neurons_per_layer + 1));

  for(const auto& w : _weights) {

    y_nnz = y.row_array[num_inputs];
    cudaMalloc(&d_y.col_array, sizeof(int) * y_nnz);
    cudaMalloc(&d_y.data_array, sizeof(T) * y_nnz);

    cudaMemcpy(d_y.row_array, y.row_array, sizeof(int) * (num_inputs + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y.col_array, y.col_array, sizeof(int) * y_nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y.data_array, y.data_array, sizeof(T) * y_nnz, cudaMemcpyHostToDevice);

    int w_nnz{w.row_array[_num_neurons_per_layer]};
    cudaMalloc(&d_w.col_array, sizeof(int) * (w_nnz));
    cudaMalloc(&d_w.data_array, sizeof(T) * (w_nnz));

    cudaMemcpy(d_w.row_array, w.row_array, sizeof(int) * (_num_neurons_per_layer + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w.col_array, w.col_array, sizeof(int) * (w_nnz), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w.data_array, w.data_array, sizeof(T) * (w_nnz), cudaMemcpyHostToDevice);
    
    cusparse_mutiplication(d_y, d_w, num_inputs, _num_neurons_per_layer, _num_neurons_per_layer, y_nnz, w_nnz, d_z);

    cudaMemcpy(y.row_array, d_z.row_array, sizeof(int) * (num_inputs + 1), cudaMemcpyDeviceToHost);
    delete [] y.col_array;
    delete [] y.data_array;
    y.col_array = new int[y.row_array[num_inputs]];
    y.data_array = new T[y.row_array[num_inputs]];

    cudaMemcpy(y.col_array, d_z.col_array, sizeof(int) * y.row_array[num_inputs], cudaMemcpyDeviceToHost);
    cudaMemcpy(y.data_array, d_z.data_array, sizeof(T) * y.row_array[num_inputs], cudaMemcpyDeviceToHost);

    add_bias_relu_CPU<T>(y.data_array, _bias, y.row_array[num_inputs]);

    resize_CPU<T>(y, num_inputs + 1);

    cudaFree(d_y.col_array);
    cudaFree(d_y.data_array);
    cudaFree(d_z.col_array);
    cudaFree(d_z.data_array);
    cudaFree(d_w.col_array);
    cudaFree(d_w.data_array);
  }
  

  cudaFree(d_z.row_array);
  cudaFree(d_y.row_array);
  cudaFree(d_w.row_array);


  auto score = get_score(y, num_inputs);

  delete [] y.row_array;
  delete [] y.col_array;
  delete [] y.data_array;

  return score;


}

}// end of namespace sparse_dnn ----------------------------------------------
