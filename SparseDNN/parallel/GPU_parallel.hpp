#include<Eigen/Sparse>
#include<SparseDNN/utility/reader.hpp>
#include<SparseDNN/utility/matrix_operation.hpp>
#include<SparseDNN/parallel/task.hpp>
#include<numeric>
#include<algorithm>

namespace std{
  namespace fs = experimental::filesystem;
}

namespace sparse_dnn {
template <typename T>
class GPUParallel{
    
  static_assert(
  std::is_same<T, float>::value || std::is_same<T, double>::value,
  "data type must be either float or double"
  );
  
  private:

    std::vector<CSCMatrix<T> > _weights;
    size_t* _num_neurons_per_layer;
    size_t* _num_layers;
    T* _bias;
    
  public:

    GPUParallel(
      const std::fs::path& weight_path,
      const T bias,
      const size_t num_neurons_per_layer=1024,
      const size_t num_layers=120 
    );

    ~GPUParallel();

    size_t num_neurons_per_layer() const { return _num_neurons_per_layer; };
    size_t num_layers() const { return *_num_layers; };
    T bias() const { return _bias; };

    Eigen::SparseVector<T> infer(
      const std::fs::path& input_path,
      const size_t num_inputs
    ) const;


};

// ----------------------------------------------------------------------------
// Definition of GPUParallel
// ----------------------------------------------------------------------------

template<typename T>
GPUParallel<T>::GPUParallel(
  const std::fs::path& weight_path,
  const T bias,
  const size_t num_neurons_per_layer,
  const size_t num_layers
)
{
  std::cout << "Constructing a GPU parallel network.\n";
  std::cout << "Loading the weight.............." << std::flush;
  cudaMallocManaged(&_bias, sizeof(T));
  cudaMallocManaged(&_num_neurons_per_layer, sizeof(size_t));
  cudaMallocManaged(&_num_layers, sizeof(size_t));
  *_num_neurons_per_layer = num_neurons_per_layer;
  *_num_layers = num_layers;
  *_bias = bias;
  _weights.reserve(num_layers);
  for(size_t i = 0; i < num_layers; ++i){
    std::fs::path p = weight_path;
    p /= "n" + std::to_string(num_neurons_per_layer) + "-l"
      + std::to_string(i + 1) + ".tsv";
    std::string data_str = read_file_to_string(p);
    size_t nnz = std::count(data_str.begin(), data_str.end(), '\n');

//improve:only alloc once
    CSCMatrix<T> weight;
    cudaMallocManaged(
      &weight.col_array,
      sizeof(size_t) * (num_neurons_per_layer + 1)
    );
    cudaMallocManaged(&weight.row_array, sizeof(size_t) * nnz);
    cudaMallocManaged(&weight.data_array, sizeof(T) * nnz);
    tsv_string_to_CSC_matrix<T>(data_str, num_neurons_per_layer, weight);

    _weights.push_back(weight);
  }
  std::cout << "Done\n";
}

template<typename T>
GPUParallel<T>::~GPUParallel(){

  for(auto& w:_weights){
    cudaFree(w.col_array);
    cudaFree(w.row_array);
    cudaFree(w.data_array);
  }
  cudaFree(_bias);
  cudaFree(_num_neurons_per_layer);
  cudaFree(_num_layers);
}

template<typename T>
Eigen::SparseVector<T> GPUParallel<T>::infer(
  const std::fs::path& input_path,
  const size_t num_inputs
) const {

  int block_size = 512;
  int num_blocks = (num_inputs + block_size - 1) / block_size;

  std::cout << "Reading input.............................." << std::flush;
  std::string data_str = read_file_to_string(input_path);
  size_t nnz = std::count(data_str.begin(), data_str.end(), '\n');
  CSRMatrix<T> y;
  size_t* num_inputs_ptr;
  cudaMallocManaged(&y.row_array, sizeof(size_t) * (num_inputs + 1));
  cudaMallocManaged(&y.col_array, sizeof(size_t) * nnz);
  cudaMallocManaged(&y.data_array, sizeof(T) * nnz);

  cudaMallocManaged(&num_inputs_ptr, sizeof(size_t));
  *num_inputs_ptr = num_inputs;

  tsv_string_to_CSR_matrix<T>(data_str, num_inputs, *_num_neurons_per_layer, y);
  std::cout << "Done\n";

  std::cout << "Start inference............................" << std::flush;
  CSRMatrix<T> z;
  cudaMallocManaged(&z.row_array, sizeof(size_t) * (num_inputs + 1));

  for(const auto& w : _weights){
    check_nnz<T><<<num_blocks, block_size>>>(
      num_inputs_ptr,
      y.row_array,
      y.col_array,
      y.data_array,
      _num_neurons_per_layer,
      w.col_array,
      w.row_array,
      w.data_array,
      z.row_array,
      _bias
    );
    cudaDeviceSynchronize();
    std::partial_sum(z.row_array, z.row_array + num_inputs + 1, z.row_array);

    nnz = z.row_array[num_inputs];

    cudaMallocManaged(&z.col_array, sizeof(size_t) * nnz);
    cudaMallocManaged(&z.data_array, sizeof(T) * nnz);

    CSR_mutiply_CSC<<<num_blocks, block_size>>>(
      num_inputs_ptr,
      y.row_array,
      y.col_array,
      y.data_array,
      _num_neurons_per_layer,
      w.col_array,
      w.row_array,
      w.data_array,
      z.row_array,
      z.col_array,
      z.data_array,
      _bias
    );

    cudaDeviceSynchronize();
    cudaFree(y.col_array);
    cudaFree(y.data_array);
    cudaMallocManaged(&y.col_array, sizeof(size_t) * nnz);
    cudaMallocManaged(&y.data_array, sizeof(T) * nnz);

    for(size_t i = 0; i < num_inputs + 1; ++i){
      y.row_array[i] = z.row_array[i];
    }

    for(size_t i = 0; i < nnz; ++i){
      y.col_array[i] = z.col_array[i];
      y.data_array[i] = z.data_array[i];
    }

    cudaFree(z.col_array);
    cudaFree(z.data_array);
  }
  
  auto tmp = CSR_matrix_to_eigen_sparse(y, num_inputs, *_num_neurons_per_layer);

  cudaFree(z.row_array);
  cudaFree(y.row_array);

  return get_score(tmp);
}



}// end of namespace sparse_dnn ----------------------------------------------
