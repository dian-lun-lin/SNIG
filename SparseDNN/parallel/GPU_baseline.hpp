#pragma once
#include <Eigen/Dense>
#include <SparseDNN/utility/matrix_format.h>
#include <SparseDNN/utility/cuda_error.hpp>
#include <SparseDNN/utility/scoring.hpp>
#include <SparseDNN/utility/utility.hpp>
#include <SparseDNN/parallel/task.hpp>
#include <chrono>

namespace std {
  namespace fs = experimental::filesystem;  
}

namespace sparse_dnn{

template <typename T>  
class GPUBaseline {

  static_assert(
    std::is_same<T, float>::value || std::is_same<T, double>::value,
    "data type must be either float or double"
  );
  
  private:
    int* _h_pinned_weight;
    T _bias;
    size_t _num_neurons_per_layer;
    size_t _num_layers;

    size_t _max_nnz_per_layer;
    size_t _COL_BLK;
    size_t _pad {0};
    size_t _N_SLAB;

    size_t _p_w_index_len;
    size_t _pp_w_index_len;
    size_t _pp_wlen;
    size_t _pp_wsize;

    void _non_empty_rows(
      const size_t num_inputs,
      int* rlenY,
      int* rowsY,
      size_t& nnz
    ) const;

  public:

    GPUBaseline(
      const std::fs::path& weight_path,
      const T bias = -.3f,
      const size_t num_neurons_per_layer = 1024,
      const size_t num_layers = 120
    );

    ~GPUBaseline();
    
    size_t num_neurons_per_layer() const;

    size_t num_layers() const;

    Eigen::Matrix<int, Eigen::Dynamic, 1> infer(
      const std::fs::path& input_path,
      const size_t num_inputs
    ) const;

};

// ----------------------------------------------------------------------------
// Definition of GPUBaseline
// ----------------------------------------------------------------------------

template <typename T>
GPUBaseline<T>::GPUBaseline(
  const std::fs::path& weight_path,
  const T bias,
  const size_t num_neurons_per_layer,
  const size_t num_layers
):
  _bias{bias},
  _num_neurons_per_layer{num_neurons_per_layer},
  _num_layers{num_layers},
  _pad{0}
{
  std::cout << "Constructing a GPU parallel network.\n";

  //get tuned shared memory size
  //num_neurons_per_layer must be divisible by shared memory (a.k.a. COL_BLK)
  //only for single GPU
  //only for double float
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  size_t max_num_per_block = props.sharedMemPerBlock / sizeof(T);
  if(num_neurons_per_layer <= max_num_per_block) {
    _COL_BLK = num_neurons_per_layer;
  }
  else{
    int max_divisor = 2;
    while((num_neurons_per_layer % max_divisor != 0) || (max_num_per_block < (num_neurons_per_layer / max_divisor))) {
      ++max_divisor;
    }
    _COL_BLK = num_neurons_per_layer / max_divisor;
  }


  std::cout << "Loading the weight........................." << std::flush;
  auto reading_beg = std::chrono::steady_clock::now();

  _N_SLAB = num_neurons_per_layer / _COL_BLK; 

  _max_nnz_per_layer = find_max_nnz_binary(
                         weight_path,
                         num_layers,
                         num_neurons_per_layer
                       );

  // total length of row and col index
  // value index should consider sizeof(T)
  _p_w_index_len  = num_neurons_per_layer * _N_SLAB + _max_nnz_per_layer + 1;

  //handle aligned (only deal with double and float)
  if(_p_w_index_len % sizeof(T) != 0){
    ++_pad;
  }

  _pp_w_index_len = _p_w_index_len + _pad;

  //pad packed weight length
  _pp_wlen = _pp_w_index_len + (sizeof(T) / sizeof(float)) * _max_nnz_per_layer;
  //pad packed weight size
  _pp_wsize = sizeof(int) * (_pp_w_index_len) + sizeof(T) * _max_nnz_per_layer;

  checkCuda(cudaMallocHost(
    (void**)&_h_pinned_weight,
    _pp_wsize * num_layers
  ));

  std::memset(
    _h_pinned_weight,
    0,
    _pp_wsize * num_layers
  );

  read_weight_binary<T>(
    weight_path,
    num_neurons_per_layer,
    _max_nnz_per_layer,
    num_layers,
    _N_SLAB,
    _pad,
    _h_pinned_weight
  );

  auto reading_end = std::chrono::steady_clock::now();
  std::cout << "finished reading DNN layers with " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(reading_end - reading_beg).count()
            << "ms"
            << '\n';
}

template <typename T>
GPUBaseline<T>:: ~GPUBaseline() {
  checkCuda(cudaFreeHost(_h_pinned_weight));
}

template <typename T>
size_t GPUBaseline<T>::num_neurons_per_layer() const {
 return _num_neurons_per_layer; 
}

template <typename T>
size_t GPUBaseline<T>::num_layers() const { 
  return _num_layers; 
}

template <typename T>
Eigen::Matrix<int, Eigen::Dynamic, 1> GPUBaseline<T>::infer(
const std::fs::path& input_path,
  const size_t num_inputs
) const {

  std::cout << "Preprocessing.............................." << std::flush;
  auto pp_beg = std::chrono::steady_clock::now();
  
  
  // d_W[0]: present layer 
  // d_W[1]: next layer
  int *d_W[2];
  checkCuda(cudaMalloc(
    &d_W[0],
    _pp_wsize
  ));
  checkCuda(cudaMalloc(
    &d_W[1],
    _pp_wsize
  ));
  checkCuda(cudaMemcpy(
    d_W[0],
    _h_pinned_weight,
    _pp_wsize,
    cudaMemcpyHostToDevice
  ));

  T* Y[2];  
  int *rowsY[2], *rlenY[2];

  checkCuda(cudaMallocManaged(&Y[0], sizeof(T) * num_inputs * _num_neurons_per_layer));
  checkCuda(cudaMallocManaged(&Y[1], sizeof(T) * num_inputs * _num_neurons_per_layer));
  checkCuda(cudaMallocManaged(&rowsY[0], sizeof(int) * num_inputs));
  checkCuda(cudaMallocManaged(&rowsY[1], sizeof(int) * num_inputs));
  checkCuda(cudaMallocManaged(&rlenY[0], sizeof(int) * num_inputs));
  checkCuda(cudaMallocManaged(&rlenY[1], sizeof(int) * num_inputs));
  checkCuda(cudaMemset(Y[0], 0, sizeof(T) * num_inputs * _num_neurons_per_layer));
  checkCuda(cudaMemset(Y[1], 0, sizeof(T) * num_inputs * _num_neurons_per_layer));
  checkCuda(cudaMemset(rowsY[0], 0, sizeof(int) * num_inputs));
  checkCuda(cudaMemset(rowsY[1], 0, sizeof(int) * num_inputs));
  checkCuda(cudaMemset(rlenY[0], 0, sizeof(int) * num_inputs));
  checkCuda(cudaMemset(rlenY[1], 0, sizeof(int) * num_inputs));
  checkCuda(cudaDeviceSynchronize());

  size_t nerowsY{0};
  read_input_binary<T>(input_path, Y[0], rlenY[0], rowsY[0], nerowsY);

  auto pp_end = std::chrono::steady_clock::now();
  
  std::cout << "finished preprocessing with " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(pp_end - pp_beg).count()
            << "ms"
            << std::endl;

  std::cout << "Start inference............................" << std::flush;

  auto exec_beg = std::chrono::steady_clock::now();

  cudaStream_t stream[2];

  checkCuda(cudaStreamCreate(&stream[0]));
  checkCuda(cudaStreamCreate(&stream[1]));

  dim3 threads_dim(32, 32, 1);

  for(size_t cur_layer = 0; cur_layer < _num_layers - 1; ++cur_layer) {
    
    checkCuda(cudaMemcpyAsync(
      d_W[(cur_layer + 1) % 2],
      _h_pinned_weight + (cur_layer + 1) * (_pp_wlen),
      _pp_wsize,
      cudaMemcpyHostToDevice,
      stream[0]
    ));

    baseline_inference<T><<<nerowsY, threads_dim, sizeof(T) * _COL_BLK, stream[1]>>>(
      Y[cur_layer % 2],
      nerowsY,
      rowsY[cur_layer % 2],
      rlenY[cur_layer % 2],
      _COL_BLK,
      _N_SLAB,
      _num_neurons_per_layer,
      d_W[cur_layer % 2],
      d_W[cur_layer % 2] + _num_neurons_per_layer * _N_SLAB + 1,
      (T*)(d_W[cur_layer % 2] + _p_w_index_len),
      _bias,
      Y[(cur_layer + 1) % 2],
      rlenY[(cur_layer + 1) % 2]
    );

    checkCuda(cudaStreamSynchronize(stream[1]));

    _non_empty_rows(
      num_inputs,
      rlenY[(cur_layer + 1) % 2],
      rowsY[(cur_layer + 1) % 2],
      nerowsY
    );

    checkCuda(cudaStreamSynchronize(stream[0]));

    checkCuda(cudaMemset(
      Y[cur_layer % 2],
      0,
      sizeof(T) * num_inputs * _num_neurons_per_layer)
    );
  }

  baseline_inference<T><<<nerowsY, threads_dim, sizeof(T) * _COL_BLK, stream[1]>>>(
    Y[(_num_layers - 1) % 2],
    nerowsY,
    rowsY[(_num_layers - 1) % 2],
    rlenY[(_num_layers - 1) % 2],
    _COL_BLK,
    _N_SLAB,
    _num_neurons_per_layer,
    d_W[(_num_layers - 1 ) % 2],
    d_W[(_num_layers - 1) % 2] + _num_neurons_per_layer * _N_SLAB + 1,
    (T*)(d_W[(_num_layers - 1) % 2] + _p_w_index_len),
    _bias,
    Y[(_num_layers) % 2],
    rlenY[(_num_layers) % 2]
  );
  checkCuda(cudaStreamSynchronize(stream[1]));

  auto exec_end = std::chrono::steady_clock::now();
  std::cout << "finished execution with " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(exec_end - exec_beg).count()
            << "ms"
            << std::endl;

  std::cout << "Start scoring..............................." << std::flush;
  auto score_beg = std::chrono::steady_clock::now();

  auto score = get_score<T>(Y[_num_layers % 2], num_inputs, _num_neurons_per_layer);

  auto score_end = std::chrono::steady_clock::now();
  std::cout << "finished scoring with "
            << std::chrono::duration_cast<std::chrono::milliseconds>(score_end - score_beg).count()
            << "ms"
            << std::endl;

  checkCuda(cudaStreamDestroy(stream[0]));
  checkCuda(cudaStreamDestroy(stream[1]));
  
  checkCuda(cudaFree(Y[0]));
  checkCuda(cudaFree(Y[1]));
  checkCuda(cudaFree(rowsY[0]));
  checkCuda(cudaFree(rowsY[1]));
  checkCuda(cudaFree(rlenY[0]));
  checkCuda(cudaFree(rlenY[1]));
  checkCuda(cudaFree(d_W[0]));
  checkCuda(cudaFree(d_W[1]));

  return score;
}

template <typename T>
void GPUBaseline<T>::_non_empty_rows(
  const size_t num_inputs,
  int* rlenY,
  int* rowsY,
  size_t& nnz
) const {
  
  nnz = 0;

  for(size_t i = 0; i < num_inputs; ++i) {
    if(rlenY[i] > 0) {
      rowsY[nnz++] = i;
    }
  }
}


}// end of namespace sparse_dnn ----------------------------------------------
