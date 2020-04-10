#pragma once
#include <Eigen/Dense>
#include <SparseDNN/utility/matrix_format.h>
#include <SparseDNN/utility/cuda_error.hpp>
#include <SparseDNN/utility/scoring.hpp>
#include <SparseDNN/utility/utility.hpp>
#include <SparseDNN/parallel/task.hpp>
#include <omp.h>
#include <chrono>

namespace std {
  namespace fs = experimental::filesystem;  
}

namespace sparse_dnn{

template <typename T>  
class GPUBaselineMulti {

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

    std::tuple<size_t, size_t> _partition(
      size_t nerowsY,
      size_t num_dev
    ) const;

  public:

    GPUBaselineMulti(
      const std::fs::path& weight_path,
      const T bias = -.3f,
      const size_t num_neurons_per_layer = 1024,
      const size_t num_layers = 120
    );

    ~GPUBaselineMulti();
    
    size_t num_neurons_per_layer() const;

    size_t num_layers() const;

    Eigen::Matrix<int, Eigen::Dynamic, 1> infer(
      const std::fs::path& input_path,
      const size_t num_inputs,
      const size_t num_dev
    ) const;

};

// ----------------------------------------------------------------------------
// Definition of GPUBaselineMulti
// ----------------------------------------------------------------------------

template <typename T>
GPUBaselineMulti<T>::GPUBaselineMulti(
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
GPUBaselineMulti<T>:: ~GPUBaselineMulti() {
  checkCuda(cudaFreeHost(_h_pinned_weight));
}

template <typename T>
size_t GPUBaselineMulti<T>::num_neurons_per_layer() const {
 return _num_neurons_per_layer; 
}

template <typename T>
size_t GPUBaselineMulti<T>::num_layers() const { 
  return _num_layers; 
}

template <typename T>
Eigen::Matrix<int, Eigen::Dynamic, 1> GPUBaselineMulti<T>::infer(
  const std::fs::path& input_path,
  const size_t num_inputs,
  const size_t num_dev
) const {

  std::cout << "Preprocessing.............................." << std::flush;
  auto pp_beg = std::chrono::steady_clock::now();
  
  
  std::vector<std::vector<int*> > dev_W;
  dev_W.reserve(num_dev);
  std::vector<int*> W(2, nullptr);
  for(size_t dev = 0; dev < num_dev; ++dev) {
    checkCuda(cudaSetDevice(dev));
    checkCuda(cudaMalloc(
      &W[0],
      _pp_wsize
    ));
    checkCuda(cudaMalloc(
      &W[1],
      _pp_wsize
    ));
    checkCuda(cudaMemcpy(
      W[0],
      _h_pinned_weight,
      _pp_wsize,
      cudaMemcpyHostToDevice
    ));
    dev_W.emplace_back(W);
  }
  checkCuda(cudaSetDevice(0));

  //input allocation
  size_t ylen = num_inputs * _num_neurons_per_layer;
  size_t ysize = ylen * sizeof(T);
  size_t ry_size = sizeof(int) * num_inputs;

  T* Y[2];  
  int *rowsY[2], *rlenY[2];

  checkCuda(cudaMallocManaged(&Y[0], ysize));
  checkCuda(cudaMallocManaged(&Y[1], ysize));
  checkCuda(cudaMallocManaged(&rowsY[0], ry_size));
  checkCuda(cudaMallocManaged(&rowsY[1], ry_size));
  checkCuda(cudaMallocManaged(&rlenY[0], ry_size));
  checkCuda(cudaMallocManaged(&rlenY[1], ry_size));
  checkCuda(cudaMemset(Y[0], 0, ysize));
  checkCuda(cudaMemset(Y[1], 0, ysize));
  checkCuda(cudaMemset(rowsY[0], 0, ry_size));
  checkCuda(cudaMemset(rowsY[1], 0, ry_size));
  checkCuda(cudaMemset(rlenY[0], 0, ry_size));
  checkCuda(cudaMemset(rlenY[1], 0, ry_size));
  checkCuda(cudaDeviceSynchronize());

  size_t nerowsY{0};
  read_input_binary<T>(input_path, Y[0], rlenY[0], rowsY[0], nerowsY);

  size_t quotient{0};
  size_t remains{0};
  std::tie(quotient, remains) = _partition(nerowsY, num_dev);

  size_t begRow{0};
  for(size_t dev = 0; dev < num_dev - 1; ++dev) {
    for(int buff = 0; buff < 2; ++buff) {
      checkCuda(cudaMemAdvise(
        Y[buff] + begRow * _num_neurons_per_layer,
        quotient * _num_neurons_per_layer * sizeof(T),
        cudaMemAdviseSetPreferredLocation,
        dev 
      ));
      checkCuda(cudaMemAdvise(
        rowsY[buff] + begRow,
        quotient * sizeof(int),
        cudaMemAdviseSetPreferredLocation,
        dev 
      ));
      checkCuda(cudaMemAdvise(
        rlenY[buff] + begRow,
        quotient * sizeof(int),
        cudaMemAdviseSetPreferredLocation,
        dev 
      ));
    }
    begRow += quotient;
  }
  //last device
  for(int buff = 0; buff < 2; ++buff) {
    checkCuda(cudaMemAdvise(
      Y[buff] + begRow * _num_neurons_per_layer,
      (quotient + remains) * _num_neurons_per_layer * sizeof(T),
      cudaMemAdviseSetPreferredLocation,
      num_dev - 1
    ));
    checkCuda(cudaMemAdvise(
      rowsY[buff] + begRow,
      (quotient + remains) * sizeof(int),
      cudaMemAdviseSetPreferredLocation,
      num_dev - 1 
    ));
    checkCuda(cudaMemAdvise(
      rlenY[buff] + begRow,
      (quotient + remains) * sizeof(int),
      cudaMemAdviseSetPreferredLocation,
      num_dev - 1 
    ));
  }


  auto pp_end = std::chrono::steady_clock::now();
  
  std::cout << "finished preprocessing with " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(pp_end - pp_beg).count()
            << "ms"
            << std::endl;
  int dim_y{0};
  if(_num_neurons_per_layer == 1024) {
    dim_y = 16;
  }
  else if(_num_neurons_per_layer == 4096) {
    dim_y = 32;
  }
  else if(_num_neurons_per_layer == 16384) {
    dim_y = 64;
  }
  dim3 threads(4, dim_y, 1);

  std::cout << "Start inference............................" << std::flush;

  auto exec_beg = std::chrono::steady_clock::now();

  std::vector<std::vector<cudaStream_t> > dev_stream;
  dev_stream.reserve(num_dev);
  std::vector<cudaStream_t> stream(2);
  for(size_t dev = 0; dev < num_dev; ++dev) {
    dev_stream.emplace_back(stream);
  }
  

  for(size_t cur_layer = 0; cur_layer < _num_layers; ++cur_layer) {
    #pragma omp parallel num_threads(num_dev)
    {
      int dev = omp_get_thread_num(); 
      checkCuda(cudaSetDevice(dev));
      checkCuda(cudaStreamCreate(&dev_stream[dev][0]));
      checkCuda(cudaStreamCreate(&dev_stream[dev][1]));
      if(cur_layer != _num_layers - 1) {
        checkCuda(cudaMemcpyAsync(
          dev_W[dev][(cur_layer + 1) % 2],
          _h_pinned_weight + (cur_layer + 1) * (_pp_wlen),
          _pp_wsize,
          cudaMemcpyHostToDevice,
          dev_stream[dev][0]
        ));
      }

      int* roffw = dev_W[dev][cur_layer % 2];
      int* colsw = dev_W[dev][cur_layer % 2] + _num_neurons_per_layer * _N_SLAB + 1;
      T* valsw = (T*)(dev_W[dev][cur_layer % 2] + _p_w_index_len);
      size_t handle_nerowsY{0};
      if(dev == num_dev - 1) {
        handle_nerowsY = quotient + remains;
      }
      else {
        handle_nerowsY = quotient;
      }
      
      baseline_inference<T><<<handle_nerowsY, threads, sizeof(T) * _COL_BLK, dev_stream[dev][1]>>>(
        Y[cur_layer % 2],
        handle_nerowsY,
        rowsY[cur_layer % 2] + quotient * dev,
        rlenY[cur_layer % 2],
        _COL_BLK,
        _N_SLAB,
        _num_neurons_per_layer,
        roffw,
        colsw,
        valsw,
        _bias,
        Y[(cur_layer + 1) % 2],
        rlenY[(cur_layer + 1) % 2]
      );

      checkCuda(cudaDeviceSynchronize());
      checkCuda(cudaStreamDestroy(dev_stream[dev][0]));
      checkCuda(cudaStreamDestroy(dev_stream[dev][1]));
    }

    _non_empty_rows(
      num_inputs,
      rlenY[(cur_layer + 1) % 2],
      rowsY[(cur_layer + 1) % 2],
      nerowsY
    );
    std::tie(quotient, remains) = _partition(nerowsY, num_dev);
    checkCuda(cudaMemset(
      Y[cur_layer % 2],
      0,
      ysize
    ));
  }

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

  
  checkCuda(cudaFree(Y[0]));
  checkCuda(cudaFree(Y[1]));
  checkCuda(cudaFree(rowsY[0]));
  checkCuda(cudaFree(rowsY[1]));
  checkCuda(cudaFree(rlenY[0]));
  checkCuda(cudaFree(rlenY[1]));
  for(auto& each_dev_W : dev_W) {
    for(auto& w : each_dev_W) {
      checkCuda(cudaFree(w));
    }
  }

  return score;
}

template <typename T>
void GPUBaselineMulti<T>::_non_empty_rows(
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

template <typename T>
std::tuple<size_t, size_t> GPUBaselineMulti<T>::_partition(
  size_t nerowsY,
  size_t num_dev
) const {
  size_t quotient = nerowsY / num_dev;
  size_t remains = nerowsY % num_dev;
  return std::tie(quotient, remains);
}

}// end of namespace sparse_dnn ----------------------------------------------
