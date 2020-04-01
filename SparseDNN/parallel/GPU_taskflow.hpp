#pragma once
#include <taskflow/taskflow.hpp>
#include <Eigen/Dense>
#include <SparseDNN/utility/reader.hpp>
#include <SparseDNN/utility/matrix_format.h>
#include <SparseDNN/utility/cuda_error.hpp>
#include <SparseDNN/utility/scoring.hpp>
#include <SparseDNN/parallel/task.hpp>
#include <chrono>

namespace std {
  namespace fs = experimental::filesystem;  
}

namespace sparse_dnn{


template <typename T>
class GPUTaskflow {

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
    size_t _pad;
    size_t _N_SLAB;

    size_t _p_w_index_len;
    size_t _pp_w_index_len;
    size_t _pp_wlen;
    size_t _pp_wsize;

    void _infer_cudaflow(
      T* h_Y,
      T** Y,
      bool** rowsY,
      int** d_W,
      const size_t num_inputs,
      const size_t num_buff,
      const size_t batch_size,
      const size_t batch_ylen,
      const size_t batch_ysize
    ) const;

  public:

    GPUTaskflow(
      const std::fs::path& weight_path,
      const T bias = -.3f,
      const size_t num_neurons_per_layer = 1024,
      const size_t num_layers = 120
    );

    ~GPUTaskflow();

    size_t num_neurons_per_layer() const;
    size_t num_layers() const;

    Eigen::Matrix<int, Eigen::Dynamic, 1> infer(
      const std::fs::path& input_path,
      const size_t num_inputs,
      const size_t batch_size,
      const size_t num_buff
    ) const;

};

// ----------------------------------------------------------------------------
// Definition of GPUTaskflow
// ----------------------------------------------------------------------------

template <typename T>
GPUTaskflow<T>::GPUTaskflow(
  const std::fs::path& weight_path,
  const T bias,
  const size_t num_neurons_per_layer,
  const size_t num_layers
):
  _bias{bias},
  _num_neurons_per_layer{num_neurons_per_layer},
  _num_layers{num_layers}
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

  std::cout << "Loading the weight..........................." << std::flush;
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
  _pp_wlen = _pp_w_index_len + (sizeof(T) / sizeof(int)) * _max_nnz_per_layer;
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
GPUTaskflow<T>::~GPUTaskflow() {
  checkCuda(cudaFreeHost(_h_pinned_weight));
}

template <typename T>
size_t GPUTaskflow<T>::num_neurons_per_layer() const {
   return _num_neurons_per_layer; 
}

template <typename T>
size_t GPUTaskflow<T>::num_layers() const { 
  return _num_layers; 
}

template <typename T>
Eigen::Matrix<int, Eigen::Dynamic, 1> GPUTaskflow<T>::infer(
  const std::fs::path& input_path,
  const size_t num_inputs,
  const size_t batch_size,
  const size_t num_buff
) const {

  std::cout << "Preprocessing.............................." << std::flush;
  auto pp_beg = std::chrono::steady_clock::now();

  //weight allocation
  int *d_W[num_buff];
  for(size_t i = 0; i < num_buff; ++i) {
    checkCuda(cudaMalloc(
      &d_W[i],
      _pp_wsize
    ));
  }

  size_t batch_ylen = batch_size * _num_neurons_per_layer;
  size_t batch_ysize = batch_ylen * sizeof(T);
  size_t ylen = num_inputs * _num_neurons_per_layer;
  size_t ysize = ylen * sizeof(T);

  //input allocation
  T* h_Y;
  checkCuda(cudaMallocHost(
    (void**)&h_Y,
    ysize
  ));

  T* Y[2];  
  bool* rowsY[2];

  checkCuda(cudaMalloc(&Y[0], batch_ysize));
  checkCuda(cudaMalloc(&Y[1], batch_ysize));
  checkCuda(cudaMallocManaged(&rowsY[0], sizeof(bool) * batch_size));
  checkCuda(cudaMallocManaged(&rowsY[1], sizeof(bool) * batch_size));
  checkCuda(cudaMemset(Y[0], 0, batch_ysize));
  checkCuda(cudaMemset(Y[1], 0, batch_ysize));
  checkCuda(cudaMemset(rowsY[0], 1, sizeof(bool) * batch_size));
  checkCuda(cudaMemset(rowsY[1], 0, sizeof(bool) * batch_size));

  read_input_binary<T>(input_path, batch_size, h_Y);

  auto pp_end = std::chrono::steady_clock::now();
  
  std::cout << "finished preprocessing with " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(pp_end - pp_beg).count()
            << "ms"
            << std::endl;

  std::cout << "Start inference............................" << std::flush;
  auto exec_beg = std::chrono::steady_clock::now();

  _infer_cudaflow(h_Y, Y, rowsY, d_W, num_inputs, num_buff, batch_size, batch_ylen, batch_ysize);

  auto exec_end = std::chrono::steady_clock::now();
  std::cout << "finished execution with "
            << std::chrono::duration_cast<std::chrono::milliseconds>(exec_end - exec_beg).count()
            << "ms"
            << std::endl;

  std::cout << "Start scoring..............................." << std::flush;
  auto score_beg = std::chrono::steady_clock::now();

  auto score = get_score<T>(h_Y, num_inputs, _num_neurons_per_layer);

  auto score_end = std::chrono::steady_clock::now();
  std::cout << "finished scoring with "
            << std::chrono::duration_cast<std::chrono::milliseconds>(score_end - score_beg).count()
            << "ms"
            << std::endl;

  checkCuda(cudaFreeHost(h_Y));
  checkCuda(cudaFree(Y[0]));
  checkCuda(cudaFree(Y[1]));
  checkCuda(cudaFree(rowsY[0]));
  checkCuda(cudaFree(rowsY[1]));

  for(size_t k =0; k < num_buff; ++k) {
    checkCuda(cudaFree(d_W[k]));
  }

  return score;
}

//Due to the packed weight array, this infer is only for float type.
template <typename T>
void GPUTaskflow<T>:: _infer_cudaflow(
  T* h_Y,
  T** Y,
  bool** rowsY,
  int** d_W,
  const size_t num_inputs,
  const size_t num_buff,
  const size_t batch_size,
  const size_t batch_ylen,
  const size_t batch_ysize
) const {

  int device = -1;
  cudaGetDevice(&device);
  checkCuda(cudaMemPrefetchAsync(rowsY[0], sizeof(bool) * batch_size, device, NULL));
  checkCuda(cudaMemPrefetchAsync(rowsY[1], sizeof(bool) * batch_size, device, NULL));

  tf::Taskflow taskflow("SparseDNN");
  tf::Executor executor;
  dim3 grid_dim(batch_size, 1, 1);
  dim3 block_dim(16, 16, 1);

  auto cudaflow = taskflow.emplace([&](tf::cudaFlow& cf){
    std::vector<tf::cudaTask> weight_copies;
    std::vector<tf::cudaTask> infers;
    std::vector<tf::cudaTask> memsets;
    weight_copies.reserve(_num_layers);
    infers.reserve(_num_layers);
    memsets.reserve(_num_layers);
    for(size_t cur_layer = 0; cur_layer < _num_layers; cur_layer += num_buff) {
      for(size_t k = 0; k < num_buff; ++k) {
        ////create tasks
        weight_copies.emplace_back(cf.copy(
          d_W[k],
          _h_pinned_weight + (cur_layer + k) * _pp_wlen,
          _pp_wlen
        ));

        int* roffw = d_W[k];
        int* colsw = d_W[k] + _num_neurons_per_layer * _N_SLAB + 1;
        T* valsw = (T*)(d_W[k] + _p_w_index_len);
        infers.emplace_back(cf.kernel(
          grid_dim,
          block_dim,
          sizeof(T) * _COL_BLK,
          wo_host_inference_test<T>,
          Y[k % 2],
          rowsY[k % 2],
          _COL_BLK,
          _N_SLAB,
          _num_neurons_per_layer,
          roffw,
          colsw,
          valsw,
          _bias,
          rowsY[(k + 1) % 2],
          Y[(k + 1) % 2]
        ));

        memsets.emplace_back(cf.memset(Y[k % 2], 0, batch_ysize));
      }
    }
    //create dependencies
    for(size_t cur_layer = 0; cur_layer < _num_layers; ++cur_layer) {
      weight_copies[cur_layer].precede(infers[cur_layer]);
      infers[cur_layer].precede(memsets[cur_layer]);

      if(cur_layer + num_buff < _num_layers) {
        infers[cur_layer].precede(weight_copies[cur_layer + num_buff]);
        weight_copies[cur_layer].precede(weight_copies[cur_layer + num_buff]);
      }
      if(cur_layer + 1 < _num_layers) {
        memsets[cur_layer].precede(infers[cur_layer + 1]);
      }
    }

  });
  
  for(size_t batch = 0; batch < num_inputs; batch += batch_size) {
    checkCuda(cudaMemcpy(
      Y[0],
      h_Y + batch * _num_neurons_per_layer,
      batch_ysize,
      cudaMemcpyHostToDevice
    ));

    executor.run(taskflow).wait();

    checkCuda(cudaMemcpy(
      h_Y + batch * _num_neurons_per_layer,
      Y[num_buff % 2],
      batch_ysize,
      cudaMemcpyDeviceToHost
    ));
    checkCuda(cudaMemset(rowsY[0], 1, sizeof(bool) * batch_size));
  }

}

}// end of namespace sparse_dnn ----------------------------------------------
