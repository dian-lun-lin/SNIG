#pragma once
#include <Eigen/Dense>
#include <SparseDNN/utility/reader.hpp>
#include <SparseDNN/utility/matrix_format.h>
#include <SparseDNN/utility/cuda_error.hpp>
#include <SparseDNN/utility/scoring.hpp>
#include <SparseDNN/parallel/task.hpp>
#include <chrono>
#include <vector>
#include <queue>
#include <mutex>
#include <tuple>
#include <thread>

namespace std {
  namespace fs = experimental::filesystem;  
}

namespace sparse_dnn{


template <typename T>
class GPUTaskflowAddIdent {

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

    void  _infer_taskflow(
      T* source_Y,
      bool* source_rowsY,
      std::vector<std::vector<T*> >& dev_Y,
      std::vector<std::vector<bool*> >& dev_rowsY,
      std::vector<std::vector<int*> >& dev_W,
      const size_t num_inputs,
      const size_t num_buff,
      const size_t num_dev,
      const size_t batch_size,
      const size_t batch_ylen,
      const size_t batch_ysize,
      int* results
    ) const;

  public:

    GPUTaskflowAddIdent(
      const std::fs::path& weight_path,
      const T bias = -.3f,
      const size_t num_neurons_per_layer = 1024,
      const size_t num_layers = 120
    );

    ~GPUTaskflowAddIdent();

    size_t num_neurons_per_layer() const;
    size_t num_layers() const;

    Eigen::Matrix<int, Eigen::Dynamic, 1> infer(
      const std::fs::path& input_path,
      const size_t num_inputs,
      const size_t batch_size,
      const size_t num_buff,
      const size_t num_dev
    ) const;

};

// ----------------------------------------------------------------------------
// Definition of GPUTaskflowAddIdent
// ----------------------------------------------------------------------------

template <typename T>
GPUTaskflowAddIdent<T>::GPUTaskflowAddIdent(
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
GPUTaskflowAddIdent<T>::~GPUTaskflowAddIdent() {
  checkCuda(cudaFreeHost(_h_pinned_weight));
}

template <typename T>
size_t GPUTaskflowAddIdent<T>::num_neurons_per_layer() const {
   return _num_neurons_per_layer; 
}

template <typename T>
size_t GPUTaskflowAddIdent<T>::num_layers() const { 
  return _num_layers; 
}

template <typename T>
Eigen::Matrix<int, Eigen::Dynamic, 1> GPUTaskflowAddIdent<T>::infer(
  const std::fs::path& input_path,
  const size_t num_inputs,
  const size_t batch_size,
  const size_t num_buff,
  const size_t num_dev
) const {
  std::cout << "Preprocessing.............................." << std::flush;
  auto pp_beg = std::chrono::steady_clock::now();

  //weight allocation
  std::vector<std::vector<int*> > dev_W;
  dev_W.reserve(num_dev);
  std::vector<int*> W(num_buff, nullptr);

  for(size_t dev = 0; dev < num_dev; ++dev) {
    cudaSetDevice(dev);
    for(auto& each_W : W) {
      checkCuda(cudaMalloc(
        &each_W,
        _pp_wsize
      ));
    }
    dev_W.push_back(W);
  }
  cudaSetDevice(0);

  //input allocation
  size_t batch_ylen = batch_size * _num_neurons_per_layer;
  size_t batch_ysize = batch_ylen * sizeof(T);
  size_t ylen = num_inputs * _num_neurons_per_layer;
  size_t ysize = ylen * sizeof(T);

  T* source_Y;
  int* results;
  bool* source_rowsY;
  checkCuda(cudaMallocManaged(&source_Y, ysize));
  checkCuda(cudaMallocManaged(&results, sizeof(int) * num_inputs));
  checkCuda(cudaMallocManaged(&source_rowsY, sizeof(bool) * num_inputs));
  checkCuda(cudaMemset(source_rowsY, 1, sizeof(bool) * num_inputs));
  checkCuda(cudaMemset(results, 0, sizeof(int) * num_inputs));

  std::vector<std::vector<T*> > dev_Y;
  std::vector<std::vector<bool*> > dev_rowsY;
  dev_Y.reserve(num_dev);
  dev_rowsY.reserve(num_dev);

  std::vector<T*> Y{2, nullptr};
  std::vector<bool*> rowsY{2, nullptr};
  for(size_t dev = 0; dev < num_dev; ++dev) {
    cudaSetDevice(dev);
    checkCuda(cudaMalloc(&Y[1], batch_ysize));
    checkCuda(cudaMalloc(&rowsY[1], sizeof(bool) * batch_size));
    checkCuda(cudaMemset(Y[1], 0, batch_ysize));
    checkCuda(cudaMemset(rowsY[1], 0, sizeof(bool) * batch_size));
    dev_Y.push_back(Y);
    dev_rowsY.push_back(rowsY);
  }

  read_input_binary<T>(input_path, batch_size, source_Y);

  auto pp_end = std::chrono::steady_clock::now();
  
  std::cout << "finished preprocessing with " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(pp_end - pp_beg).count()
            << "ms"
            << std::endl;

  std::cout << "Start inferencing and Identifying categories......................." << std::flush;
  auto exec_beg = std::chrono::steady_clock::now();

  _infer_taskflow(source_Y, source_rowsY, dev_Y, dev_rowsY, dev_W, num_inputs, num_buff, num_dev, batch_size, batch_ylen, batch_ysize, results);

  auto exec_end = std::chrono::steady_clock::now();
  std::cout << "finished execution and identification with "
            << std::chrono::duration_cast<std::chrono::milliseconds>(exec_end - exec_beg).count()
            << "ms"
            << std::endl;

  cudaSetDevice(0);
  checkCuda(cudaFree(source_Y));
  checkCuda(cudaFree(source_rowsY));

  for(auto& W_in_dev : dev_W) {
    for(auto& each_W : W_in_dev) {
      checkCuda(cudaFree(each_W));
    }
  }
  for(auto& Y_in_dev : dev_Y) {
      checkCuda(cudaFree(Y_in_dev[1]));
  }
  for(auto& rowsY_in_dev : dev_rowsY) {
      checkCuda(cudaFree(rowsY_in_dev[1]));
  }

  return arr_to_Eigen_int(results, num_inputs);
}

template <typename T>
void GPUTaskflowAddIdent<T>:: _infer_taskflow(
  T* source_Y,
  bool* source_rowsY,
  std::vector<std::vector<T*> >& dev_Y,
  std::vector<std::vector<bool*> >& dev_rowsY,
  std::vector<std::vector<int*> >& dev_W,
  const size_t num_inputs,
  const size_t num_buff,
  const size_t num_dev,
  const size_t batch_size,
  const size_t batch_ylen,
  const size_t batch_ysize,
  int* results
) const {
  tf::Taskflow taskflow("SparseDNN");
  tf::Executor executor;
  std::vector<tf::Task> first_fetch_conditions;
  std::vector<tf::Task> cudaflows;
  std::vector<tf::Task> conditions;
  first_fetch_conditions.reserve(num_dev);
  cudaflows.reserve(num_dev);
  conditions.reserve(num_dev);

  //dev_results indicate where to identify results to different categories
  std::atomic<size_t> finished_inputs{0};
  std::vector<int*> dev_results(num_dev);

  dim3 grid_dim(batch_size, 1, 1);
  dim3 block_dim(2, 512, 1);

  tf::Task start = taskflow.emplace([](){}).name("start");

  for(size_t dev = 0; dev < num_dev; ++dev) {
    first_fetch_conditions.emplace_back(taskflow.emplace([&, dev](){
      cudaSetDevice(dev);
      int is_end = 1;
      size_t beg_inputs = finished_inputs.fetch_add(batch_size);
      if(beg_inputs < num_inputs) {
        dev_Y[dev][0] = source_Y + beg_inputs * _num_neurons_per_layer;
        dev_rowsY[dev][0] = source_rowsY + beg_inputs;
        dev_results[dev] = results + beg_inputs;
        checkCuda(cudaMemPrefetchAsync(dev_Y[dev][0], batch_ysize, dev, NULL));
        checkCuda(cudaMemPrefetchAsync(dev_rowsY[dev][0], sizeof(bool) * batch_size, dev, NULL));
        checkCuda(cudaMemPrefetchAsync(dev_results[dev], sizeof(int) * batch_size, dev, NULL));
        is_end = 0;
      }
      return is_end;
    }).name("first_fetch"));

    cudaflows.emplace_back(taskflow.emplace([&, dev](tf::cudaFlow& cf){
      cf.device(dev);
      std::vector<tf::cudaTask> weight_copies;
      std::vector<tf::cudaTask> infers;
      weight_copies.reserve(_num_layers);
      infers.reserve(_num_layers);

      for(size_t cur_layer = 0; cur_layer < _num_layers; cur_layer += num_buff) {
        for(size_t k = 0; k < num_buff; ++k) {
          //tasks of cudaflow
          weight_copies.emplace_back(cf.copy(
            dev_W[dev][k],
            _h_pinned_weight + (cur_layer + k) * _pp_wlen,
            _pp_wlen
          ).name("Weight_copy_H2D"));

          int* roffw = dev_W[dev][k];
          int* colsw = dev_W[dev][k] + _num_neurons_per_layer * _N_SLAB + 1;
          T* valsw = (T*)(dev_W[dev][k] + _p_w_index_len);
          infers.emplace_back(cf.kernel(
            grid_dim,
            block_dim,
            sizeof(T) * _COL_BLK,
            wo_host_inference_test_2<T>,
            dev_Y[dev][k % 2],
            dev_rowsY[dev][k % 2],
            _COL_BLK,
            _N_SLAB,
            _num_neurons_per_layer,
            roffw,
            colsw,
            valsw,
            _bias,
            dev_rowsY[dev][(k + 1) % 2],
            dev_Y[dev][(k + 1) % 2]
          ).name("Inference"));
        }
      }
      tf::cudaTask ident = cf.kernel(16, 256, 0, identify<T>, dev_Y[dev][0], batch_size, _num_neurons_per_layer, dev_results[dev]);

      //dependencies of cudaflow
      for(size_t cur_layer = 0; cur_layer < _num_layers; ++cur_layer) {
        weight_copies[cur_layer].precede(infers[cur_layer]);

        if(cur_layer + num_buff < _num_layers) {
          infers[cur_layer].precede(weight_copies[cur_layer + num_buff]);
          weight_copies[cur_layer].precede(weight_copies[cur_layer + num_buff]);
        }
        if(cur_layer + 1 < _num_layers) {
          infers[cur_layer].precede(infers[cur_layer + 1]);
        }
      }
      infers[_num_layers - 1].precede(ident);
    }).name("GPU"));

    conditions.emplace_back(taskflow.emplace([&, dev](){
      cudaSetDevice(dev);
      int is_end = 1;
      size_t beg_inputs = finished_inputs.fetch_add(batch_size);
      if(beg_inputs < num_inputs) {
        dev_Y[dev][0] = source_Y + beg_inputs * _num_neurons_per_layer;
        dev_rowsY[dev][0] = source_rowsY + beg_inputs;
        dev_results[dev] = results + beg_inputs;
        checkCuda(cudaMemPrefetchAsync(dev_Y[dev][0], batch_ysize, dev, NULL));
        checkCuda(cudaMemPrefetchAsync(dev_rowsY[dev][0], sizeof(bool) * batch_size, dev, NULL));
        checkCuda(cudaMemPrefetchAsync(dev_results[dev], sizeof(int) * batch_size, dev, NULL));
        is_end = 0;
      }
      return is_end;
    }).name("fetch"));

  }

  tf::Task stop = taskflow.emplace([](){}).name("stop");

  //dependencies of taskflow
  for(size_t dev = 0; dev < num_dev; ++dev) {
    start.precede(first_fetch_conditions[dev]);
    first_fetch_conditions[dev].precede(cudaflows[dev], stop);
    cudaflows[dev].precede(conditions[dev]);
    conditions[dev].precede(cudaflows[dev], stop);
  }
  
  executor.run(taskflow).wait();

  checkCuda(cudaSetDevice(0));
}


}// end of namespace sparse_dnn ----------------------------------------------
