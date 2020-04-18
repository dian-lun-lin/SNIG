#pragma once
#include <Eigen/Dense>
#include <Eigen/Core>
#include <taskflow/taskflow.hpp>
#include <SNIG/utility/reader.hpp>
#include <SNIG/utility/matrix_format.h>
#include <SNIG/utility/cuda_error.hpp>
#include <SNIG/utility/scoring.hpp>
#include <SNIG/utility/task.hpp>
#include <chrono>
#include <vector>
#include <queue>
#include <mutex>

namespace std {
  namespace fs = experimental::filesystem;  
}

namespace snig{


template <typename T>
class SNIGPipeline {

  static_assert(
    std::is_same<T, float>::value || std::is_same<T, double>::value || std::is_same<T, half>::value,
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
      std::vector<int* >& dev_W,
      const size_t num_inputs,
      const size_t num_dev,
      const size_t batch_size,
      const size_t batch_ylen,
      const size_t batch_ysize,
      int* results
    ) const;

  public:

    SNIGPipeline(
      const std::fs::path& weight_path,
      const T bias = -.3f,
      const size_t num_neurons_per_layer = 1024,
      const size_t num_layers = 120
    );

    ~SNIGPipeline();

    size_t num_neurons_per_layer() const;
    size_t num_layers() const;

    Eigen::Matrix<int, Eigen::Dynamic, 1> infer(
      const std::fs::path& input_path,
      const size_t num_inputs,
      const size_t batch_size,
      const size_t num_dev
    ) const;

};

// ----------------------------------------------------------------------------
// Definition of SNIGPipeline
// ----------------------------------------------------------------------------

template <typename T>
SNIGPipeline<T>::SNIGPipeline(
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

  //handle aligned
  if((sizeof(int) * _p_w_index_len) % sizeof(T) != 0) {
    ++_pad;
  }

  _pp_w_index_len = _p_w_index_len + _pad;

  //pad packed weight length
  //half is 2 byte
  //max_nnz should be even, otherwis it needs to be padded
  if(std::is_same<T, half>::value) {
    _pp_wlen = _pp_w_index_len + int(0.5 * _max_nnz_per_layer);

  }
  else {
    _pp_wlen = _pp_w_index_len + (sizeof(T) / sizeof(int)) * _max_nnz_per_layer;
  }
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
SNIGPipeline<T>::~SNIGPipeline() {
  checkCuda(cudaFreeHost(_h_pinned_weight));
}

template <typename T>
size_t SNIGPipeline<T>::num_neurons_per_layer() const {
   return _num_neurons_per_layer; 
}

template <typename T>
size_t SNIGPipeline<T>::num_layers() const { 
  return _num_layers; 
}

template <typename T>
Eigen::Matrix<int, Eigen::Dynamic, 1> SNIGPipeline<T>::infer(
  const std::fs::path& input_path,
  const size_t num_inputs,
  const size_t batch_size,
  const size_t num_dev
) const {
  std::cout << "Preprocessing.............................." << std::flush;
  auto pp_beg = std::chrono::steady_clock::now();

  //weight allocation
  std::vector<int*> dev_W;
  dev_W.reserve(_num_layers);
  size_t num_layers_per_gpu = _num_layers / num_dev;
  //record to delete
  std::vector<int*> dev_record_W;

  for(size_t dev = 0; dev < num_dev; ++dev) {
    cudaSetDevice(dev);
    int* W;
    checkCuda(cudaMalloc(
      &W,
      _pp_wsize * num_layers_per_gpu
    ));
    checkCuda(cudaMemcpy(
      W,
      _h_pinned_weight + dev * num_layers_per_gpu * _pp_wlen,
      _pp_wsize * num_layers_per_gpu,
      cudaMemcpyHostToDevice
    ));
    dev_record_W.emplace_back(W);
    for(size_t cur_layer = 0; cur_layer < num_layers_per_gpu; ++cur_layer) {
      //record location of weight of each layer
      dev_W.emplace_back(W + cur_layer * _pp_wlen);
    }
  }
  cudaSetDevice(0);

  //input allocation
  size_t batch_ylen = batch_size * _num_neurons_per_layer;
  size_t batch_ysize = batch_ylen * sizeof(T);
  size_t ylen = num_inputs * _num_neurons_per_layer;
  size_t ysize = ylen * sizeof(T);

  T* source_Y;
  bool* source_rowsY;
  checkCuda(cudaMallocManaged(&source_Y, ysize));
  checkCuda(cudaMallocManaged(&source_rowsY, sizeof(bool) * num_inputs * _N_SLAB));
  checkCuda(cudaMemset(source_rowsY, 1, sizeof(bool) * num_inputs * _N_SLAB));

  std::vector<std::vector<T*> > dev_Y;
  std::vector<std::vector<bool*> > dev_rowsY;
  dev_Y.reserve(num_dev);
  dev_rowsY.reserve(num_dev);

  std::vector<T*> Y{2, nullptr};
  std::vector<bool*> rowsY{2, nullptr};
  for(size_t dev = 0; dev < num_dev; ++dev) {
    cudaSetDevice(dev);
    checkCuda(cudaMalloc(&Y[1], batch_ysize));
    checkCuda(cudaMalloc(&rowsY[1], sizeof(bool) * batch_size * _N_SLAB));
    checkCuda(cudaMemset(Y[1], 0, batch_ysize));
    checkCuda(cudaMemset(rowsY[1], 0, sizeof(bool) * batch_size * _N_SLAB));
    dev_Y.push_back(Y);
    dev_rowsY.push_back(rowsY);
  }

  read_input_binary<T>(input_path, source_Y);
  
  //final results allocation
  int* results;
  checkCuda(cudaMallocManaged(&results, sizeof(int) * num_inputs));
  checkCuda(cudaMemset(results, 0, sizeof(int) * num_inputs));

  auto pp_end = std::chrono::steady_clock::now();
  
  std::cout << "finished preprocessing with " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(pp_end - pp_beg).count()
            << "ms"
            << std::endl;

  std::cout << "Start inferencing and Identifying categories......................." << std::flush;
  auto exec_beg = std::chrono::steady_clock::now();

  _infer_taskflow(source_Y, source_rowsY, dev_Y, dev_rowsY, dev_W, num_inputs, num_dev, batch_size, batch_ylen, batch_ysize, results);

  auto exec_end = std::chrono::steady_clock::now();
  std::cout << "finished execution and identification with "
            << std::chrono::duration_cast<std::chrono::milliseconds>(exec_end - exec_beg).count()
            << "ms"
            << std::endl;

  auto results_eigen = arr_to_Eigen_int(results, num_inputs);

  checkCuda(cudaFree(source_Y));
  checkCuda(cudaFree(source_rowsY));
  for(auto& W_in_dev : dev_record_W) {
      checkCuda(cudaFree(W_in_dev));
  }
  for(auto& Y_in_dev : dev_Y) {
      checkCuda(cudaFree(Y_in_dev[1]));
  }
  for(auto& rowsY_in_dev : dev_rowsY) {
      checkCuda(cudaFree(rowsY_in_dev[1]));
  }
  checkCuda(cudaFree(results));

  return results_eigen;
}

template <typename T>
void SNIGPipeline<T>:: _infer_taskflow(
  T* source_Y,
  bool* source_rowsY,
  std::vector<std::vector<T*> >& dev_Y,
  std::vector<std::vector<bool*> >& dev_rowsY,
  std::vector<int* >& dev_W,
  const size_t num_inputs,
  const size_t num_dev,
  const size_t batch_size,
  const size_t batch_ylen,
  const size_t batch_ysize,
  int* results
) const {

  size_t num_batches = num_inputs / batch_size;
  size_t num_layers_per_gpu = _num_layers / num_dev;

  std::vector<int*> dev_results(num_dev, nullptr);

  std::vector<std::queue<size_t> > dev_start_batch(num_dev);
  std::vector<std::mutex> dev_que_mutex(num_dev);
  std::vector<std::condition_variable> dev_que_cv(num_dev);
  std::queue<size_t> first_dev_que;
  for(size_t i = 0; i < num_batches + 1; ++i) {
    first_dev_que.emplace(i * batch_size);
  }
  dev_start_batch[0] = std::move(first_dev_que);

  dim3 grid_dim(batch_size, _N_SLAB, 1);
  dim3 block_dim(2, 512, 1);

  #pragma omp parallel num_threads(num_dev)
  {
    bool stop = false;
    int dev = omp_get_thread_num(); 
    checkCuda(cudaSetDevice(dev));
    cudaStream_t infer_stream;
    checkCuda(cudaStreamCreate(&infer_stream));

    while(!stop) {
      size_t beg_inputs;

      if(dev != 0) {
        //check if current queue is empty
        {
          std::unique_lock<std::mutex> lock(dev_que_mutex[dev]);
          dev_que_cv[dev].wait(lock, [&](){ return !dev_start_batch[dev].empty();});
        }
      }

      {
        //get batch to infer
        std::unique_lock<std::mutex> lock(dev_que_mutex[dev]);
        beg_inputs = dev_start_batch[dev].front();
        dev_start_batch[dev].pop();
      }

      if(beg_inputs == num_inputs) {
        //notify next device to finish
        if(dev != num_dev - 1) {
          {
            std::unique_lock<std::mutex> lock(dev_que_mutex[dev]);
            dev_start_batch[dev + 1].emplace(num_inputs);
            dev_que_cv[dev + 1].notify_one();
          }
        }
        //this device finished all batches
        stop = true;
        continue;
      }
      dev_Y[dev][0] = source_Y + beg_inputs * _num_neurons_per_layer;
      dev_rowsY[dev][0] = source_rowsY + beg_inputs * _N_SLAB;
      dev_results[dev] = results + beg_inputs;

      for(size_t cur_layer = dev * num_layers_per_gpu; cur_layer < (dev + 1) * num_layers_per_gpu; ++cur_layer) {
        int* roffw = dev_W[cur_layer];
        int* colsw = dev_W[cur_layer] + _num_neurons_per_layer * _N_SLAB + 1;
        T* valsw = (T*)(dev_W[cur_layer] + _p_w_index_len);

        snig_inference<T><<<grid_dim, block_dim, sizeof(T) * _COL_BLK, infer_stream>>>(
          dev_Y[dev][cur_layer % 2],
          dev_rowsY[dev][cur_layer % 2],
          _COL_BLK,
          _N_SLAB,
          _num_neurons_per_layer,
          roffw,
          colsw,
          valsw,
          _bias,
          dev_rowsY[dev][(cur_layer + 1) % 2],
          dev_Y[dev][(cur_layer + 1) % 2]
        );
        checkCuda(cudaStreamSynchronize(infer_stream));
      }
      if(dev != num_dev - 1) {
        //notify next device to infer
        {
          std::unique_lock<std::mutex> lock(dev_que_mutex[dev]);
          dev_start_batch[dev + 1].emplace(beg_inputs);
        }
        dev_que_cv[dev + 1].notify_one();
      }
      else {
        //last device identify
        identify<T><<<16, 512, 0, infer_stream>>>(dev_Y[dev][0], batch_size, _num_neurons_per_layer, dev_results[dev]);
        checkCuda(cudaStreamSynchronize(infer_stream));
      }
    }
  }

  checkCuda(cudaSetDevice(0));
}


}// end of namespace snig ----------------------------------------------
