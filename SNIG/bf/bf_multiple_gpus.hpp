#pragma once
#include <Eigen/Dense>
#include <SNIG/utility/reader.hpp>
#include <SNIG/utility/matrix_format.h>
#include <SNIG/utility/cuda_error.hpp>
#include <SNIG/utility/scoring.hpp>
#include <SNIG/utility/utility.hpp>
#include <SNIG/utility/task.hpp>
#include <omp.h>
#include <chrono>

namespace std {
  namespace fs = experimental::filesystem;  
}

namespace snig{

template <typename T>  
class BFMultiGpu {

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

    void _infer_BF(
      std::vector<std::vector<int*> >& dev_W,
      std::vector<std::vector<int*> >& dev_rowsY,
      std::vector<std::vector<int*> >& dev_rlenY,
      std::vector<std::vector<T*> >& dev_Y,
      std::vector<size_t>& dev_nerowsY,
      size_t num_dev,
      size_t each_partition,
      size_t remains,
      int* results
    ) const;

  public:

    BFMultiGpu(
      const std::fs::path& weight_path,
      const T bias = -.3f,
      const size_t num_neurons_per_layer = 1024,
      const size_t num_layers = 120
    );

    ~BFMultiGpu();
    
    size_t num_neurons_per_layer() const;

    size_t num_layers() const;

    Eigen::Matrix<int, Eigen::Dynamic, 1> infer(
      const std::fs::path& input_path,
      const size_t num_inputs,
      const size_t num_dev
    ) const;

};

// ----------------------------------------------------------------------------
// Definition of BFMultiGpu
// ----------------------------------------------------------------------------

template <typename T>
BFMultiGpu<T>::BFMultiGpu(
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
BFMultiGpu<T>:: ~BFMultiGpu() {
  checkCuda(cudaFreeHost(_h_pinned_weight));
}

template <typename T>
size_t BFMultiGpu<T>::num_neurons_per_layer() const {
 return _num_neurons_per_layer; 
}

template <typename T>
size_t BFMultiGpu<T>::num_layers() const { 
  return _num_layers; 
}

template <typename T>
Eigen::Matrix<int, Eigen::Dynamic, 1> BFMultiGpu<T>::infer(
  const std::fs::path& input_path,
  const size_t num_inputs,
  const size_t num_dev
) const {
  //Due to without NVlink,
  //this implementation doesn't do load balancing.
  //It actually let GPUs work in their own partition Ys
  //Hence, rowsY for each GPU should be indexed individually.

  std::cout << "Preprocessing.............................." << std::flush;
  auto pp_beg = std::chrono::steady_clock::now();
  
  
  //weight allocation
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
  size_t ry_size = num_inputs * sizeof(int);

  std::vector<int*> rowsY(2, nullptr);
  std::vector<int*> rlenY(2, nullptr);
  std::vector<T*> Y(2, nullptr);  
  checkCuda(cudaMallocManaged(&rowsY[0], ry_size));
  checkCuda(cudaMallocManaged(&rowsY[1], ry_size));
  checkCuda(cudaMallocManaged(&rlenY[0], ry_size));
  checkCuda(cudaMallocManaged(&rlenY[1], ry_size));
  checkCuda(cudaMallocManaged(&Y[0], ysize));
  checkCuda(cudaMallocManaged(&Y[1], ysize));
  checkCuda(cudaMemset(rowsY[0], 0, ry_size));
  checkCuda(cudaMemset(rowsY[1], 0, ry_size));
  checkCuda(cudaMemset(rlenY[0], 1, ry_size));
  checkCuda(cudaMemset(rlenY[1], 0, ry_size));

  read_input_binary<T>(input_path, Y[0]);

  //final results allocation
  int* results;
  checkCuda(cudaMallocManaged(&results, sizeof(int) * num_inputs));
  checkCuda(cudaMemset(results, 0, sizeof(int) * num_inputs));

  //partition
  //assume all rows of inputs are non-empty.
  size_t each_partition = num_inputs / num_dev;
  size_t remains = num_inputs % num_dev;

  //use dev_Y, dev_rowsY,  dev_rlenY, and dev_nerowsY to record each GPUs's own data
  std::vector<std::vector<int*> > dev_rowsY;
  std::vector<std::vector<int*> > dev_rlenY;
  std::vector<std::vector<T*> > dev_Y;
  std::vector<size_t> dev_nerowsY;
  dev_rowsY.reserve(num_dev);
  dev_rlenY.reserve(num_dev);
  dev_Y.reserve(num_dev);
  dev_nerowsY.reserve(num_dev);
  
  std::vector<int*> each_GPU_rowsY(2, nullptr);
  std::vector<int*> each_GPU_rlenY(2, nullptr);
  std::vector<T*> each_GPU_Y(2, nullptr);
  for(size_t dev = 0 ; dev < num_dev; ++dev) {
    dev_nerowsY.emplace_back(each_partition);
    for(int buff = 0; buff < 2; ++buff) {
      each_GPU_rowsY[buff] = rowsY[buff] + dev * each_partition; 
      each_GPU_rlenY[buff] = rlenY[buff] + dev * each_partition; 
      each_GPU_Y[buff] = Y[buff] + dev * each_partition * _num_neurons_per_layer;
    }
    dev_rowsY.emplace_back(each_GPU_rowsY);
    dev_rlenY.emplace_back(each_GPU_rlenY);
    dev_Y.emplace_back(each_GPU_Y);
  }
  //last device
  dev_nerowsY[num_dev - 1] += remains;
  
  //rowsY should be indexed by each GPUs' own partition inputs, rather than indexed by whole inputs
  for(size_t dev = 0; dev < num_dev - 1; ++dev) {
    _non_empty_rows(each_partition, dev_rlenY[dev][0], dev_rowsY[dev][0], dev_nerowsY[dev]);
  }
  //last device
  _non_empty_rows(each_partition + remains, dev_rlenY[num_dev - 1][0], dev_rowsY[num_dev - 1][0], dev_nerowsY[num_dev - 1]);
  
  //Advise
  for(size_t dev = 0; dev < num_dev - 1; ++dev) {
    for(int buff = 0; buff < 2; ++buff) {
      checkCuda(cudaMemAdvise(
        dev_Y[dev][buff],
        each_partition * _num_neurons_per_layer * sizeof(T),
        cudaMemAdviseSetPreferredLocation,
        dev 
      ));
      checkCuda(cudaMemAdvise(
        dev_rowsY[dev][buff],
        each_partition * sizeof(int),
        cudaMemAdviseSetPreferredLocation,
        dev 
      ));
      checkCuda(cudaMemAdvise(
        dev_rlenY[dev][buff],
        each_partition * sizeof(int),
        cudaMemAdviseSetPreferredLocation,
        dev 
      ));
    }
  }

  //Advise for last device
  for(int buff = 0; buff < 2; ++buff) {
    checkCuda(cudaMemAdvise(
      dev_Y[num_dev - 1][buff],
      (each_partition + remains) * _num_neurons_per_layer * sizeof(T),
      cudaMemAdviseSetPreferredLocation,
      num_dev - 1
    ));
    checkCuda(cudaMemAdvise(
      dev_rowsY[num_dev - 1][buff],
      (each_partition + remains) * sizeof(int),
      cudaMemAdviseSetPreferredLocation,
      num_dev - 1 
    ));
    checkCuda(cudaMemAdvise(
      dev_rlenY[num_dev - 1][buff],
      (each_partition + remains) * sizeof(int),
      cudaMemAdviseSetPreferredLocation,
      num_dev - 1 
    ));
  }

  auto pp_end = std::chrono::steady_clock::now();

  std::cout << "finished preprocessing with " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(pp_end - pp_beg).count()
            << "ms"
            << std::endl;

  std::cout << "Start inferencing and Identifying categories......................." << std::flush;
  auto exec_beg = std::chrono::steady_clock::now();

  _infer_BF(
    dev_W,
    dev_rowsY,
    dev_rlenY,
    dev_Y,
    dev_nerowsY,
    num_dev,
    each_partition,
    remains,
    results
  );

  auto exec_end = std::chrono::steady_clock::now();
  std::cout << "finished execution and identification with "
            << std::chrono::duration_cast<std::chrono::milliseconds>(exec_end - exec_beg).count()
            << "ms"
            << std::endl;

  auto results_eigen = arr_to_Eigen_int(results, num_inputs);
  
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
  checkCuda(cudaFree(results));

  return results_eigen;
}

template <typename T>
void BFMultiGpu<T>::_infer_BF(
  std::vector<std::vector<int*> >& dev_W,
  std::vector<std::vector<int*> >& dev_rowsY,
  std::vector<std::vector<int*> >& dev_rlenY,
  std::vector<std::vector<T*> >& dev_Y,
  std::vector<size_t>& dev_nerowsY,
  size_t num_dev,
  size_t each_partition,
  size_t remains,
  int* results
) const {

  std::vector<int*> dev_results(num_dev);
  for(size_t dev = 0; dev < num_dev; ++dev) {
    dev_results[dev] = results + dev * each_partition;
  }

  dim3 threads(2, 512, 1);

  std::vector<std::vector<cudaStream_t> > dev_stream;
  dev_stream.reserve(num_dev);
  std::vector<cudaStream_t> stream(2);
  for(size_t dev = 0; dev < num_dev; ++dev) {
    dev_stream.emplace_back(stream);
  }

  #pragma omp parallel num_threads(num_dev)
  {
    int dev = omp_get_thread_num(); 
    checkCuda(cudaSetDevice(dev));
    checkCuda(cudaStreamCreate(&dev_stream[dev][0]));
    checkCuda(cudaStreamCreate(&dev_stream[dev][1]));
    for(size_t cur_layer = 0; cur_layer < _num_layers; ++cur_layer) {
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

      bf_inference<T><<<dev_nerowsY[dev], threads, sizeof(T) * _COL_BLK, dev_stream[dev][1]>>>(
        dev_Y[dev][cur_layer % 2],
        dev_nerowsY[dev],
        dev_rowsY[dev][cur_layer % 2],
        dev_rlenY[dev][cur_layer % 2],
        _COL_BLK,
        _N_SLAB,
        _num_neurons_per_layer,
        roffw,
        colsw,
        valsw,
        _bias,
        dev_Y[dev][(cur_layer + 1) % 2],
        dev_rlenY[dev][(cur_layer + 1) % 2]
      );
      checkCuda(cudaStreamSynchronize(dev_stream[dev][1]));

      if(dev == num_dev - 1) {
        //last device
        _non_empty_rows(
          each_partition + remains,
          dev_rlenY[dev][(cur_layer + 1) % 2],
          dev_rowsY[dev][(cur_layer + 1) % 2],
          dev_nerowsY[dev]
        );
        checkCuda(cudaMemset(
          dev_Y[dev][cur_layer % 2],
          0,
          (each_partition + remains) * _num_neurons_per_layer * sizeof(T)
        ));
      }
      else {
        _non_empty_rows(
          each_partition,
          dev_rlenY[dev][(cur_layer + 1) % 2],
          dev_rowsY[dev][(cur_layer + 1) % 2],
          dev_nerowsY[dev]
        );
        checkCuda(cudaMemset(
          dev_Y[dev][cur_layer % 2],
          0,
          (each_partition) * _num_neurons_per_layer * sizeof(T)
        ));
      }
      checkCuda(cudaStreamSynchronize(dev_stream[dev][0]));
      #pragma omp barrier
    }
    if(dev == num_dev - 1) {
      identify<T><<<16, 512>>>(dev_Y[dev][0], each_partition + remains, _num_neurons_per_layer, dev_results[dev]);
    }
    else {
      identify<T><<<16, 512>>>(dev_Y[dev][0], each_partition, _num_neurons_per_layer, dev_results[dev]);
    }
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaStreamDestroy(dev_stream[dev][0]));
    checkCuda(cudaStreamDestroy(dev_stream[dev][1]));
  }

}

template <typename T>
void BFMultiGpu<T>::_non_empty_rows(
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

}// end of namespace snig ----------------------------------------------
