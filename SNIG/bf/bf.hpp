#pragma once
#include <Eigen/Core>
#include <SNIG/utility/reader.hpp>
#include <SNIG/utility/matrix_format.h>
#include <SNIG/utility/cuda_error.hpp>
#include <SNIG/utility/scoring.hpp>
#include <SNIG/bf/kernel.hpp>
#include <SNIG/utility/utility.hpp>
#include <SNIG/base/base.hpp>
#include <omp.h>

namespace std {
  namespace fs = experimental::filesystem;  
}

namespace snig{

template <typename T>  
class BF : public Base<T> {
  //Since we don't have NVlink,
  //this implementation doesn't do load balancing at each iteration.
  //It actually let GPUs work in their own partitioned input data

  static_assert(
    std::is_same<T, float>::value || std::is_same<T, double>::value,
    "data type must be either float or double"
  );
  
  private:

    //Both BF and SNIG use the maximum externel shared memory for inference
    // COL_BLK == Base<T>::_sec_size
    // N_SLAB  == Base<T>::_num_secs

    std::vector<int*> _rowsY{2, nullptr};
    std::vector<int*> _rlenY{2, nullptr};
    std::vector<T*> _Y{2, nullptr};
    
    //use dev_Y, dev_rowsY,  dev_rlenY, and dev_nerowsY to record each GPUs' own data
    //Since each GPU owns parts of inputs
    //each rowsY in _dev_rowsY is indexed individually by each GPU, rather than indexed by whole inputs
    std::vector<std::vector<int*> > _dev_W;
    std::vector<std::vector<int*> > _dev_rowsY;
    std::vector<std::vector<int*> > _dev_rlenY;
    std::vector<std::vector<T*> > _dev_Y;
    std::vector<size_t> _dev_nerowsY;
    std::vector<size_t> _dev_num_inputs;

    int* _results;

    void _infer();

    void _non_empty_rows(const size_t dev, const size_t buff);

    void _set_parameters(
      const size_t num_inputs,
      const size_t num_gpus
    );
    
    void _preprocess(const std::fs::path& input_path);
    
    void _weight_alloc();

    void _input_alloc();

    void _result_alloc();

  public:

    BF(
      const dim3& threads,
      const std::fs::path& weight_path,
      const T bias = -.3f,
      const size_t num_neurons_per_layer = 1024,
      const size_t num_layers = 120
    );

    ~BF();
    
    Eigen::Matrix<int, Eigen::Dynamic, 1> infer(
      const std::fs::path& input_path,
      const size_t num_inputs,
      const size_t num_gpus
    );

};

// ----------------------------------------------------------------------------
// Definition of BF
// ----------------------------------------------------------------------------

template <typename T>
BF<T>::BF(
  const dim3& threads,
  const std::fs::path& weight_path,
  const T bias,
  const size_t num_neurons,
  const size_t num_layers
):
  Base<T>(threads, weight_path, bias, num_neurons, num_layers)
{
  Base<T>::log("Constructing BF method......", "\n");
}

template <typename T>
BF<T>:: ~BF() {
  for(auto& each_Y : _Y) {
    checkCuda(cudaFree(each_Y));
  }
  for(auto& each_rowsY : _rowsY) {
    checkCuda(cudaFree(each_rowsY));
  }
  for(auto& each_rlenY : _rlenY) {
    checkCuda(cudaFree(each_rlenY));
  }
  for(auto& each_dev_W : _dev_W) {
    for(auto& w : each_dev_W) {
      checkCuda(cudaFree(w));
    }
  }
  checkCuda(cudaFree(_results));
}

template <typename T>
Eigen::Matrix<int, Eigen::Dynamic, 1> BF<T>::infer(
  const std::fs::path& input_path,
  const size_t num_inputs,
  const size_t num_gpus
) {
  _set_parameters(num_inputs, num_gpus);

  _preprocess(input_path);

  _infer();

  return arr_to_Eigen_int(_results, Base<T>::_num_inputs);
}

template <typename T>
void BF<T>::_set_parameters(
  const size_t num_inputs,
  const size_t num_gpus
) {
  Base<T>::log("Using ", num_gpus, " GPUs", "\n");
  Base<T>::log("Total input size : ", num_inputs, "\n\n");

  Base<T>::_num_gpus = num_gpus;
  Base<T>::_num_inputs = num_inputs;

  _dev_rowsY.reserve(Base<T>::_num_gpus);
  _dev_rlenY.reserve(Base<T>::_num_gpus);
  _dev_Y.reserve(Base<T>::_num_gpus);
  _dev_W.reserve(Base<T>::_num_gpus);
  _dev_nerowsY.reserve(Base<T>::_num_gpus);
  _dev_num_inputs.reserve(Base<T>::_num_gpus);
  
}

template <typename T>
void BF<T>::_preprocess(const std::fs::path& input_path) {
  Base<T>::log("Preprocessing...... ");
  Base<T>::tic();

  //weight allocation
  _weight_alloc();

  //input allocation
  _input_alloc();

  //final results allocation
  _result_alloc();
  
  //read input
  read_input_binary<T>(input_path, _Y[0]);

  Base<T>::toc();
  Base<T>::log("Finish preprocessing with ", Base<T>::duration(), " ms", "\n");
}


template <typename T>
void BF<T>::_infer() {
  Base<T>::log("Start inference...... ", "\n");
  Base<T>::tic();

  //store results
  std::vector<int*> dev_results(Base<T>::_num_gpus);
  dev_results[0] = _results;
  for(size_t dev = 1; dev < Base<T>::_num_gpus; ++dev) {
    dev_results[dev] = dev_results[dev - 1] +  _dev_num_inputs[dev - 1];
  }

  std::vector<std::vector<cudaStream_t> > dev_stream;
  dev_stream.reserve(Base<T>::_num_gpus);

  std::vector<cudaStream_t> stream(2);
  for(size_t dev = 0; dev < Base<T>::_num_gpus; ++dev) {
    dev_stream.emplace_back(stream);
  }

  #pragma omp parallel num_threads(Base<T>::_num_gpus)
  {
    int dev = omp_get_thread_num(); 
    checkCuda(cudaSetDevice(dev));
    checkCuda(cudaStreamCreate(&dev_stream[dev][0]));
    checkCuda(cudaStreamCreate(&dev_stream[dev][1]));
    for(size_t cur_layer = 0; cur_layer < Base<T>::_num_layers; ++cur_layer) {
      if(cur_layer != Base<T>::_num_layers - 1) {
        checkCuda(cudaMemcpyAsync(
          _dev_W[dev][(cur_layer + 1) % 2],
          Base<T>::_host_pinned_weight + (cur_layer + 1) * (Base<T>::_pp_wlen),
          Base<T>::_pp_wsize,
          cudaMemcpyHostToDevice,
          dev_stream[dev][0]
        ));
      }

      int* roffw = _dev_W[dev][cur_layer % 2];
      int* colsw = _dev_W[dev][cur_layer % 2] + Base<T>::_num_neurons * Base<T>::_num_secs + 1;
      T* valsw = (T*)(_dev_W[dev][cur_layer % 2] + Base<T>::_p_w_index_len);

      bf_inference<T><<<_dev_nerowsY[dev], Base<T>::_threads, sizeof(T) * Base<T>::_sec_size, dev_stream[dev][1]>>>(
        _dev_Y[dev][cur_layer % 2],
        _dev_nerowsY[dev],
        _dev_rowsY[dev][cur_layer % 2],
        _dev_rlenY[dev][cur_layer % 2],
        Base<T>::_sec_size,
        Base<T>::_num_secs,
        Base<T>::_num_neurons,
        roffw,
        colsw,
        valsw,
        Base<T>::_bias,
        _dev_Y[dev][(cur_layer + 1) % 2],
        _dev_rlenY[dev][(cur_layer + 1) % 2]
      );
      checkCuda(cudaStreamSynchronize(dev_stream[dev][1]));

      _non_empty_rows(dev, (cur_layer + 1) % 2);

      //Rolling swap requires resetting memory for next iteration
      checkCuda(cudaMemset(
        _dev_Y[dev][cur_layer % 2],
        0,
        _dev_num_inputs[dev] * Base<T>::_num_neurons * sizeof(T)
      ));

      checkCuda(cudaStreamSynchronize(dev_stream[dev][0]));

      //simulate BF load balancing
      #pragma omp barrier
    }
    identify<T><<<16, 512>>>(_dev_Y[dev][0], _dev_num_inputs[dev], Base<T>::_num_neurons, dev_results[dev]);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaStreamDestroy(dev_stream[dev][0]));
    checkCuda(cudaStreamDestroy(dev_stream[dev][1]));
    checkCuda(cudaSetDevice(0));
  }

  Base<T>::toc();
  Base<T>::log("Finish inference with ", Base<T>::duration(), " ms", "\n");
}

template <typename T>
void BF<T>::_non_empty_rows(const size_t dev, const size_t buff) {
  _dev_nerowsY[dev] = 0;
  for(size_t i = 0; i < _dev_num_inputs[dev]; ++i) {
    if((_dev_rlenY[dev][buff])[i] > 0) {
      (_dev_rowsY[dev][buff])[_dev_nerowsY[dev]++] = i;
    }
  }
}

template <typename T>
void BF<T>::_weight_alloc() {
  std::vector<int*> W(2, nullptr);
  for(size_t dev = 0; dev < Base<T>::_num_gpus; ++dev) {
    checkCuda(cudaSetDevice(dev));
    checkCuda(cudaMalloc(
      &W[0],
      Base<T>::_pp_wsize
    ));
    checkCuda(cudaMalloc(
      &W[1],
      Base<T>::_pp_wsize
    ));
    checkCuda(cudaMemcpy(
      W[0],
      Base<T>::_host_pinned_weight,
      Base<T>::_pp_wsize,
      cudaMemcpyHostToDevice
    ));
    _dev_W.emplace_back(W);
  }
  checkCuda(cudaSetDevice(0));
}

template <typename T>
void BF<T>::_input_alloc() {
  size_t ylen = Base<T>::_num_inputs * Base<T>::_num_neurons;
  size_t ysize = ylen * sizeof(T);
  size_t ry_size = Base<T>::_num_inputs * sizeof(int);

  for(int buff = 0; buff < 2; ++buff) {
    checkCuda(cudaMallocManaged(&_rowsY[buff], ry_size));
    checkCuda(cudaMallocManaged(&_rlenY[buff], ry_size));
    checkCuda(cudaMallocManaged(&_Y[buff], ysize));
    checkCuda(cudaMemset(_rowsY[buff], 0, ry_size));
  }
  checkCuda(cudaMemset(_rlenY[0], 1, ry_size));
  checkCuda(cudaMemset(_rlenY[1], 0, ry_size));

  //partition
  size_t each_partition = Base<T>::_num_inputs / Base<T>::_num_gpus;
  size_t remains = Base<T>::_num_inputs % Base<T>::_num_gpus;

  //use dev_Y, dev_rowsY,  dev_rlenY, and dev_nerowsY to record each GPUs' own data
  std::vector<int*> each_GPU_rowsY(2, nullptr);
  std::vector<int*> each_GPU_rlenY(2, nullptr);
  std::vector<T*> each_GPU_Y(2, nullptr);
  for(size_t dev = 0 ; dev < Base<T>::_num_gpus; ++dev) {
    for(int buff = 0; buff < 2; ++buff) {
      each_GPU_rowsY[buff] = _rowsY[buff] + dev * each_partition; 
      each_GPU_rlenY[buff] = _rlenY[buff] + dev * each_partition; 
      each_GPU_Y[buff] = _Y[buff] + dev * each_partition * Base<T>::_num_neurons;
    }
    _dev_rowsY.emplace_back(each_GPU_rowsY);
    _dev_rlenY.emplace_back(each_GPU_rlenY);
    _dev_Y.emplace_back(each_GPU_Y);
    _dev_nerowsY.emplace_back(each_partition);
    _dev_num_inputs.emplace_back(each_partition);
  }
  //last device handle remain inputs
  _dev_nerowsY[Base<T>::_num_gpus - 1] += remains;
  _dev_num_inputs[Base<T>::_num_gpus - 1] += remains;
  
  //find non-empty rows at the beginning
  //reindex rowsY
  for(size_t dev = 0; dev < Base<T>::_num_gpus; ++dev) {
    _non_empty_rows(dev, 0);
  }
  
  //Advise
  for(size_t dev = 0; dev < Base<T>::_num_gpus; ++dev) {
    for(int buff = 0; buff < 2; ++buff) {
      checkCuda(cudaMemAdvise(
        _dev_rowsY[dev][buff],
        _dev_num_inputs[dev] * sizeof(int),
        cudaMemAdviseSetPreferredLocation,
        dev 
      ));
      checkCuda(cudaMemAdvise(
        _dev_rlenY[dev][buff],
        _dev_num_inputs[dev] * sizeof(int),
        cudaMemAdviseSetPreferredLocation,
        dev 
      ));
      checkCuda(cudaMemAdvise(
        _dev_Y[dev][buff],
        _dev_num_inputs[dev] * Base<T>::_num_neurons * sizeof(T),
        cudaMemAdviseSetPreferredLocation,
        dev 
      ));
    }
  }

}

template <typename T>
void BF<T>::_result_alloc() {
  //final results allocation
  checkCuda(cudaMallocManaged(&_results, sizeof(int) * Base<T>::_num_inputs));
  checkCuda(cudaMemset(_results, 0, sizeof(int) * Base<T>::_num_inputs));
}

}// end of namespace snig ----------------------------------------------
