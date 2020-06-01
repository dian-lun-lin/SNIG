#pragma once

#include <Eigen/Core>
#include <SNIG/utility/reader.hpp>
#include <SNIG/utility/matrix_format.h>
#include <SNIG/utility/cuda_error.hpp>
#include <SNIG/snig/kernel.hpp>
#include <SNIG/utility/scoring.hpp>
#include <SNIG/base/base.hpp>
#include <vector>
#include <queue>
#include <mutex>

namespace std {
  namespace fs = experimental::filesystem;  
}

namespace snig{


template <typename T>
class SNIGPipeline : public Base<T> {

  static_assert(
    std::is_same<T, float>::value || std::is_same<T, double>::value || std::is_same<T, half>::value,
    "data type must be either float or double"
  );
  
  private:

    size_t _batch_size;
    T* _source_Y;
    bool* _source_is_nonzero_row;
    std::vector<std::vector<T*> > _dev_Y;
    std::vector<std::vector<bool*> > _dev_is_nonzero_row;
    std::vector<int*> _dev_W;
    //record weight to delete
    std::vector<int*> _dev_record_W;

    //this pipeline partitiones _num_layers evenly to each GPU
    size_t _num_layers_per_gpu;

    size_t _batch_ylen;
    size_t _batch_ysize;
    int* _results;

    void _set_parameters(
      const size_t num_inputs,
      const size_t batch_size,
      const size_t num_gpus
    );

    void _preprocess(const std::fs::path& input_path);

    void  _infer();

    void _input_alloc();

    void _weight_alloc();

    void _result_alloc();

  public:

    SNIGPipeline(
      const dim3& threads,
      const std::fs::path& weight_path,
      const T bias = -.3f,
      const size_t num_neurons_per_layer = 1024,
      const size_t num_layers = 120
    );

    ~SNIGPipeline();

    Eigen::Matrix<int, Eigen::Dynamic, 1> infer(
      const std::fs::path& input_path,
      const size_t num_inputs,
      const size_t batch_size,
      const size_t num_gpus
    ) ;

};

// ----------------------------------------------------------------------------
// Definition of SNIGPipeline
// ----------------------------------------------------------------------------

template <typename T>
SNIGPipeline<T>::SNIGPipeline(
  const dim3& threads,
  const std::fs::path& weight_path,
  const T bias,
  const size_t num_neurons_per_layer,
  const size_t num_layers
):
  Base<T>(threads, weight_path, bias, num_neurons_per_layer, num_layers)
{
  Base<T>::log("Constructing SNIG engine using pipeline method......", "\n");
}

template <typename T>
SNIGPipeline<T>::~SNIGPipeline() {
  checkCuda(cudaFree(_source_Y));
  checkCuda(cudaFree(_source_is_nonzero_row));
  for(auto& W_in_dev : _dev_record_W) {
      checkCuda(cudaFree(W_in_dev));
  }
  for(auto& Y_in_dev : _dev_Y) {
      checkCuda(cudaFree(Y_in_dev[1]));
  }
  for(auto& rowsY_in_dev : _dev_is_nonzero_row) {
      checkCuda(cudaFree(rowsY_in_dev[1]));
  }
  checkCuda(cudaFree(_results));
}

template <typename T>
Eigen::Matrix<int, Eigen::Dynamic, 1> SNIGPipeline<T>::infer(
  const std::fs::path& input_path,
  const size_t num_inputs,
  const size_t batch_size,
  const size_t num_gpus
) {
  _set_parameters(
    num_inputs,
    batch_size,
    num_gpus
  );

  _preprocess(input_path);

  _infer();

  return arr_to_Eigen_int(_results, num_inputs);
}

template <typename T>
void SNIGPipeline<T>::_set_parameters(
  const size_t num_inputs,
  const size_t batch_size,
  const size_t num_gpus
) {
  Base<T>::log("Using ", num_gpus, " GPUs", "\n");
  Base<T>::log("Total input size : ", num_inputs, "\n");
  Base<T>::log("Input batch size : ", batch_size, "\n\n");

  Base<T>::_num_inputs = num_inputs;
  Base<T>::_num_gpus = num_gpus;
  _num_layers_per_gpu = Base<T>::_num_layers / Base<T>::_num_gpus;

  _batch_size = batch_size;
  _batch_ylen = _batch_size * Base<T>::_num_neurons;
  _batch_ysize = _batch_ylen * sizeof(T);

  _dev_W.reserve(Base<T>::_num_layers);
  _dev_Y.reserve(Base<T>::_num_gpus);
  _dev_is_nonzero_row.reserve(Base<T>::_num_gpus);
}

template <typename T>
void SNIGPipeline<T>::_preprocess(const std::fs::path& input_path) {
  Base<T>::log("Preprocessing...... ");
  Base<T>::tic();

  //weight allocation
  _weight_alloc();

  //input allocation
  _input_alloc();

  //final results allocation
  _result_alloc();

  //read input
  read_input_binary<T>(input_path, _source_Y);

  Base<T>::toc();
  Base<T>::log("Finish preprocessing with ", Base<T>::duration(), " ms", "\n");
}

template <typename T>
void SNIGPipeline<T>::_infer() {
  Base<T>::log("Start inference...... ", "\n");
  Base<T>::tic();

  for(size_t dev = 0; dev < Base<T>::_num_gpus; ++dev) {
    cudaSetDevice(dev);
    checkCuda(cudaMemcpy(
      _dev_record_W[dev],
      Base<T>::_host_pinned_weight + dev * _num_layers_per_gpu * Base<T>::_pp_wlen,
      Base<T>::_pp_wsize * _num_layers_per_gpu,
      cudaMemcpyHostToDevice
    ));
  }
  cudaSetDevice(0);

  size_t num_batches = Base<T>::_num_inputs / _batch_size;

  std::vector<int*> dev_results(Base<T>::_num_gpus, nullptr);

  std::vector<std::queue<size_t> > dev_start_batch(Base<T>::_num_gpus);
  std::vector<std::mutex> dev_que_mutex(Base<T>::_num_gpus);
  std::vector<std::condition_variable> dev_que_cv(Base<T>::_num_gpus);
  std::queue<size_t> first_dev_que;
  for(size_t i = 0; i < num_batches + 1; ++i) {
    first_dev_que.emplace(i * _batch_size);
  }
  dev_start_batch[0] = std::move(first_dev_que);

  dim3 grid_dim(_batch_size, Base<T>::_num_secs, 1);

  #pragma omp parallel num_threads(Base<T>::_num_gpus)
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

      if(beg_inputs == Base<T>::_num_inputs) {
        //notify next device to finish
        if(dev != Base<T>::_num_gpus - 1) {
          {
            std::unique_lock<std::mutex> lock(dev_que_mutex[dev]);
            dev_start_batch[dev + 1].emplace(Base<T>::_num_inputs);
            dev_que_cv[dev + 1].notify_one();
          }
        }
        //this device finished all batches
        stop = true;
        continue;
      }
      _dev_Y[dev][0] = _source_Y + beg_inputs * Base<T>::_num_neurons;
      _dev_is_nonzero_row[dev][0] = _source_is_nonzero_row + beg_inputs * Base<T>::_num_secs;
      dev_results[dev] = _results + beg_inputs;

      for(size_t cur_layer = dev * _num_layers_per_gpu; cur_layer < (dev + 1) * _num_layers_per_gpu; ++cur_layer) {
        int* roffw = _dev_W[cur_layer];
        int* colsw = _dev_W[cur_layer] + Base<T>::_num_neurons * Base<T>::_num_secs + 1;
        T* valsw = (T*)(_dev_W[cur_layer] + Base<T>::_p_w_index_len);

        snig_inference<T><<<grid_dim, Base<T>::_threads, sizeof(T) * Base<T>::_sec_size, infer_stream>>>(
          _dev_Y[dev][cur_layer % 2],
          _dev_is_nonzero_row[dev][cur_layer % 2],
          Base<T>::_sec_size,
          Base<T>::_num_secs,
          Base<T>::_num_neurons,
          roffw,
          colsw,
          valsw,
          Base<T>::_bias,
          _dev_is_nonzero_row[dev][(cur_layer + 1) % 2],
          _dev_Y[dev][(cur_layer + 1) % 2]
        );
        checkCuda(cudaStreamSynchronize(infer_stream));
      }
      if(dev != Base<T>::_num_gpus - 1) {
        //notify next device to infer
        {
          std::unique_lock<std::mutex> lock(dev_que_mutex[dev]);
          dev_start_batch[dev + 1].emplace(beg_inputs);
        }
        dev_que_cv[dev + 1].notify_one();
      }
      else {
        //last device identify
        identify<T><<<16, 512, 0, infer_stream>>>(_dev_Y[dev][0], _batch_size, Base<T>::_num_neurons, dev_results[dev]);
        checkCuda(cudaStreamSynchronize(infer_stream));
      }
    }
  }

  checkCuda(cudaSetDevice(0));

  Base<T>::toc();
  Base<T>::log("Finish inference with ", Base<T>::duration(), " ms", "\n");
}

template <typename T>
void SNIGPipeline<T>::_weight_alloc() {
  for(size_t dev = 0; dev < Base<T>::_num_gpus; ++dev) {
    cudaSetDevice(dev);
    int* W;
    checkCuda(cudaMallocManaged(
      &W,
      Base<T>::_pp_wsize * _num_layers_per_gpu
    ));
    _dev_record_W.emplace_back(W);
    for(size_t cur_layer = 0; cur_layer < _num_layers_per_gpu; ++cur_layer) {
      //record location of weight of each layer
      _dev_W.emplace_back(W + cur_layer * Base<T>::_pp_wlen);
    }
  }
  cudaSetDevice(0);
}

template <typename T>
void SNIGPipeline<T>::_input_alloc() {
  size_t ylen = Base<T>::_num_inputs * Base<T>::_num_neurons;
  size_t ysize = ylen * sizeof(T);

  checkCuda(cudaMallocManaged(&_source_Y, ysize));
  checkCuda(cudaMallocManaged(&_source_is_nonzero_row, sizeof(bool) * Base<T>::_num_inputs * Base<T>::_num_secs));
  checkCuda(cudaMemset(_source_is_nonzero_row, 1, sizeof(bool) * Base<T>::_num_inputs * Base<T>::_num_secs));

  std::vector<T*> Y{2, nullptr};
  std::vector<bool*> rowsY{2, nullptr};
  for(size_t dev = 0; dev < Base<T>::_num_gpus; ++dev) {
    cudaSetDevice(dev);
    checkCuda(cudaMalloc(&Y[1], _batch_ysize));
    checkCuda(cudaMalloc(&rowsY[1], sizeof(bool) * _batch_size * Base<T>::_num_secs));
    checkCuda(cudaMemset(Y[1], 0, _batch_ysize));
    checkCuda(cudaMemset(rowsY[1], 0, sizeof(bool) * _batch_size * Base<T>::_num_secs));
    _dev_Y.push_back(Y);
    _dev_is_nonzero_row.push_back(rowsY);
  }
  cudaSetDevice(0);
}

template <typename T>
void SNIGPipeline<T>::_result_alloc() {
  checkCuda(cudaMallocManaged(&_results, sizeof(int) * Base<T>::_num_inputs));
  checkCuda(cudaMemset(_results, 0, sizeof(int) * Base<T>::_num_inputs));
}

}// end of namespace snig ----------------------------------------------
