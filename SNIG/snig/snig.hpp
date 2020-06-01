#pragma once

#include <Eigen/Core>
#include <taskflow/taskflow.hpp>
#include <SNIG/utility/reader.hpp>
#include <SNIG/utility/matrix_format.h>
#include <SNIG/utility/cuda_error.hpp>
#include <SNIG/snig/kernel.hpp>
#include <SNIG/utility/scoring.hpp>
#include <SNIG/base/base.hpp>
#include <vector>

namespace std {
  namespace fs = experimental::filesystem;  
}

namespace snig{

template <typename T>
class SNIG : public Base<T> {

  static_assert(
    std::is_same<T, float>::value || std::is_same<T, double>::value,
    "data type must be either float or double"
  );
  
  private:
    
    size_t _batch_size;
    size_t _num_weight_buffers;
    T* _source_Y;
    bool* _source_is_nonzero_row;
    std::vector<std::vector<T*> > _dev_Y;
    std::vector<std::vector<bool*> > _dev_is_nonzero_row;
    std::vector<std::vector<int*> > _dev_W;

    size_t _batch_ylen;
    size_t _batch_ysize;
    int* _results;

    void _set_parameters(
      const size_t num_inputs,
      const size_t batch_size,
      const size_t num_weight_buffers,
      const size_t num_gpus
    );

    void _preprocess(const std::fs::path& input_path);
  
    void  _infer();

    void _input_alloc();

    void _weight_alloc();

    void _result_alloc();

  public:

    SNIG(
      const dim3& threads,
      const std::fs::path& weight_path,
      const T bias = -.3f,
      const size_t num_neurons_per_layer = 1024,
      const size_t num_layers = 120
    );

    ~SNIG();

    Eigen::Matrix<int, Eigen::Dynamic, 1> infer(
      const std::fs::path& input_path,
      const size_t num_inputs,
      const size_t batch_size,
      const size_t num_buff,
      const size_t num_gpus
    );

};

// ----------------------------------------------------------------------------
// Definition of SNIG
// ----------------------------------------------------------------------------

template <typename T>
SNIG<T>::SNIG(
  const dim3& threads,
  const std::fs::path& weight_path,
  const T bias,
  const size_t num_neurons_per_layer,
  const size_t num_layers
):
  Base<T>(threads, weight_path, bias, num_neurons_per_layer, num_layers)
{
  Base<T>::log("Constructing SNIG engine......", "\n");
}

template <typename T>
SNIG<T>::~SNIG() {

  checkCuda(cudaFree(_source_Y));
  checkCuda(cudaFree(_source_is_nonzero_row));

  for(auto& W_in_dev : _dev_W) {
    for(auto& each_W : W_in_dev) {
      checkCuda(cudaFree(each_W));
    }
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
Eigen::Matrix<int, Eigen::Dynamic, 1> SNIG<T>::infer(
  const std::fs::path& input_path,
  const size_t num_inputs,
  const size_t batch_size,
  const size_t num_weight_buffers,
  const size_t num_gpus
) {
  
  Base<T>::log("Using ", num_gpus, " GPUs", "\n");
  Base<T>::log("Total input size : ", num_inputs, "\n");
  Base<T>::log("Input batch size : ", batch_size, "\n");
  Base<T>::log("Number of weight buffers : ", num_weight_buffers, "\n\n");

  _set_parameters(
    num_inputs,
    batch_size,
    num_weight_buffers,
    num_gpus
  );

  _preprocess(input_path);

  _infer();

  return arr_to_Eigen_int(_results, Base<T>::_num_inputs);
}

template <typename T>
void SNIG<T>::_set_parameters(
  const size_t num_inputs,
  const size_t batch_size,
  const size_t num_weight_buffers,
  const size_t num_gpus
) {
  Base<T>::_num_inputs = num_inputs;
  Base<T>::_num_gpus = num_gpus;
  _num_weight_buffers = num_weight_buffers;

  _batch_size = batch_size;
  _batch_ylen = _batch_size * Base<T>::_num_neurons;
  _batch_ysize = _batch_ylen * sizeof(T);

  _dev_W.reserve(Base<T>::_num_gpus);
  _dev_Y.reserve(Base<T>::_num_gpus);
  _dev_is_nonzero_row.reserve(Base<T>::_num_gpus);
}

template <typename T>
void SNIG<T>::_preprocess(const std::fs::path& input_path) {
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
void SNIG<T>::_infer() {
  Base<T>::log("Start inference...... ", "\n");
  Base<T>::tic();

  //Use taskflow and cudaGraph to implement task graph
  tf::Taskflow taskflow("SNIG");
  tf::Executor executor;
  std::vector<tf::Task> first_fetchs;
  std::vector<tf::Task> cudaflows;
  std::vector<tf::Task> fetchs;
  first_fetchs.reserve(Base<T>::_num_gpus);
  cudaflows.reserve(Base<T>::_num_gpus);
  fetchs.reserve(Base<T>::_num_gpus);

  std::atomic<size_t> finished_inputs{0};
  std::vector<int*> dev_results(Base<T>::_num_gpus, nullptr);

  dim3 grid_dim(_batch_size, Base<T>::_num_secs, 1);

  tf::Task start = taskflow.emplace([](){
  }).name("start");

  for(size_t dev = 0; dev < Base<T>::_num_gpus; ++dev) {
    first_fetchs.emplace_back(taskflow.emplace([&, dev](){
      cudaSetDevice(dev);
      int is_end = 1;
      size_t beg_inputs = finished_inputs.fetch_add(_batch_size);
      if(beg_inputs < Base<T>::_num_inputs) {
        _dev_Y[dev][0] = _source_Y + beg_inputs * Base<T>::_num_neurons;
        _dev_is_nonzero_row[dev][0] = _source_is_nonzero_row + beg_inputs * Base<T>::_num_secs;
        dev_results[dev] = _results + beg_inputs;
        checkCuda(cudaMemPrefetchAsync(_dev_Y[dev][0], _batch_ysize, dev, NULL));
        checkCuda(cudaMemPrefetchAsync(_dev_is_nonzero_row[dev][0], sizeof(bool) * _batch_size * Base<T>::_num_secs, dev, NULL));
        checkCuda(cudaMemPrefetchAsync(dev_results[dev], sizeof(int) * _batch_size, dev, NULL));
        is_end = 0;
      }
      return is_end;
    }).name("first_fetch"));

    cudaflows.emplace_back(taskflow.emplace([&, dev](tf::cudaFlow& cf){
      cf.device(dev);
      std::vector<tf::cudaTask> weight_copies;
      std::vector<tf::cudaTask> infers;
      weight_copies.reserve(Base<T>::_num_layers);
      infers.reserve(Base<T>::_num_layers);

      for(size_t cur_layer = 0; cur_layer < Base<T>::_num_layers; cur_layer += _num_weight_buffers) {
        for(size_t k = 0; k < _num_weight_buffers; ++k) {
          //tasks of cudaflow
          weight_copies.emplace_back(cf.copy(
            _dev_W[dev][k],
            Base<T>::_host_pinned_weight + (cur_layer + k) * Base<T>::_pp_wlen,
            Base<T>::_pp_wlen
          ).name("weight_copy"));

          // transformed CSC weight matrix equals to CSR with exchanged row and col
          int* col_w = _dev_W[dev][k];
          int* row_w = _dev_W[dev][k] + Base<T>::_num_neurons * Base<T>::_num_secs + 1;
          T* val_w = (T*)(_dev_W[dev][k] + Base<T>::_p_w_index_len);
          infers.emplace_back(cf.kernel(
            grid_dim,
            Base<T>::_threads,
            sizeof(T) * Base<T>::_sec_size,
            snig_inference<T>,
            _dev_Y[dev][k % 2],
            _dev_is_nonzero_row[dev][k % 2],
            Base<T>::_sec_size,
            Base<T>::_num_secs,
            Base<T>::_num_neurons,
            col_w,
            row_w,
            val_w,
            Base<T>::_bias,
            _dev_is_nonzero_row[dev][(k + 1) % 2],
            _dev_Y[dev][(k + 1) % 2]
          ).name("Inference"));
        }
      }

      // TODO: consider parameterizing the thread numbers
      tf::cudaTask ident = cf.kernel(16, 512, 0, identify<T>, _dev_Y[dev][0], _batch_size, Base<T>::_num_neurons, dev_results[dev]);

      //dependencies of cudaflow
      for(size_t cur_layer = 0; cur_layer < Base<T>::_num_layers; ++cur_layer) {
        weight_copies[cur_layer].precede(infers[cur_layer]);

        if(cur_layer + _num_weight_buffers < Base<T>::_num_layers) {
          infers[cur_layer].precede(weight_copies[cur_layer + _num_weight_buffers]);
        }
        if(cur_layer + 1 < Base<T>::_num_layers) {
          infers[cur_layer].precede(infers[cur_layer + 1]);
        }
      }
      infers[Base<T>::_num_layers - 1].precede(ident);
    }).name("GPU"));

    fetchs.emplace_back(taskflow.emplace([&, dev](){
      cudaSetDevice(dev);
      int is_end = 1;
      size_t beg_inputs = finished_inputs.fetch_add(_batch_size);
      if(beg_inputs < Base<T>::_num_inputs) {
        _dev_Y[dev][0] = _source_Y + beg_inputs * Base<T>::_num_neurons;
        _dev_is_nonzero_row[dev][0] = _source_is_nonzero_row + beg_inputs * Base<T>::_num_secs;
        dev_results[dev] = _results + beg_inputs;
        checkCuda(cudaMemPrefetchAsync(_dev_Y[dev][0], _batch_ysize, dev, NULL));
        checkCuda(cudaMemPrefetchAsync(_dev_is_nonzero_row[dev][0], sizeof(bool) * _batch_size * Base<T>::_num_secs, dev, NULL));
        checkCuda(cudaMemPrefetchAsync(dev_results[dev], sizeof(int) * _batch_size, dev, NULL));
        is_end = 0;
      }
      return is_end;
    }).name("fetch"));

  }

  tf::Task stop = taskflow.emplace([](){}).name("stop");

  //dependencies of taskflow
  for(size_t dev = 0; dev < Base<T>::_num_gpus; ++dev) {
    start.precede(first_fetchs[dev]);
    first_fetchs[dev].precede(cudaflows[dev], stop);
    cudaflows[dev].precede(fetchs[dev]);
    fetchs[dev].precede(cudaflows[dev], stop);
  }
  
  executor.run(taskflow).wait();

  checkCuda(cudaSetDevice(0));

  Base<T>::toc();
  Base<T>::log("Finish inference with ", Base<T>::duration(), " ms", "\n");
}

template <typename T>
void SNIG<T>::_weight_alloc() {
  std::vector<int*> W(_num_weight_buffers, nullptr);

  for(size_t dev = 0; dev < Base<T>::_num_gpus; ++dev) {
    cudaSetDevice(dev);
    for(auto& each_W : W) {
      checkCuda(cudaMalloc(
        &each_W,
        Base<T>::_pp_wsize
      ));
    }
    _dev_W.push_back(W);
  }
  cudaSetDevice(0);
}

template <typename T>
void SNIG<T>::_input_alloc() {
  size_t ylen = Base<T>::_num_inputs *  Base<T>::_num_neurons;
  size_t ysize = ylen * sizeof(T);

  checkCuda(cudaMallocManaged(&_source_Y, ysize));
  checkCuda(cudaMallocManaged(&_source_is_nonzero_row, sizeof(bool) * Base<T>::_num_inputs * Base<T>::_num_secs));
  checkCuda(cudaMemset(_source_is_nonzero_row, 1, sizeof(bool) * Base<T>::_num_inputs * Base<T>::_num_secs));

  std::vector<T*> Y{2, nullptr};
  std::vector<bool*> is_nonzero_row{2, nullptr};
  for(size_t dev = 0; dev < Base<T>::_num_gpus; ++dev) {
    cudaSetDevice(dev);
    checkCuda(cudaMalloc(&Y[1], _batch_ysize));
    checkCuda(cudaMalloc(&is_nonzero_row[1], sizeof(bool) * _batch_size * Base<T>::_num_secs));
    checkCuda(cudaMemset(Y[1], 0, _batch_ysize));
    checkCuda(cudaMemset(is_nonzero_row[1], 0, sizeof(bool) * _batch_size * Base<T>::_num_secs));
    _dev_Y.push_back(Y);
    _dev_is_nonzero_row.push_back(is_nonzero_row);
  }
  cudaSetDevice(0);
}

template <typename T>
void SNIG<T>::_result_alloc() {
  checkCuda(cudaMallocManaged(&_results, sizeof(int) * Base<T>::_num_inputs));
  checkCuda(cudaMemset(_results, 0, sizeof(int) * Base<T>::_num_inputs));
}

}// end of namespace snig ----------------------------------------------
