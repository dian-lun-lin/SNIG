#pragma once
#include <Eigen/Dense>
#include <SparseDNN/utility/reader.hpp>
#include <SparseDNN/utility/matrix_format.h>
#include <SparseDNN/utility/cuda_error.hpp>
#include <SparseDNN/utility/scoring.hpp>
#include <SparseDNN/parallel/task.hpp>
#include <chrono>
#include <vector>
#include <tuple>
#include <thread>

namespace std {
  namespace fs = experimental::filesystem;  
}

namespace sparse_dnn{


template <typename T>
class GPUDecompMulti {

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

    void _infer_flatterned_graph(
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
      const size_t batch_ysize
    ) const;

    std::tuple<cudaGraph_t, std::vector<cudaGraphNode_t>, cudaKernelNodeParams> _flatterned_graph_manual(
      std::vector<T*>& Y,
      std::vector<bool*>& rowsY,
      std::vector<int*>& d_W,
      const size_t num_inputs,
      const size_t num_buff,
      const size_t batch_size,
      const size_t batch_ylen,
      const size_t batch_ysize
    ) const;

    void _graph_launch(
      const cudaGraphExec_t& exec,
      const cudaStream_t& stream_for_graph
    ) const;

    void _set_graph_param(
      const std::vector<int*>& d_W,
      const std::vector<T*>& Y,
      const std::vector<bool*>& rowsY,
      const size_t num_buff,
      cudaGraphExec_t& exec,
      std::vector<cudaGraphNode_t>& infer_nodes,
      cudaKernelNodeParams& infer_param
    ) const;

  public:

    GPUDecompMulti(
      const std::fs::path& weight_path,
      const T bias = -.3f,
      const size_t num_neurons_per_layer = 1024,
      const size_t num_layers = 120
    );

    ~GPUDecompMulti();

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
// Definition of GPUDecompMulti
// ----------------------------------------------------------------------------

template <typename T>
GPUDecompMulti<T>::GPUDecompMulti(
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
GPUDecompMulti<T>::~GPUDecompMulti() {
  checkCuda(cudaFreeHost(_h_pinned_weight));
}

template <typename T>
size_t GPUDecompMulti<T>::num_neurons_per_layer() const {
   return _num_neurons_per_layer; 
}

template <typename T>
size_t GPUDecompMulti<T>::num_layers() const { 
  return _num_layers; 
}

template <typename T>
Eigen::Matrix<int, Eigen::Dynamic, 1> GPUDecompMulti<T>::infer(
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
  bool* source_rowsY;
  checkCuda(cudaMallocManaged(&source_Y, ysize));
  checkCuda(cudaMallocManaged(&source_rowsY, sizeof(bool) * num_inputs));
  checkCuda(cudaMemset(source_rowsY, 1, sizeof(bool) * num_inputs));

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

  std::cout << "Start inference............................" << std::flush;
  auto exec_beg = std::chrono::steady_clock::now();

  _infer_flatterned_graph(source_Y, source_rowsY, dev_Y, dev_rowsY, dev_W, num_inputs, num_buff, num_dev, batch_size, batch_ylen, batch_ysize);

  auto exec_end = std::chrono::steady_clock::now();
  std::cout << "finished execution with "
            << std::chrono::duration_cast<std::chrono::milliseconds>(exec_end - exec_beg).count()
            << "ms"
            << std::endl;

  std::cout << "Start scoring..............................." << std::flush;
  auto score_beg = std::chrono::steady_clock::now();

  auto score = get_score<T>(source_Y, num_inputs, _num_neurons_per_layer);

  auto score_end = std::chrono::steady_clock::now();
  std::cout << "finished scoring with "
            << std::chrono::duration_cast<std::chrono::milliseconds>(score_end - score_beg).count()
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

  return score;
}

template <typename T>
void GPUDecompMulti<T>::_infer_flatterned_graph(
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
  const size_t batch_ysize
) const {
  std::vector<cudaStream_t> stream_for_graphs(num_dev);
  std::vector<cudaGraphExec_t> executors(num_dev);
  std::vector<cudaGraph_t> graphs;
  graphs.reserve(num_dev);
  //nodes that change their params each iteration
  std::vector<std::vector<cudaGraphNode_t> > dev_infer_nodes;
  std::vector<cudaKernelNodeParams> dev_infer_params;
  dev_infer_nodes.reserve(num_dev);
  dev_infer_params.reserve(num_dev);

  //create graph to each GPU
  cudaGraph_t graph;
  std::vector<cudaGraphNode_t> infer_nodes;
  cudaKernelNodeParams infer_param = {0};
  for(size_t dev = 0; dev < num_dev; ++dev) {
    checkCuda(cudaSetDevice(dev));
    dev_Y[dev][0] = source_Y;
    dev_rowsY[dev][0] = source_rowsY;
    std::tie(graph, infer_nodes, infer_param) = _flatterned_graph_manual(
      dev_Y[dev],
      dev_rowsY[dev],
      dev_W[dev],
      num_inputs,
      num_buff,
      batch_size,
      batch_ylen,
      batch_ysize
    );
    graphs.emplace_back(std::move(graph));
    dev_infer_nodes.emplace_back(std::move(infer_nodes));
    dev_infer_params.emplace_back(std::move(infer_param));
  }

  //initialize graph to each GPU
  for(size_t dev = 0; dev < num_dev; ++dev) {
    checkCuda(cudaSetDevice(dev));
    checkCuda(cudaStreamCreate(&stream_for_graphs[dev]));
    checkCuda(cudaGraphInstantiate(&executors[dev], graphs[dev], NULL, NULL, 0));
  }
  checkCuda(cudaSetDevice(0));

  //launch graph on each GPU
  //each thread manage one GPU
  std::atomic<size_t> finished_inputs{0};
  std::vector<std::thread> thread_for_graphs;
  thread_for_graphs.reserve(num_dev);
  for(size_t dev = 0; dev < num_dev; ++dev) {
    thread_for_graphs.emplace_back([&, dev](){
      checkCuda(cudaSetDevice(dev));
      size_t beg_inputs = finished_inputs.fetch_add(batch_size);
      while(beg_inputs < num_inputs) {
        dev_Y[dev][0] = source_Y + beg_inputs * _num_neurons_per_layer;
        dev_rowsY[dev][0] = source_rowsY + beg_inputs;
        checkCuda(cudaMemPrefetchAsync(dev_Y[dev][0], batch_ysize, dev, NULL));
        checkCuda(cudaMemPrefetchAsync(dev_rowsY[dev][0], sizeof(bool) * batch_size, dev, NULL));
        _set_graph_param(
          dev_W[dev],
          dev_Y[dev],
          dev_rowsY[dev],
          num_buff,
          executors[dev],
          dev_infer_nodes[dev],
          dev_infer_params[dev]
        );
        _graph_launch(executors[dev],stream_for_graphs[dev]);
        beg_inputs = finished_inputs.fetch_add(batch_size);
      }
    });
  }

  for(auto& each_thread : thread_for_graphs) {
    each_thread.join();
  }

  for(auto& graph : graphs) {
    checkCuda(cudaGraphDestroy(graph));
  }
  for(auto& exec : executors) {
    checkCuda(cudaGraphExecDestroy(exec));
  }
  for(auto& stream_for_graph : stream_for_graphs) {
    checkCuda(cudaStreamDestroy(stream_for_graph));
  }
}

template <typename T>
  std::tuple<cudaGraph_t, std::vector<cudaGraphNode_t>, cudaKernelNodeParams> GPUDecompMulti<T>::_flatterned_graph_manual(
  std::vector<T*>& Y,
  std::vector<bool*>& rowsY,
  std::vector<int*>& d_W,
  const size_t num_inputs,
  const size_t num_buff,
  const size_t batch_size,
  const size_t batch_ylen,
  const size_t batch_ysize
) const {
  dim3 threads(2, 512, 1);
  cudaGraph_t graph;

  std::vector<std::vector<cudaGraphNode_t> >infer_dependencies(num_buff);
  std::vector<std::vector<cudaGraphNode_t> >cpy_dependencies(num_buff);
  std::vector<cudaGraphNode_t> w_memcpy_nodes(num_buff);
  std::vector<cudaGraphNode_t> infer_nodes(num_buff);
  //record nodes that args would be update after this function
  std::vector<cudaGraphNode_t> infer_record_nodes;
  cudaKernelNodeParams infer_record_param = {0};
  infer_record_nodes.reserve(_num_layers);

  checkCuda(cudaGraphCreate(&graph, 0));

  cudaMemcpy3DParms w_memcpy_params = {0};
  cudaKernelNodeParams infer_params = {0};

  //weight memcpy
  w_memcpy_params.srcArray      = NULL;
  w_memcpy_params.srcPos        = make_cudaPos(0, 0, 0);
  w_memcpy_params.dstArray      = NULL;
  w_memcpy_params.dstPos        = make_cudaPos(0, 0, 0);
  w_memcpy_params.extent        = make_cudaExtent(
                                  _pp_wsize,
                                  1,
                                  1
                                );
  w_memcpy_params.kind          = cudaMemcpyHostToDevice;

  //infer
  infer_params.func           = (void*)wo_host_inference_test_2<T>;
  infer_params.gridDim        = dim3(batch_size, 1, 1);
  infer_params.blockDim       = threads;
  infer_params.sharedMemBytes = sizeof(T) * _COL_BLK;
  infer_params.extra          = NULL;
  
  for(size_t cur_layer = 0; cur_layer < _num_layers; cur_layer += num_buff) {
    for(size_t k = 0; k < num_buff; ++k) {

      w_memcpy_params.srcPtr = make_cudaPitchedPtr(
                                _h_pinned_weight + (k + cur_layer) * (_pp_wlen),
                                _pp_wsize,
                                _pp_wlen,
                                1
                              );

      w_memcpy_params.dstPtr = make_cudaPitchedPtr(
                                d_W[k],
                                _pp_wsize,
                                _pp_wlen,
                                1
                              );

      checkCuda(cudaGraphAddMemcpyNode(
        &w_memcpy_nodes[k],
        graph,
        cpy_dependencies[k].data(),
        cpy_dependencies[k].size(),
        &w_memcpy_params)
      );
      cpy_dependencies[k].clear();
      cpy_dependencies[k].push_back(w_memcpy_nodes[k]);

      infer_dependencies[k].push_back(w_memcpy_nodes[k]);
      
      int* roffw = d_W[k];
      int* colsw = d_W[k] + _num_neurons_per_layer * _N_SLAB + 1;
      T* valsw = (T*)(d_W[k] + _p_w_index_len);

      void* infer_args[] = {
        (void*)&Y[k % 2],
        (void*)&rowsY[k % 2],
        (void*)&_COL_BLK,
        (void*)&_N_SLAB,
        (void*)&_num_neurons_per_layer,
        (void*)&roffw,
        (void*)&colsw,
        (void*)&valsw,
        (void*)&_bias,
        (void*)&rowsY[(k + 1) % 2],
        (void*)&Y[(k + 1) % 2]
      };

      infer_params.kernelParams = infer_args;

      checkCuda(cudaGraphAddKernelNode(
        &infer_nodes[k],
        graph,
        infer_dependencies[k].data(),
        infer_dependencies[k].size(),
        &infer_params)
      );

      infer_dependencies[k].clear();
      if(k < num_buff - 1) {
        infer_dependencies[k + 1].push_back(infer_nodes[k]);
      }
      cpy_dependencies[k].push_back(infer_nodes[k]);

      infer_record_nodes.push_back(infer_nodes[k]);
    }
    infer_dependencies[0].push_back(infer_nodes[num_buff - 1]);
  }
  infer_record_param = infer_params;
  return std::tie(graph, infer_record_nodes, infer_record_param);
}

template <typename T>
void GPUDecompMulti<T>::_set_graph_param(
  const std::vector<int*>& d_W,
  const std::vector<T*>& Y,
  const std::vector<bool*>& rowsY,
  const size_t num_buff,
  cudaGraphExec_t& exec,
  std::vector<cudaGraphNode_t>& infer_nodes,
  cudaKernelNodeParams& infer_param
) const {
  for(size_t cur_layer = 0; cur_layer < _num_layers; ++cur_layer) {
    int* roffW = d_W[cur_layer % num_buff];
    int* colsW = d_W[cur_layer % num_buff] + _num_neurons_per_layer * _N_SLAB + 1;
    T* valsW = (T*)(d_W[cur_layer % num_buff] + _p_w_index_len);

    void* args[] = {
      (void*)&Y[cur_layer % 2],
      (void*)&rowsY[cur_layer % 2],
      (void*)&_COL_BLK,
      (void*)&_N_SLAB,
      (void*)&_num_neurons_per_layer,
      (void*)&roffW,
      (void*)&colsW,
      (void*)&valsW,
      (void*)&_bias,
      (void*)&rowsY[(cur_layer + 1) % 2],
      (void*)&Y[(cur_layer + 1) % 2]
    };
    infer_param.kernelParams = args;
    checkCuda(cudaGraphExecKernelNodeSetParams(exec, infer_nodes[cur_layer], &infer_param));
  }
}

template <typename T>
void GPUDecompMulti<T>::_graph_launch(
  const cudaGraphExec_t& exec,
  const cudaStream_t& stream_for_graph
) const {
  checkCuda(cudaGraphLaunch(exec, stream_for_graph));
  checkCuda(cudaStreamSynchronize(stream_for_graph));
}

}// end of namespace sparse_dnn ----------------------------------------------
