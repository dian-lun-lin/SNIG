#pragma once
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
class GPUCugraph {

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

    void _infer_flatterned_graph(
      T** Y,
      int** rowsY,
      int** rlenY,
      int** d_W,
      const size_t num_inputs
    ) const;

    void _infer_unflatterned_graph(
      T** Y,
      int** rowsY,
      int** rlenY,
      int** d_W,
      const size_t num_inputs
    ) const;

  public:

    GPUCugraph(
      const std::fs::path& weight_path,
      const T bias = -.3f,
      const size_t num_neurons_per_layer = 1024,
      const size_t num_layers = 120
    );

    ~GPUCugraph();

    size_t num_neurons_per_layer() const;
    size_t num_layers() const;

    Eigen::Matrix<int, Eigen::Dynamic, 1> infer(
      const std::fs::path& input_path,
      const size_t num_inputs,
      const bool is_flatten
    ) const;

};

// ----------------------------------------------------------------------------
// Definition of GPUCugraph
// ----------------------------------------------------------------------------

template <typename T>
GPUCugraph<T>::GPUCugraph(
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
GPUCugraph<T>::~GPUCugraph() {
  checkCuda(cudaFreeHost(_h_pinned_weight));
}

template <typename T>
size_t GPUCugraph<T>::num_neurons_per_layer() const {
   return _num_neurons_per_layer; 
}

template <typename T>
size_t GPUCugraph<T>::num_layers() const { 
  return _num_layers; 
}

template <typename T>
Eigen::Matrix<int, Eigen::Dynamic, 1> GPUCugraph<T>::infer(
  const std::fs::path& input_path,
  const size_t num_inputs,
  const bool is_flatterned
) const {

  std::cout << "Preprocessing.............................." << std::flush;
  auto pp_beg = std::chrono::steady_clock::now();

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
  int* rowsY[2];
  int* rlenY[2];

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

  if(is_flatterned) {
    _infer_flatterned_graph(Y, rowsY, rlenY, d_W, num_inputs);
  }
  else{
    _infer_unflatterned_graph(Y, rowsY, rlenY, d_W, num_inputs);
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
  checkCuda(cudaFree(d_W[0]));
  checkCuda(cudaFree(d_W[1]));

  return score;
}

template <typename T>
void GPUCugraph<T>::_infer_unflatterned_graph(
  T** Y,
  int** rowsY,
  int** rlenY,
  int** d_W,
  const size_t num_inputs
) const {
  dim3 threads(32, 32, 1);
  cudaStream_t stream_for_graph;
  cudaGraph_t graph;
  std::vector<cudaGraphNode_t> cpy_dependencies;
  std::vector<cudaGraphNode_t> infer_dependencies;
  cudaGraphNode_t memcpy_node, infer_node, memset_node;
  checkCuda(cudaStreamCreate(&stream_for_graph));
  checkCuda(cudaGraphCreate(&graph, 0));

  cudaMemcpy3DParms memcpy_params = {0};
  cudaKernelNodeParams infer_params = {0};
  cudaMemsetParams memset_params = {0};

  size_t cur_layer = 0;
  
  //memcpy
  memcpy_params.srcPtr        = make_cudaPitchedPtr(
                                  _h_pinned_weight + (cur_layer + 1) * (_pp_wlen),
                                  _pp_wsize,
                                  _pp_wlen,
                                  1
                                );
  memcpy_params.srcArray      = NULL;
  memcpy_params.srcPos        = make_cudaPos(0, 0, 0);
  memcpy_params.dstPtr        = make_cudaPitchedPtr(
                                  d_W[(cur_layer + 1) % 2],
                                  _pp_wsize,
                                  _pp_wlen,
                                  1
                                );
  memcpy_params.dstArray      = NULL;
  memcpy_params.dstPos        = make_cudaPos(0, 0, 0);
  memcpy_params.extent        = make_cudaExtent(
                                  _pp_wsize,
                                  1,
                                  1
                                );
  memcpy_params.kind          = cudaMemcpyHostToDevice;

  checkCuda(cudaGraphAddMemcpyNode(
    &memcpy_node,
    graph,
    cpy_dependencies.data(),
    cpy_dependencies.size(),
    &memcpy_params)
  );

  //infer
  int* roffW = d_W[cur_layer % 2];
  int* colsW = d_W[cur_layer % 2] + _num_neurons_per_layer * _N_SLAB + 1;
  T* valsW = (T*)(d_W[cur_layer % 2] + _p_w_index_len);
  
  void* infer_args[] = {
    (void*)&Y[cur_layer % 2],
    (void*)&rowsY[cur_layer % 2],
    (void*)&rlenY[cur_layer % 2],
    (void*)&_COL_BLK,
    (void*)&_N_SLAB,
    (void*)&_num_neurons_per_layer,
    (void*)&roffW,
    (void*)&colsW,
    (void*)&valsW,
    (void*)&_bias,
    (void*)&Y[(cur_layer + 1) % 2],
    (void*)&rlenY[(cur_layer + 1) % 2]
  };

  infer_params.func           = (void*)wo_host_inference<T>;
  infer_params.gridDim        = dim3(num_inputs, 1, 1);
  infer_params.blockDim       = threads;
  infer_params.sharedMemBytes = sizeof(T) * _COL_BLK;
  infer_params.extra          = NULL;
  infer_params.kernelParams   = infer_args;

  checkCuda(cudaGraphAddKernelNode(
    &infer_node,
    graph,
    infer_dependencies.data(),
    infer_dependencies.size(),
    &infer_params)
  );

  infer_dependencies.clear();
  infer_dependencies.push_back(infer_node);
  
  //memset
  memset_params.value         = 0;
  memset_params.pitch         = 0;
  memset_params.elementSize   = sizeof(float); // elementSize can be max 4 bytes
  memset_params.width         = _num_neurons_per_layer * num_inputs * (sizeof(T) / sizeof(float)); 
  memset_params.height        = 1;
  memset_params.dst           = (void*)Y[cur_layer % 2];

  checkCuda(cudaGraphAddMemsetNode(
    &memset_node,
    graph,
    infer_dependencies.data(),
    infer_dependencies.size(),
    &memset_params)
  );

  cudaGraphExec_t exec;
  checkCuda(cudaGraphInstantiate(&exec, graph, NULL, NULL, 0));

  checkCuda(cudaGraphLaunch(exec, stream_for_graph));
  checkCuda(cudaStreamSynchronize(stream_for_graph));
  //update
  for(cur_layer = 1; cur_layer < _num_layers; ++cur_layer) {

    if(cur_layer != _num_layers - 1) {
      memcpy_params.srcPtr = make_cudaPitchedPtr(
                               _h_pinned_weight + (cur_layer + 1) * _pp_wlen,
                               _pp_wsize,
                               _pp_wlen,
                               1
                              );

      memcpy_params.dstPtr = make_cudaPitchedPtr(
                               d_W[(cur_layer + 1) % 2],
                               _pp_wsize,
                               _pp_wlen,
                               1
                              );

      checkCuda(cudaGraphExecMemcpyNodeSetParams(exec, memcpy_node, &memcpy_params));
    }

    roffW = d_W[cur_layer % 2];
    colsW = d_W[cur_layer % 2] + _num_neurons_per_layer * _N_SLAB + 1;
    valsW = (T*)(d_W[cur_layer % 2] + _p_w_index_len);
    void* infer_args[] = {
      (void*)&Y[cur_layer % 2],
      (void*)&rowsY[cur_layer % 2],
      (void*)&rlenY[cur_layer % 2],
      (void*)&_COL_BLK,
      (void*)&_N_SLAB,
      (void*)&_num_neurons_per_layer,
      (void*)&roffW,
      (void*)&colsW,
      (void*)&valsW,
      (void*)&_bias,
      (void*)&Y[(cur_layer + 1) % 2],
      (void*)&rlenY[(cur_layer + 1) % 2]
    };
    infer_params.kernelParams = infer_args;
    checkCuda(cudaGraphExecKernelNodeSetParams(exec, infer_node, &infer_params));

    memset_params.dst = (void*)Y[cur_layer % 2];
    checkCuda(cudaGraphExecMemsetNodeSetParams(exec, memset_node, &memset_params));

    checkCuda(cudaGraphLaunch(exec, stream_for_graph));
    checkCuda(cudaStreamSynchronize(stream_for_graph));
  }

  checkCuda(cudaGraphExecDestroy(exec));
  checkCuda(cudaGraphDestroy(graph));
  checkCuda(cudaStreamDestroy(stream_for_graph));

}


template <typename T>
void GPUCugraph<T>::_infer_flatterned_graph(
  T** Y,
  int** rowsY,
  int** rlenY,
  int** d_W,
  const size_t num_inputs
) const {
  dim3 threads(32, 32, 1);
  cudaStream_t stream_for_graph;
  cudaGraph_t graph;
  std::vector<cudaGraphNode_t> cpy_dependencies;
  std::vector<cudaGraphNode_t> infer_dependencies;
  cudaGraphNode_t memcpy_node, infer_node, memset_node;
  checkCuda(cudaStreamCreate(&stream_for_graph));

  checkCuda(cudaGraphCreate(&graph, 0));

  cudaMemcpy3DParms memcpy_params = {0};
  cudaKernelNodeParams infer_params = {0};
  cudaMemsetParams memset_params = {0};

  //memcpy
  memcpy_params.srcArray      = NULL;
  memcpy_params.srcPos        = make_cudaPos(0, 0, 0);
  memcpy_params.dstArray      = NULL;
  memcpy_params.dstPos        = make_cudaPos(0, 0, 0);
  memcpy_params.extent        = make_cudaExtent(
                                  _pp_wsize,
                                  1,
                                  1
                                );
  memcpy_params.kind          = cudaMemcpyHostToDevice;

  //infer
  infer_params.func           = (void*)wo_host_inference<T>;
  infer_params.gridDim        = dim3(num_inputs, 1, 1);
  infer_params.blockDim       = threads;
  infer_params.sharedMemBytes = sizeof(T) * _COL_BLK;
  infer_params.extra          = NULL;
  
  //memset
  memset_params.value         = 0;
  memset_params.pitch         = 0;
  memset_params.elementSize   = sizeof(float); // elementSize can be max 4 bytes
  memset_params.width         = _num_neurons_per_layer * num_inputs * (sizeof(T) / sizeof(float)); 
  memset_params.height        = 1;

  for(size_t cur_layer = 0; cur_layer < _num_layers; ++cur_layer) {

    if(cur_layer != _num_layers - 1) {
      memcpy_params.srcPtr = make_cudaPitchedPtr(
                                _h_pinned_weight + (cur_layer + 1) * (_pp_wlen),
                                _pp_wsize,
                                _pp_wlen,
                                1
                              );

      memcpy_params.dstPtr = make_cudaPitchedPtr(
                                d_W[(cur_layer + 1) % 2],
                                _pp_wsize,
                                _pp_wlen,
                                1
                              );

      checkCuda(cudaGraphAddMemcpyNode(
        &memcpy_node,
        graph,
        cpy_dependencies.data(),
        cpy_dependencies.size(),
        &memcpy_params)
      );

    }

    int* roffW = d_W[cur_layer % 2];
    int* colsW = d_W[cur_layer % 2] + _num_neurons_per_layer * _N_SLAB + 1;
    T* valsW = (T*)(d_W[cur_layer % 2] + _p_w_index_len);

    void* infer_args[] = {
      (void*)&Y[cur_layer % 2],
      (void*)&rowsY[cur_layer % 2],
      (void*)&rlenY[cur_layer % 2],
      (void*)&_COL_BLK,
      (void*)&_N_SLAB,
      (void*)&_num_neurons_per_layer,
      (void*)&roffW,
      (void*)&colsW,
      (void*)&valsW,
      (void*)&_bias,
      (void*)&Y[(cur_layer + 1) % 2],
      (void*)&rlenY[(cur_layer + 1) % 2]
    };
    infer_params.kernelParams = infer_args;

    checkCuda(cudaGraphAddKernelNode(
      &infer_node,
      graph,
      infer_dependencies.data(),
      infer_dependencies.size(),
      &infer_params)
    );

    infer_dependencies.clear();
    infer_dependencies.push_back(infer_node);

    memset_params.dst = (void*)Y[cur_layer % 2];
    checkCuda(cudaGraphAddMemsetNode(
      &memset_node,
      graph,
      infer_dependencies.data(),
      infer_dependencies.size(),
      &memset_params)
    );

    infer_dependencies.clear();
    infer_dependencies.push_back(memcpy_node);
    infer_dependencies.push_back(memset_node);

    cpy_dependencies.clear();
    cpy_dependencies.push_back(memcpy_node);
    cpy_dependencies.push_back(infer_node);
  }


  cudaGraphExec_t exec;

  checkCuda(cudaGraphInstantiate(&exec, graph, NULL, NULL, 0));

  checkCuda(cudaGraphLaunch(exec, stream_for_graph));
  checkCuda(cudaStreamSynchronize(stream_for_graph));

  checkCuda(cudaGraphExecDestroy(exec));
  checkCuda(cudaGraphDestroy(graph));
  checkCuda(cudaStreamDestroy(stream_for_graph));

}

}// end of namespace sparse_dnn ----------------------------------------------
