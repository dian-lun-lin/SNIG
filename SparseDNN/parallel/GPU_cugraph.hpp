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
  
  private:
    
    int* _h_pinned_weight;
    T _bias;
    int _num_neurons_per_layer;
    int _num_layers;

    int _max_nnz_per_layer;
    int _COL_BLK;
    int _pad;
    int _N_SLAB;

    void _infer_flatten_graph(
      T** Y,
      int** rowsY,
      int** rlenY,
      int** d_W,
      const int num_inputs
    ) const;

    void _infer_update_graph(
      T** Y,
      int** rowsY,
      int** rlenY,
      int** d_W,
      const int num_inputs
    ) const;

  public:

    GPUCugraph(
      const std::fs::path& weight_path,
      const T bias = -.3f,
      const int num_neurons_per_layer = 1024,
      const int num_layers = 120
    );

    ~GPUCugraph();

    int num_neurons_per_layer() const { return _num_neurons_per_layer; };
    int num_layers() const { return _num_layers; };
    T bias() const { return _bias; };

    Eigen::Matrix<int, Eigen::Dynamic, 1> infer(
      const std::fs::path& input_path,
      const int num_inputs,
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
  const int num_neurons_per_layer,
  const int num_layers
):
  _bias{bias},
  _num_neurons_per_layer{num_neurons_per_layer},
  _num_layers{num_layers}
{
  //get tuned shared memory size
  //num_neurons_per_layer must be divisible by shared memory (a.k.a. COL_BLK)
  //only for single GPU
  //only for double float
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  int max_num_per_block = props.sharedMemPerBlock / sizeof(T);
  if(num_neurons_per_layer <= max_num_per_block){
    _COL_BLK = num_neurons_per_layer;
  }
  else{
    int max_divisor = 2;
    while((num_neurons_per_layer % max_divisor != 0) || (max_num_per_block < (num_neurons_per_layer / max_divisor))){
      ++max_divisor;
    }
    _COL_BLK = num_neurons_per_layer / max_divisor;
  }

  std::cout << "Constructing a GPU parallel network.\n";
  std::cout << "Loading the weight.............." << std::flush;

  _N_SLAB = num_neurons_per_layer / _COL_BLK; 

  //_max_nnz_per_layer = find_max_nnz(weight_path, num_layers, num_neurons_per_layer);
  _max_nnz_per_layer = find_max_nnz_binary(weight_path, num_layers, num_neurons_per_layer);

  //handle aligned (only deal with double and float)
  if((num_neurons_per_layer * _N_SLAB + 1 + _max_nnz_per_layer) % sizeof(T) != 0){
    ++_pad;
  }

  checkCuda(cudaMallocHost(
    (void**)&_h_pinned_weight,
    (sizeof(int) * (num_neurons_per_layer * _N_SLAB + 1 + _max_nnz_per_layer + _pad) +
    sizeof(T) * _max_nnz_per_layer) * num_layers
  ));

  std::memset(_h_pinned_weight, 0, (sizeof(int) * (num_neurons_per_layer * _N_SLAB + 1 + _max_nnz_per_layer + _pad) + sizeof(T) * _max_nnz_per_layer) * num_layers);

  //read_weight<T>(weight_path, num_neurons_per_layer, _max_nnz_per_layer, num_layers, _COL_BLK, _N_SLAB, _pad, _h_pinned_weight);
  read_weight_binary<T>(
    weight_path,
    num_neurons_per_layer,
    _max_nnz_per_layer,
    num_layers,
    _N_SLAB,
    _pad,
    _h_pinned_weight
  );

  std::cout << "Done\n";
}

template <typename T>
GPUCugraph<T>::~GPUCugraph() {
  
  checkCuda(cudaFreeHost(_h_pinned_weight));
}

template <typename T>
Eigen::Matrix<int, Eigen::Dynamic, 1> GPUCugraph<T>::infer(
  const std::fs::path& input_path,
  const int num_inputs,
  const bool is_flatten
) const {

  std::cout << "Preprocessing.............................." << std::flush;

  int *d_W[2];
  checkCuda(cudaMalloc(
    &d_W[0],
    sizeof(int) * (_max_nnz_per_layer + _num_neurons_per_layer * _N_SLAB + 1 + _pad) + 
    sizeof(T) * (_max_nnz_per_layer)
  ));
  checkCuda(cudaMalloc(
    &d_W[1],
    sizeof(int) * (_max_nnz_per_layer + _num_neurons_per_layer * _N_SLAB + 1 + _pad) + 
    sizeof(T) * (_max_nnz_per_layer)
  ));
  checkCuda(cudaMemcpy(
    d_W[0],
    _h_pinned_weight,
    sizeof(int) * (_max_nnz_per_layer + _num_neurons_per_layer * _N_SLAB + 1 + _pad) + 
    sizeof(T) * (_max_nnz_per_layer),
    cudaMemcpyHostToDevice
  ));

  std::cout << "Done" << std::endl;

  std::cout << "Reading input.............................." << std::flush;

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

//issue: doesn't check boundary
  int nerowsY{0};
  read_input_binary<T>(input_path, Y[0], rlenY[0], rowsY[0], nerowsY);


  std::cout << "Done" << std::endl;

  std::cout << "Start inference............................" << std::flush;
  if(is_flatten){
    _infer_flatten_graph(Y, rowsY, rlenY, d_W, num_inputs);
  }
  else{
    _infer_update_graph(Y, rowsY, rlenY, d_W, num_inputs);
  }

  auto score = get_score<T>(Y[_num_layers % 2], num_inputs, _num_neurons_per_layer);


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
void GPUCugraph<T>::_infer_update_graph(
  T** Y,
  int** rowsY,
  int** rlenY,
  int** d_W,
  const int num_inputs
) const {
  dim3 threads(32, 32, 1);
  cudaStream_t stream_for_graph;
  cudaGraph_t graph;
  std::vector<cudaGraphNode_t> cpy_dependencies;
  std::vector<cudaGraphNode_t> infer_dependencies;
  cudaGraphNode_t memcpy_node, infer_node, memset_node;
  checkCuda(cudaStreamCreateWithFlags(&stream_for_graph, cudaStreamNonBlocking));

  checkCuda(cudaGraphCreate(&graph, 0));

  cudaMemcpy3DParms memcpy_params = {0};
  cudaKernelNodeParams infer_params = {0};
  cudaMemsetParams memset_params = {0};

  int cur_layer = 0;

  //memcpy
  memcpy_params.srcPtr        = make_cudaPitchedPtr(
    _h_pinned_weight + (cur_layer + 1) * (_num_neurons_per_layer * _N_SLAB + 1 + _max_nnz_per_layer + _pad + ((sizeof(T) / sizeof(int)) * _max_nnz_per_layer)),
    sizeof(int) * (_num_neurons_per_layer * _N_SLAB + 1 + _max_nnz_per_layer + _pad) + sizeof(T) * (_max_nnz_per_layer),
    (_num_neurons_per_layer * _N_SLAB + 1 + _pad + _max_nnz_per_layer) + (sizeof(T) / sizeof(int)) * (_max_nnz_per_layer),
    1
  );
  memcpy_params.srcArray      = NULL;
  memcpy_params.srcPos        = make_cudaPos(0, 0, 0);
  memcpy_params.dstPtr        = make_cudaPitchedPtr(
    d_W[(cur_layer + 1) % 2],
    sizeof(int) * (_num_neurons_per_layer * _N_SLAB + 1 + _max_nnz_per_layer + _pad) + sizeof(T) * (_max_nnz_per_layer),
    (_num_neurons_per_layer * _N_SLAB + 1 + _pad + _max_nnz_per_layer) + (sizeof(T) / sizeof(int)) * (_max_nnz_per_layer),
    1
  );
  memcpy_params.dstArray      = NULL;
  memcpy_params.dstPos        = make_cudaPos(0, 0, 0);
  memcpy_params.extent        = make_cudaExtent(
    sizeof(int) * (_max_nnz_per_layer + _num_neurons_per_layer * _N_SLAB + 1 + _pad) + 
    sizeof(T) * (_max_nnz_per_layer),
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
  T* valsW = (T*)(d_W[cur_layer % 2] + _num_neurons_per_layer * _N_SLAB + 1 + _max_nnz_per_layer);

  void* infer_args[12] = {
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
  auto begin = std::chrono::steady_clock::now();
  checkCuda(cudaGraphInstantiate(&exec, graph, NULL, NULL, 0));
  auto end = std::chrono::steady_clock::now();
  std::cout <<"Initial: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

  begin = std::chrono::steady_clock::now();
  checkCuda(cudaGraphLaunch(exec, stream_for_graph));
  checkCuda(cudaStreamSynchronize(stream_for_graph));
  //update
  for(cur_layer = 1; cur_layer < _num_layers; ++cur_layer) {

    if(cur_layer != _num_layers - 1) {
      memcpy_params.srcPtr = make_cudaPitchedPtr(
        _h_pinned_weight + (cur_layer + 1) * (_num_neurons_per_layer * _N_SLAB + 1 + _max_nnz_per_layer + _pad + ((sizeof(T) / sizeof(int)) * _max_nnz_per_layer)),
        sizeof(int) * (_num_neurons_per_layer * _N_SLAB + 1 + _max_nnz_per_layer + _pad) + sizeof(T) * (_max_nnz_per_layer),
        (_num_neurons_per_layer * _N_SLAB + 1 + _pad + _max_nnz_per_layer) + (sizeof(T) / sizeof(int)) * (_max_nnz_per_layer),
        1
      );

      memcpy_params.dstPtr = make_cudaPitchedPtr(
        d_W[(cur_layer + 1) % 2],
        sizeof(int) * (_num_neurons_per_layer * _N_SLAB + 1 + _max_nnz_per_layer + _pad) + sizeof(T) * (_max_nnz_per_layer),
        (_num_neurons_per_layer * _N_SLAB + 1 + _pad + _max_nnz_per_layer) + (sizeof(T) / sizeof(int)) * (_max_nnz_per_layer),
        1
      );

      checkCuda(cudaGraphExecMemcpyNodeSetParams(exec, memcpy_node, &memcpy_params));
    }

    roffW = d_W[cur_layer % 2];
    colsW = d_W[cur_layer % 2] + _num_neurons_per_layer * _N_SLAB + 1;
    valsW = (T*)(d_W[cur_layer % 2] + _num_neurons_per_layer * _N_SLAB + 1 + _max_nnz_per_layer);
    void* infer_args[12] = {
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

  end = std::chrono::steady_clock::now();
  std::cout << "Exec and update: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

  checkCuda(cudaGraphExecDestroy(exec));
  checkCuda(cudaGraphDestroy(graph));
  checkCuda(cudaStreamDestroy(stream_for_graph));

}


template <typename T>
void GPUCugraph<T>::_infer_flatten_graph(
  T** Y,
  int** rowsY,
  int** rlenY,
  int** d_W,
  const int num_inputs
) const {
  
  dim3 threads(32, 32, 1);
  cudaStream_t stream_for_graph;
  cudaGraph_t graph;
  std::vector<cudaGraphNode_t> cpy_dependencies;
  std::vector<cudaGraphNode_t> infer_dependencies;
  cudaGraphNode_t memcpy_node, infer_node, memset_node;
  checkCuda(cudaStreamCreateWithFlags(&stream_for_graph, cudaStreamNonBlocking));

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
    sizeof(int) * (_max_nnz_per_layer + _num_neurons_per_layer * _N_SLAB + 1 + _pad) + 
    sizeof(T) * (_max_nnz_per_layer),
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

  auto begin = std::chrono::steady_clock::now();
  for(int cur_layer = 0; cur_layer < _num_layers; ++cur_layer) {

    if(cur_layer != _num_layers - 1) {
      memcpy_params.srcPtr = make_cudaPitchedPtr(
        _h_pinned_weight + (cur_layer + 1) * (_num_neurons_per_layer * _N_SLAB + 1 + _max_nnz_per_layer + _pad + ((sizeof(T) / sizeof(int)) * _max_nnz_per_layer)),
        sizeof(int) * (_num_neurons_per_layer * _N_SLAB + 1 + _max_nnz_per_layer + _pad) + sizeof(T) * (_max_nnz_per_layer),
        (_num_neurons_per_layer * _N_SLAB + 1 + _pad + _max_nnz_per_layer) + (sizeof(T) / sizeof(int)) * (_max_nnz_per_layer),
        1
      );

      memcpy_params.dstPtr = make_cudaPitchedPtr(
        d_W[(cur_layer + 1) % 2],
        sizeof(int) * (_num_neurons_per_layer * _N_SLAB + 1 + _max_nnz_per_layer + _pad) + sizeof(T) * (_max_nnz_per_layer),
        (_num_neurons_per_layer * _N_SLAB + 1 + _pad + _max_nnz_per_layer) + (sizeof(T) / sizeof(int)) * (_max_nnz_per_layer),
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
    T* valsW = (T*)(d_W[cur_layer % 2] + _num_neurons_per_layer * _N_SLAB + 1 + _max_nnz_per_layer);

    void* infer_args[12] = {
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

  std::cout << std::endl;
  auto end = std::chrono::steady_clock::now();
  std::cout << "Add nodes:" << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

  cudaGraphExec_t exec;

  begin = std::chrono::steady_clock::now();
  checkCuda(cudaGraphInstantiate(&exec, graph, NULL, NULL, 0));
  end = std::chrono::steady_clock::now();
  std::cout <<"Initial: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

  begin = std::chrono::steady_clock::now();
  checkCuda(cudaGraphLaunch(exec, stream_for_graph));

  checkCuda(cudaStreamSynchronize(stream_for_graph));
  end = std::chrono::steady_clock::now();
  std::cout <<"Exec: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

  checkCuda(cudaGraphExecDestroy(exec));
  checkCuda(cudaGraphDestroy(graph));
  checkCuda(cudaStreamDestroy(stream_for_graph));

}

}// end of namespace sparse_dnn ----------------------------------------------
