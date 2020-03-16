#pragma once
#include <Eigen/Dense>
#include <SparseDNN/utility/reader.hpp>
#include <SparseDNN/utility/matrix_format.h>
#include <SparseDNN/utility/cuda_error.hpp>
#include <SparseDNN/parallel/task.hpp>
#include <SparseDNN/utility/scoring.hpp>

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


    //void CUDART_CB _non_empty_rows(
      //const int *rlen,
      //const int num_inputs,
      //int* nnz_rows,
      //int& nnz
    //) const;

    //void _cuda_graph_manual(
    //) const;

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

    Eigen::Matrix<int, Eigen::Dynamic, 1> infer_stream(
      const std::fs::path& input_path,
      const int num_inputs
    ) const;

    Eigen::Matrix<int, Eigen::Dynamic, 1> infer_graph_manual(
      const std::fs::path& input_path,
      const int num_inputs
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
//issue max_divisor is large?
    int max_divisor = 2;
    while((num_neurons_per_layer % max_divisor != 0) || (max_num_per_block < (num_neurons_per_layer / max_divisor))){
      ++max_divisor;
    }
    _COL_BLK = num_neurons_per_layer / max_divisor;
  }

  std::cout << "Constructing a GPU parallel network.\n";
  std::cout << "Loading the weight.............." << std::flush;

  int N_SLAB = num_neurons_per_layer / _COL_BLK; 

  _max_nnz_per_layer = find_max_nnz(weight_path, num_layers, num_neurons_per_layer);

  //handle aligned (only deal with double and float)
  if((num_neurons_per_layer * N_SLAB + 1 + _max_nnz_per_layer) % sizeof(T) != 0){
    ++_pad;
  }

  checkCuda(cudaMallocHost(
    (void**)&_h_pinned_weight,
    (sizeof(int) * (num_neurons_per_layer * N_SLAB + 1 + _max_nnz_per_layer + _pad) +
    sizeof(T) * _max_nnz_per_layer) * num_layers
  ));

  std::memset(_h_pinned_weight, 0, (sizeof(int) * (num_neurons_per_layer * N_SLAB + 1 + _max_nnz_per_layer + _pad) + sizeof(T) * _max_nnz_per_layer) * num_layers);

  read_weight<T>(weight_path, num_neurons_per_layer, _max_nnz_per_layer, num_layers, _COL_BLK, N_SLAB, _pad, _h_pinned_weight);

  std::cout << "Done\n";
}

template <typename T>
GPUCugraph<T>::~GPUCugraph() {
  
  checkCuda(cudaFreeHost(_h_pinned_weight));
}

template <typename T>
Eigen::Matrix<int, Eigen::Dynamic, 1> GPUCugraph<T>::infer_stream(
  const std::fs::path& input_path,
  const int num_inputs
) const {

  std::cout << "Preprocessing.............................." << std::flush;

  int N_SLAB = _num_neurons_per_layer / _COL_BLK; 

  int *d_W[2];
  checkCuda(cudaMalloc(
    &d_W[0],
    sizeof(int) * (_max_nnz_per_layer + _num_neurons_per_layer * N_SLAB + 1 + _pad) + 
    sizeof(T) * (_max_nnz_per_layer)
  ));
  checkCuda(cudaMalloc(
    &d_W[1],
    sizeof(int) * (_max_nnz_per_layer + _num_neurons_per_layer * N_SLAB + 1 + _pad) + 
    sizeof(T) * (_max_nnz_per_layer)
  ));
  checkCuda(cudaMemcpy(
    d_W[0],
    _h_pinned_weight,
    sizeof(int) * (_max_nnz_per_layer + _num_neurons_per_layer * N_SLAB + 1 + _pad) + 
    sizeof(T) * (_max_nnz_per_layer),
    cudaMemcpyHostToDevice
  ));

  std::cout << "Done" << std::endl;

  std::cout << "Reading input.............................." << std::flush;

  T* Y[2];  
  int *rowsY[2], *rlenY[2];

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
  read_input<T>(input_path, num_inputs, _num_neurons_per_layer, Y[0], rowsY[0], nerowsY);

  std::cout << "Done" << std::endl;

  std::cout << "Start inference............................" << std::flush;

//issue: how many threads
  dim3 threads(32, 32, 1);

  cudaStream_t stream[2], stream_for_graph;
  checkCuda(cudaStreamCreateWithFlags(&stream[0], cudaStreamNonBlocking));
  checkCuda(cudaStreamCreateWithFlags(&stream[1], cudaStreamNonBlocking));
  checkCuda(cudaStreamCreateWithFlags(&stream_for_graph, cudaStreamNonBlocking));

  cudaGraph_t graph;
  cudaEvent_t event_cpy, event_infer, fork_event;
  cudaEventCreate (&event_cpy);
  cudaEventCreate (&event_infer);
  cudaEventCreate (&fork_event);
  HostFuncArgs h_func_args = {0};
  int cur_layer = 0;
  h_func_args.cur_layer = &cur_layer; 
  h_func_args.num_inputs = num_inputs;
  h_func_args.rlenY = rlenY;
  h_func_args.rowsY = rowsY;
  h_func_args.nerowsY = &nerowsY;

  checkCuda(cudaStreamBeginCapture(stream[1],  cudaStreamCaptureModeGlobal));

  checkCuda(cudaEventRecord(fork_event, stream[1]));
  checkCuda(cudaStreamWaitEvent(stream[0], fork_event, 0));

  for(int k = 0; k < _num_layers - 1; ++k){

    checkCuda(cudaMemcpyAsync(
      d_W[(k + 1) % 2],
      _h_pinned_weight + (k + 1) * (_num_neurons_per_layer * N_SLAB + 1 + _max_nnz_per_layer + _pad + ((sizeof(T) / sizeof(int)) * _max_nnz_per_layer)),
      sizeof(int) * (_num_neurons_per_layer * N_SLAB + 1 + _max_nnz_per_layer + _pad) + sizeof(T) * (_max_nnz_per_layer),
      cudaMemcpyHostToDevice,
      stream[0]
    ));
    checkCuda(cudaEventRecord(event_cpy, stream[0]));

    baseline_inference<T><<<nerowsY, threads, sizeof(T) * _COL_BLK, stream[1]>>>(
      Y[k % 2],
      nerowsY,
      rowsY[k % 2],
      rlenY[k % 2],
      _COL_BLK,
      N_SLAB,
      _num_neurons_per_layer,
      d_W[k % 2],
      d_W[k % 2] + _num_neurons_per_layer * N_SLAB + 1,
      (T*)(d_W[k % 2] + _num_neurons_per_layer * N_SLAB + 1 + _max_nnz_per_layer),
      _bias,
      Y[(k + 1) % 2],
      rlenY[(k + 1) % 2]
    );

    checkCuda(cudaEventRecord(event_infer, stream[1]));

    checkCuda(cudaMemsetAsync(Y[k % 2], 0, sizeof(T) * num_inputs * _num_neurons_per_layer, stream[1]));

    checkCuda(cudaLaunchHostFunc(stream[1], non_empty_rows, (void*)&h_func_args));

    checkCuda(cudaStreamWaitEvent(stream[1], event_cpy, 0));
    checkCuda(cudaStreamWaitEvent(stream[0], event_infer, 0));
////issue: can I write CPU code here?
    ////h_func_args.rlenY = rlenY[(k + 1) % 2];
    ////h_func_args.rowsY = rowsY[(k + 1) % 2];
////issue: non_empty_row cannot be member function
  }

  baseline_inference<T><<<nerowsY, threads, sizeof(T) * _COL_BLK, stream[1]>>>(
    Y[(_num_layers - 1) % 2],
    nerowsY,
    rowsY[(_num_layers - 1) % 2],
    rlenY[(_num_layers - 1) % 2],
    _COL_BLK,
    N_SLAB,
    _num_neurons_per_layer,
    d_W[(_num_layers - 1 ) % 2],
    d_W[(_num_layers - 1) % 2] + _num_neurons_per_layer * N_SLAB + 1,
    (T*)(d_W[(_num_layers - 1) % 2] + _num_neurons_per_layer * N_SLAB + 1 + _max_nnz_per_layer),
    _bias,
    Y[(_num_layers) % 2],
    rlenY[(_num_layers) % 2]
  );

  checkCuda(cudaStreamEndCapture(stream[1], &graph));

  cudaGraphExec_t exec;
  cudaGraphNode_t *nodes = NULL;
  size_t num_nodes = 0;
  checkCuda(cudaGraphGetNodes(graph, nodes, &num_nodes));
  std::cout << "\nNum of nodes in the graph created using stream capture API = " << num_nodes << "\n";

  checkCuda(cudaGraphInstantiate(&exec, graph, NULL, NULL, 0));

  checkCuda(cudaGraphLaunch(exec, stream_for_graph));
  checkCuda(cudaStreamSynchronize(stream_for_graph));

  auto score = get_score<T>(Y[_num_layers % 2], num_inputs, _num_neurons_per_layer);

  checkCuda(cudaGraphExecDestroy(exec));
  checkCuda(cudaGraphDestroy(graph));
  checkCuda(cudaStreamDestroy(stream[0]));
  checkCuda(cudaStreamDestroy(stream[1]));
  checkCuda(cudaStreamDestroy(stream_for_graph));

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
Eigen::Matrix<int, Eigen::Dynamic, 1> GPUCugraph<T>::infer_graph_manual(
  const std::fs::path& input_path,
  const int num_inputs
) const {

  std::cout << "Preprocessing.............................." << std::flush;

  int N_SLAB = _num_neurons_per_layer / _COL_BLK; 

  int *d_W[2];
  checkCuda(cudaMalloc(
    &d_W[0],
    sizeof(int) * (_max_nnz_per_layer + _num_neurons_per_layer * N_SLAB + 1 + _pad) + 
    sizeof(T) * (_max_nnz_per_layer)
  ));
  checkCuda(cudaMalloc(
    &d_W[1],
    sizeof(int) * (_max_nnz_per_layer + _num_neurons_per_layer * N_SLAB + 1 + _pad) + 
    sizeof(T) * (_max_nnz_per_layer)
  ));
  checkCuda(cudaMemcpy(
    d_W[0],
    _h_pinned_weight,
    sizeof(int) * (_max_nnz_per_layer + _num_neurons_per_layer * N_SLAB + 1 + _pad) + 
    sizeof(T) * (_max_nnz_per_layer),
    cudaMemcpyHostToDevice
  ));

  std::cout << "Done" << std::endl;

  std::cout << "Reading input.............................." << std::flush;

  T* Y[2];  
  int *rowsY[2], *rlenY[2];

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
  read_input<T>(input_path, num_inputs, _num_neurons_per_layer, Y[0], rowsY[0], nerowsY);

  std::cout << "Done" << std::endl;

  std::cout << "Start inference............................" << std::flush;

  dim3 threads(32, 32, 1);
  cudaStream_t stream_for_graph;
  cudaGraph_t graph;
  std::vector<cudaGraphNode_t> cpy_dependencies;
  std::vector<cudaGraphNode_t> infer_dependencies;
  cudaGraphNode_t memcpy_node, infer_node,  host_node, memset_node;
  checkCuda(cudaStreamCreateWithFlags(&stream_for_graph, cudaStreamNonBlocking));

  checkCuda(cudaGraphCreate(&graph, 0));

  cudaMemcpy3DParms memcpy_params = {0};
  cudaKernelNodeParams infer_params = {0};
  cudaHostNodeParams host_params = {0};
  cudaMemsetParams memset_params = {0};

  //memcpy
  memcpy_params.srcArray = NULL;
  memcpy_params.srcPos   = make_cudaPos(0, 0, 0);
  memcpy_params.dstArray = NULL;
  memcpy_params.dstPos   = make_cudaPos(0, 0, 0);
  memcpy_params.extent   = make_cudaExtent(
    sizeof(int) * (_max_nnz_per_layer + _num_neurons_per_layer * N_SLAB + 1 + _pad) + 
    sizeof(T) * (_max_nnz_per_layer),
    1,
    1
  );
  memcpy_params.kind     = cudaMemcpyHostToDevice;

  //infer
  infer_params.func           = (void*)baseline_inference<T>;
  infer_params.gridDim        = dim3(nerowsY, 1, 1);
  infer_params.blockDim       = dim3(32, 32, 1);
  infer_params.sharedMemBytes = sizeof(T) * _COL_BLK;
  infer_params.extra          = NULL;
  
  //host
  HostFuncArgs h_func_args = {0};
  int layer = 0;
  h_func_args.cur_layer = &layer; 
  h_func_args.num_inputs = num_inputs;
  h_func_args.rlenY = rlenY;
  h_func_args.rowsY = rowsY;
  h_func_args.nerowsY = &nerowsY;

  host_params.fn              = non_empty_rows;
  host_params.userData        = (void*)&h_func_args;

  //memset
  memset_params.value          = 0;
  memset_params.pitch          = 0;
  memset_params.elementSize    = sizeof(float); // elementSize can be max 4 bytes
  memset_params.width          = _num_neurons_per_layer * num_inputs * (sizeof(T) / sizeof(float)); 
  memset_params.height         = 1;

 

  for(int cur_layer = 0; cur_layer < _num_layers; ++cur_layer){

    if(cur_layer != _num_layers - 1){

      memcpy_params.srcPtr = make_cudaPitchedPtr(
        _h_pinned_weight + (cur_layer + 1) * (_num_neurons_per_layer * N_SLAB + 1 + _max_nnz_per_layer + _pad + ((sizeof(T) / sizeof(int)) * _max_nnz_per_layer)),
        sizeof(int) * (_num_neurons_per_layer * N_SLAB + 1 + _max_nnz_per_layer + _pad) + sizeof(T) * (_max_nnz_per_layer),
        (_num_neurons_per_layer * N_SLAB + 1 + _pad + _max_nnz_per_layer) + (sizeof(T) / sizeof(int)) * (_max_nnz_per_layer),
        1
      );

      memcpy_params.dstPtr = make_cudaPitchedPtr(
        d_W[(cur_layer + 1) % 2],
        sizeof(int) * (_num_neurons_per_layer * N_SLAB + 1 + _max_nnz_per_layer + _pad) + sizeof(T) * (_max_nnz_per_layer),
        (_num_neurons_per_layer * N_SLAB + 1 + _pad + _max_nnz_per_layer) + (sizeof(T) / sizeof(int)) * (_max_nnz_per_layer),
        1
      );

      checkCuda(cudaGraphAddMemcpyNode(
        &memcpy_node,
        graph,
        cpy_dependencies.data(),
        cpy_dependencies.size(),
        &memcpy_params)
      );

      cpy_dependencies.clear();
      cpy_dependencies.push_back(memcpy_node);

    }

    int* roffW = d_W[cur_layer % 2];
    int* colsW = d_W[cur_layer % 2] + _num_neurons_per_layer * N_SLAB + 1;
    T* valsW = (T*)(d_W[cur_layer % 2] + _num_neurons_per_layer * N_SLAB + 1 + _max_nnz_per_layer);

    void* infer_args[13] = {
      (void*)&Y[cur_layer % 2],
      &nerowsY,
      (void*)&rowsY[cur_layer % 2],
      (void*)&rlenY[cur_layer % 2],
      (void*)&_COL_BLK,
      (void*)&N_SLAB,
      (void*)&_num_neurons_per_layer,
      (void*)&roffW,
      (void*)&colsW,
      (void*)&valsW,
      (void*)&_bias,
      (void*)&Y[(cur_layer + 1) % 2],
      (void*)&rlenY[(cur_layer + 1) % 2]
    };
    infer_params.kernelParams = (void **)infer_args;

      checkCuda(cudaGraphAddKernelNode(
        &infer_node,
        graph,
        infer_dependencies.data(),
        infer_dependencies.size(),
        &infer_params)
      );

      infer_dependencies.clear();
      infer_dependencies.push_back(infer_node);
      
      checkCuda(cudaGraphAddHostNode(
        &host_node,
        graph,
        infer_dependencies.data(),
        infer_dependencies.size(),
        &host_params)
      );

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
      infer_dependencies.push_back(host_node);
      infer_dependencies.push_back(memset_node);

      cpy_dependencies.push_back(infer_node);

  }

  cudaGraphExec_t exec;
  cudaGraphNode_t *nodes = NULL;
  size_t num_nodes = 0;
  checkCuda(cudaGraphGetNodes(graph, nodes, &num_nodes));
  printf("\nNum of nodes in the graph created manually = %zu\n", num_nodes);

  checkCuda(cudaGraphInstantiate(&exec, graph, NULL, NULL, 0));

  checkCuda(cudaGraphLaunch(exec, stream_for_graph));
  checkCuda(cudaStreamSynchronize(stream_for_graph));

  auto score = get_score<T>(Y[_num_layers % 2], num_inputs, _num_neurons_per_layer);

  checkCuda(cudaGraphExecDestroy(exec));
  checkCuda(cudaGraphDestroy(graph));
  checkCuda(cudaStreamDestroy(stream_for_graph));

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

//template <typename T>
//Eigen::Matrix<int, Eigen::Dynamic, 1> GPUCugraph<T>::infer_graph_manual_updated(
  //const std::fs::path& input_path,
  //const int num_inputs
//) const {

  //std::cout << "Preprocessing.............................." << std::flush;

  //int N_SLAB = _num_neurons_per_layer / _COL_BLK; 

  //int *d_W[2];
  //checkCuda(cudaMalloc(
    //&d_W[0],
    //sizeof(int) * (_max_nnz_per_layer + _num_neurons_per_layer * N_SLAB + 1 + _pad) + 
    //sizeof(T) * (_max_nnz_per_layer)
  //));
  //checkCuda(cudaMalloc(
    //&d_W[1],
    //sizeof(int) * (_max_nnz_per_layer + _num_neurons_per_layer * N_SLAB + 1 + _pad) + 
    //sizeof(T) * (_max_nnz_per_layer)
  //));
  //checkCuda(cudaMemcpy(
    //d_W[0],
    //_h_pinned_weight,
    //sizeof(int) * (_max_nnz_per_layer + _num_neurons_per_layer * N_SLAB + 1 + _pad) + 
    //sizeof(T) * (_max_nnz_per_layer),
    //cudaMemcpyHostToDevice
  //));

  //std::cout << "Done" << std::endl;

  //std::cout << "Reading input.............................." << std::flush;

  //T* Y[2];  
  //int *rowsY[2], *rlenY[2];

  //checkCuda(cudaMallocManaged(&Y[0], sizeof(T) * num_inputs * _num_neurons_per_layer));
  //checkCuda(cudaMallocManaged(&Y[1], sizeof(T) * num_inputs * _num_neurons_per_layer));
  //checkCuda(cudaMallocManaged(&rowsY[0], sizeof(int) * num_inputs));
  //checkCuda(cudaMallocManaged(&rowsY[1], sizeof(int) * num_inputs));
  //checkCuda(cudaMallocManaged(&rlenY[0], sizeof(int) * num_inputs));
  //checkCuda(cudaMallocManaged(&rlenY[1], sizeof(int) * num_inputs));
  //checkCuda(cudaMemset(Y[0], 0, sizeof(T) * num_inputs * _num_neurons_per_layer));
  //checkCuda(cudaMemset(Y[1], 0, sizeof(T) * num_inputs * _num_neurons_per_layer));
  //checkCuda(cudaMemset(rowsY[0], 0, sizeof(int) * num_inputs));
  //checkCuda(cudaMemset(rowsY[1], 0, sizeof(int) * num_inputs));
  //checkCuda(cudaMemset(rlenY[0], 0, sizeof(int) * num_inputs));
  //checkCuda(cudaMemset(rlenY[1], 0, sizeof(int) * num_inputs));
  //checkCuda(cudaDeviceSynchronize());

////issue: doesn't check boundary
  //int nerowsY{0};
  //read_input<T>(input_path, num_inputs, _num_neurons_per_layer, Y[0], rowsY[0], nerowsY);

  //std::cout << "Done" << std::endl;

  //std::cout << "Start inference............................" << std::flush;

  //dim3 threads(32, 32, 1);
  //cudaStream_t stream_for_graph;
  //cudaGraph_t graph;
  //std::vector<cudaGraphNode_t> cpy_dependencies;
  //std::vector<cudaGraphNode_t> infer_dependencies;
  //cudaGraphNode_t memcpy_node, infer_node,  host_node, memset_node;
  //checkCuda(cudaStreamCreateWithFlags(&stream_for_graph, cudaStreamNonBlocking));

  //checkCuda(cudaGraphCreate(&graph, 0));

  //cudaMemcpy3DParms memcpy_params = {0};
  //cudaKernelNodeParams infer_params = {0};
  //cudaHostNodeParams host_params = {0};
  //cudaMemsetParams memset_params = {0};

  ////memcpy
  //memcpy_params.srcArray = NULL;
  //memcpy_params.srcPos   = make_cudaPos(0, 0, 0);
  //memcpy_params.srcPtr = make_cudaPitchedPtr(
    //_h_pinned_weight + 1 * (_num_neurons_per_layer * N_SLAB + 1 + _max_nnz_per_layer + _pad + ((sizeof(T) / sizeof(int)) * _max_nnz_per_layer)),
    //sizeof(int) * (_num_neurons_per_layer * N_SLAB + 1 + _max_nnz_per_layer + _pad) + sizeof(T) * (_max_nnz_per_layer),
    //(_num_neurons_per_layer * N_SLAB + 1 + _pad + _max_nnz_per_layer) + (sizeof(T) / sizeof(int)) * (_max_nnz_per_layer),
    //1
  //);
  //memcpy_params.dstArray = NULL;
  //memcpy_params.dstPos   = make_cudaPos(0, 0, 0);
  //memcpy_params.dstPtr = make_cudaPitchedPtr(
    //d_W[1 % 2],
    //sizeof(int) * (_num_neurons_per_layer * N_SLAB + 1 + _max_nnz_per_layer + _pad) + sizeof(T) * (_max_nnz_per_layer),
    //(_num_neurons_per_layer * N_SLAB + 1 + _pad + _max_nnz_per_layer) + (sizeof(T) / sizeof(int)) * (_max_nnz_per_layer),
    //1
  //);
  //memcpy_params.extent   = make_cudaExtent(
    //sizeof(int) * (_max_nnz_per_layer + _num_neurons_per_layer * N_SLAB + 1 + _pad) + 
    //sizeof(T) * (_max_nnz_per_layer),
    //1,
    //1
  //);
  //memcpy_params.kind     = cudaMemcpyHostToDevice;

  ////infer
  //int* roffW = d_W[cur_layer % 2];
  //int* colsW = d_W[cur_layer % 2] + _num_neurons_per_layer * N_SLAB + 1;
  //T* valsW = (T*)(d_W[cur_layer % 2] + _num_neurons_per_layer * N_SLAB + 1 + _max_nnz_per_layer);

  //void* infer_args[13] = {
    //(void*)&Y[cur_layer % 2],
    //&nerowsY,
    //(void*)&rowsY[cur_layer % 2],
    //(void*)&rlenY[cur_layer % 2],
    //(void*)&_COL_BLK,
    //(void*)&N_SLAB,
    //(void*)&_num_neurons_per_layer,
    //(void*)&roffW,
    //(void*)&colsW,
    //(void*)&valsW,
    //(void*)&_bias,
    //(void*)&Y[(cur_layer + 1) % 2],
    //(void*)&rlenY[(cur_layer + 1) % 2]
  //};

  //infer_params.func           = (void*)baseline_inference<T>;
  //infer_params.gridDim        = dim3(nerowsY, 1, 1);
  //infer_params.blockDim       = dim3(32, 32, 1);
  //infer_params.sharedMemBytes = sizeof(T) * _COL_BLK;
  //infer_params.extra          = NULL;
  //infer_params.kernelParams   = (void **)infer_args;
  
  ////host
  //HostFuncArgs h_func_args = {0};
  //int layer = 0;
  //h_func_args.cur_layer = &layer; 
  //h_func_args.num_inputs = num_inputs;
  //h_func_args.rlenY = rlenY;
  //h_func_args.rowsY = rowsY;
  //h_func_args.nerowsY = &nerowsY;

  //host_params.fn              = non_empty_rows;
  //host_params.userData        = (void*)&h_func_args;

  ////memset
  //memset_params.value          = 0;
  //memset_params.pitch          = 0;
  //memset_params.elementSize    = sizeof(float); // elementSize can be max 4 bytes
  //memset_params.width          = _num_neurons_per_layer * num_inputs * (sizeof(T) / sizeof(float)); 
  //memset_params.height         = 1;
  //memset_params.dst = (void*)Y[0];

  ////insert node
  //checkCuda(cudaGraphAddMemcpyNode(
    //&memcpy_node,
    //graph,
    //cpy_dependencies.data(),
    //cpy_dependencies.size(),
    //&memcpy_params)
  //);

  //cpy_dependencies.clear();
  //cpy_dependencies.push_back(memcpy_node);

  //checkCuda(cudaGraphAddKernelNode(
    //&infer_node,
    //graph,
    //infer_dependencies.data(),
    //infer_dependencies.size(),
    //&infer_params)
  //);

  //infer_dependencies.clear();
  //infer_dependencies.push_back(infer_node);
  
  //checkCuda(cudaGraphAddHostNode(
    //&host_node,
    //graph,
    //infer_dependencies.data(),
    //infer_dependencies.size(),
    //&host_params)
  //);


  //checkCuda(cudaGraphAddMemsetNode(
    //&memset_node,
    //graph,
    //infer_dependencies.data(),
    //infer_dependencies.size(),
    //&memset_params)
  //);

  //infer_dependencies.clear();
  //infer_dependencies.push_back(memcpy_node);
  //infer_dependencies.push_back(host_node);
  //infer_dependencies.push_back(memset_node);

  //cpy_dependencies.push_back(infer_node);


  //for(int cur_layer = 0; cur_layer < _num_layers; ++cur_layer){

  //}

  //cudaGraphExec_t exec;
  //cudaGraphNode_t *nodes = NULL;
  //size_t num_nodes = 0;
  //checkCuda(cudaGraphGetNodes(graph, nodes, &num_nodes));
  //printf("\nNum of nodes in the graph created manually = %zu\n", num_nodes);

  //checkCuda(cudaGraphInstantiate(&exec, graph, NULL, NULL, 0));

  //checkCuda(cudaGraphLaunch(exec, stream_for_graph));
  //checkCuda(cudaStreamSynchronize(stream_for_graph));

  //auto score = get_score<T>(Y[_num_layers % 2], num_inputs, _num_neurons_per_layer);

  //checkCuda(cudaGraphExecDestroy(exec));
  //checkCuda(cudaGraphDestroy(graph));
  //checkCuda(cudaStreamDestroy(stream_for_graph));

  //checkCuda(cudaFree(Y[0]));
  //checkCuda(cudaFree(Y[1]));
  //checkCuda(cudaFree(rowsY[0]));
  //checkCuda(cudaFree(rowsY[1]));
  //checkCuda(cudaFree(rlenY[0]));
  //checkCuda(cudaFree(rlenY[1]));
  //checkCuda(cudaFree(d_W[0]));
  //checkCuda(cudaFree(d_W[1]));

  //return score;
//}
  

}// end of namespace sparse_dnn ----------------------------------------------
