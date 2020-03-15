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
    int _COL_BLK;
    int _max_nnz_per_layer;

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
      const T bias,
      const int num_neurons_per_layer,
      const int num_layers,
      const int COL_BLK
    );

    ~GPUCugraph();

    int num_neurons_per_layer() const { return _num_neurons_per_layer; };
    int num_layers() const { return _num_layers; };
    T bias() const { return _bias; };

    Eigen::Matrix<int, Eigen::Dynamic, 1> infer(
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
  const int num_layers,
  const int COL_BLK
):
  _bias{bias},
  _num_neurons_per_layer{num_neurons_per_layer},
  _num_layers{num_layers},
  _COL_BLK{COL_BLK}
{
  std::cout << "Constructing a GPU parallel network.\n";
  std::cout << "Loading the weight.............." << std::flush;

  int N_SLAB = std::ceil(num_neurons_per_layer / float(COL_BLK)); 

  _max_nnz_per_layer = find_max_nnz(weight_path, num_layers, num_neurons_per_layer);

  checkCuda(cudaMallocHost(
    (void**)&_h_pinned_weight,
    (sizeof(int) * (num_neurons_per_layer * N_SLAB + 1 + _max_nnz_per_layer) +
    sizeof(T) * _max_nnz_per_layer) * num_layers
  ));

  read_weight<T>(
    weight_path, 
    num_neurons_per_layer,
    _max_nnz_per_layer,
    num_layers,
    _COL_BLK,
    N_SLAB,
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
  const int num_inputs
) const {

  std::cout << "Preprocessing.............................." << std::flush;
  int *d_W[2];
  checkCuda(cudaMalloc(
    &d_W[0],
    sizeof(int) * (_max_nnz_per_layer + _num_neurons_per_layer + 1) + 
    sizeof(T) * (_max_nnz_per_layer)
  ));
  checkCuda(cudaMalloc(
    &d_W[1],
    sizeof(int) * (_max_nnz_per_layer + _num_neurons_per_layer + 1) + 
    sizeof(T) * (_max_nnz_per_layer)
  ));
  checkCuda(cudaMemcpy(
    d_W[0],
    _h_pinned_weight,
    sizeof(int) * (_max_nnz_per_layer + _num_neurons_per_layer + 1) + 
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
  int N_SLAB = std::ceil(_num_neurons_per_layer / float(_COL_BLK)); 

//issue: how many threads
  dim3 threads(32, 32, 1);

  cudaStream_t stream[2], stream_for_graph;
  checkCuda(cudaStreamCreateWithFlags(&stream[0], cudaStreamNonBlocking));
  checkCuda(cudaStreamCreateWithFlags(&stream[1], cudaStreamNonBlocking));
  checkCuda(cudaStreamCreateWithFlags(&stream_for_graph, cudaStreamNonBlocking));

  cudaGraph_t graph;
  cudaGraphExec_t exec;
  cudaEvent_t event_cpy, event_infer, fork_event;
  cudaEventCreate (&event_cpy);
  cudaEventCreate (&event_infer);
  cudaEventCreate (&fork_event);
  HostFuncArgs h_func_args = {0};
  int cur_layer = 0;
  h_func_args.num_inputs = num_inputs;
  h_func_args.cur_layer = &cur_layer; 
  h_func_args.nerowsY = &nerowsY;
  h_func_args.rlenY = rlenY;
  h_func_args.rowsY = rowsY;

  checkCuda(cudaStreamBeginCapture(stream[0],  cudaStreamCaptureModeGlobal));

  checkCuda(cudaEventRecord(fork_event, stream[0]));
  checkCuda(cudaStreamWaitEvent(stream[1], fork_event, 0));

  for(int k = 0; k < _num_layers; ++k){


    if(k < _num_layers -1 ){
      checkCuda(cudaMemcpyAsync(
        d_W[(k + 1) % 2],
        _h_pinned_weight + (k + 1) * (_num_neurons_per_layer * N_SLAB + 1 + _max_nnz_per_layer + ((sizeof(T) / sizeof(int)) * _max_nnz_per_layer)),
        sizeof(int) * (_num_neurons_per_layer * N_SLAB + 1 + _max_nnz_per_layer) + sizeof(T) * (_max_nnz_per_layer),
        cudaMemcpyHostToDevice,
        stream[0]
      ));
      checkCuda(cudaEventRecord(event_cpy, stream[0]));
    }


    //baseline_inference<T><<<nerowsY, threads, sizeof(T) * _COL_BLK, stream[1]>>>(
      //Y[k % 2],
      //nerowsY,
      //rowsY[k % 2],
      //rlenY[k % 2],
      //_COL_BLK,
      //N_SLAB,
      //_num_neurons_per_layer,
      //_max_nnz_per_layer,
      //d_W[k % 2],
      //_bias,
      //Y[(k + 1) % 2],
      //rlenY[(k + 1) % 2]
    //);
    //checkCuda(cudaEventRecord(event_infer, stream[1]));

////issue: can I write CPU code here?
    ////h_func_args.rlenY = rlenY[(k + 1) % 2];
    ////h_func_args.rowsY = rowsY[(k + 1) % 2];

////issue: non_empty_row cannot be member function

    //checkCuda(cudaLaunchHostFunc(stream[1], non_empty_rows<T>, (void*)&h_func_args));

    //checkCuda(cudaStreamWaitEvent(stream[1], event_cpy, 0));
    //checkCuda(cudaStreamWaitEvent(stream[0], event_infer, 0));
  }

  checkCuda(cudaStreamEndCapture(stream[0], &graph));

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



//template <typename T>
//void GPUCugraph<T>::_cuda_graph_manual(
//) const {
  //dim3 threads(32, 32, 1);
  //cudaStream_t streamForGraph;
  //cudaGraph_t graph;
  //std::vector<cudaGraphNode_t> node_dependencies;
  //cudaGraphNode_t memcpy_node, kernel_node,  host_node;
  //checkCuda(cudaStreamCreateWithFlags(&streamForGraph, cudaStreamNonBlocking));

  //checkCuda(cudaGraphCreate(&graph, 0));

  //cudaMemcpy3DParms memcpy_params = {0};
  //cudaKernelNodeParams kernel_params = {0};
  //cudaHostNodeParams host_params = {0};

  //memcpy_params.srcArray = NULL;
  //memcpy_params.srcPos   = make_cudaPos(0, 0, 0);
  //memcpy_params.dstArray = NULL;
  //memcpy_params.dstPos   = make_cudaPos(0, 0, 0);
  //memcpy_params.extent   = make_cudaExtent(
    //sizeof(int) * (_nnz_per_layer + _num_neurons_per_layer + 1) + 
    //sizeof(T) * (_nnz_per_layer),
    //1,
    //1
  //);
  //memcpy_params.kind     = cudaMemcpyHostToDevice;

  //for(int cur_layer = 0; cur_layer < _num_layers; ++cur_layer){

    //if(k != _num_layers - 1){
      //memcpy_params.srcPtr = make_cudaPitchedPtr(
        //_h_pinned_weight + (cur_layer + 1) * (_num_neurons_per_layer * N_SLAB + 1 + _nnz_per_layer + ((sizeof(T) / sizeof(int)) * _nnz_per_layer)),
        //sizeof(int) * (_num_neurons_per_layer * N_SLAB + 1 + _nnz_per_layer) + sizeof(T) * (_nnz_per_layer),
        //(_num_neurons_per_layer * N_SLAB + 1 + _nnz_per_layer) + (sizeof(T) / sizeof(float)) * (_nnz_per_layer),
        //1
      //);

      //memcpy_params.dstPtr = make_cudaPitchedPtr(
        //d_W[(cur_layer + 1) % 2],
        //sizeof(int) * (_num_neurons_per_layer * N_SLAB + 1 + _nnz_per_layer) + sizeof(T) * (_nnz_per_layer),
        //(_num_neurons_per_layer * N_SLAB + 1 + _nnz_per_layer) + (sizeof(T) / sizeof(float)) * (_nnz_per_layer),
        //1
      //);
      //checkCuda(cudaGraphAddMemcpyNode(
        //&memcpy_node,
        //graph,
        //node_dependencies.data(),
        //node_dependencies.size(),
        //&memcpy_params)
      //);
    //}
    //void *kernel_args[4] = {(void*)&input_d, (void*)&weight_d, (void*)&r_d, &N};

    //kernel_params.func           = (void*)baseline_kernel<T>;
    //kernel_params.gridDim        = dim3(1, 1, 1);
    //kernel_params.blockDim       = dim3(32, 32, 1);
    //kernel_params.sharedMemBytes = sizeof(T) * _COL_BLK;
    //kernel_params.kernelParams   = (void **)kernel_args;
    //kernel_params.extra          = NULL;

    //checkCuda(cudaGraphAddKernelNode(&multiply_kernel_node, graph, node_dependencies.data(), node_dependencies.size(), &kernel_params));


  //}
//}
  

}// end of namespace sparse_dnn ----------------------------------------------
