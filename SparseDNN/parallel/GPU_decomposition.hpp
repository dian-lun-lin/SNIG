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
class GPUDecomp {

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
      T* h_Y,
      T** Y,
      bool** rowsY,
      int** d_W,
      const size_t num_inputs,
      const size_t num_buff,
      const size_t batch_size,
      const size_t batch_ylen,
      const size_t batch_ysize
    ) const;

  public:

    GPUDecomp(
      const std::fs::path& weight_path,
      const T bias = -.3f,
      const size_t num_neurons_per_layer = 1024,
      const size_t num_layers = 120
    );

    ~GPUDecomp();

    size_t num_neurons_per_layer() const;
    size_t num_layers() const;

    Eigen::Matrix<int, Eigen::Dynamic, 1> infer(
      const std::fs::path& input_path,
      const size_t num_inputs,
      const size_t batch_size,
      const size_t num_buff
    ) const;

};

// ----------------------------------------------------------------------------
// Definition of GPUDecomp
// ----------------------------------------------------------------------------

template <typename T>
GPUDecomp<T>::GPUDecomp(
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
GPUDecomp<T>::~GPUDecomp() {
  checkCuda(cudaFreeHost(_h_pinned_weight));
}

template <typename T>
size_t GPUDecomp<T>::num_neurons_per_layer() const {
   return _num_neurons_per_layer; 
}

template <typename T>
size_t GPUDecomp<T>::num_layers() const { 
  return _num_layers; 
}

template <typename T>
Eigen::Matrix<int, Eigen::Dynamic, 1> GPUDecomp<T>::infer(
  const std::fs::path& input_path,
  const size_t num_inputs,
  const size_t batch_size,
  const size_t num_buff
) const {

  std::cout << "Preprocessing.............................." << std::flush;
  auto pp_beg = std::chrono::steady_clock::now();

  //weight allocation
  int *d_W[num_buff];
  for(size_t i = 0; i < num_buff; ++i) {
    checkCuda(cudaMalloc(
      &d_W[i],
      _pp_wsize
    ));
  }

  size_t batch_ylen = batch_size * _num_neurons_per_layer;
  size_t batch_ysize = batch_ylen * sizeof(T);
  size_t ylen = num_inputs * _num_neurons_per_layer;
  size_t ysize = ylen * sizeof(T);

  //input allocation
  T* h_Y;
  checkCuda(cudaMallocHost(
    (void**)&h_Y,
    ysize
  ));

  T* Y[2];  
  bool* rowsY[2];

  checkCuda(cudaMalloc(&Y[0], batch_ysize));
  checkCuda(cudaMalloc(&Y[1], batch_ysize));
  checkCuda(cudaMallocManaged(&rowsY[0], sizeof(bool) * batch_size));
  checkCuda(cudaMallocManaged(&rowsY[1], sizeof(bool) * batch_size));
  checkCuda(cudaMemset(Y[0], 0, batch_ysize));
  checkCuda(cudaMemset(Y[1], 0, batch_ysize));
  checkCuda(cudaMemset(rowsY[0], 1, sizeof(bool) * batch_size));
  checkCuda(cudaMemset(rowsY[1], 0, sizeof(bool) * batch_size));

  read_input_binary<T>(input_path, batch_size, h_Y);

  auto pp_end = std::chrono::steady_clock::now();
  
  std::cout << "finished preprocessing with " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(pp_end - pp_beg).count()
            << "ms"
            << std::endl;

  std::cout << "Start inference............................" << std::flush;
  auto exec_beg = std::chrono::steady_clock::now();

  _infer_flatterned_graph(h_Y, Y, rowsY, d_W, num_inputs, num_buff, batch_size, batch_ylen, batch_ysize);

  auto exec_end = std::chrono::steady_clock::now();
  std::cout << "finished execution with "
            << std::chrono::duration_cast<std::chrono::milliseconds>(exec_end - exec_beg).count()
            << "ms"
            << std::endl;

  std::cout << "Start scoring..............................." << std::flush;
  auto score_beg = std::chrono::steady_clock::now();

  auto score = get_score<T>(h_Y, num_inputs, _num_neurons_per_layer);

  auto score_end = std::chrono::steady_clock::now();
  std::cout << "finished scoring with "
            << std::chrono::duration_cast<std::chrono::milliseconds>(score_end - score_beg).count()
            << "ms"
            << std::endl;

  checkCuda(cudaFreeHost(h_Y));
  checkCuda(cudaFree(Y[0]));
  checkCuda(cudaFree(Y[1]));
  checkCuda(cudaFree(rowsY[0]));
  checkCuda(cudaFree(rowsY[1]));

  for(size_t k =0; k < num_buff; ++k) {
    checkCuda(cudaFree(d_W[k]));
  }

  return score;
}

template <typename T>
void GPUDecomp<T>::_infer_flatterned_graph(
  T* h_Y,
  T** Y,
  bool** rowsY,
  int** d_W,
  const size_t num_inputs,
  const size_t num_buff,
  const size_t batch_size,
  const size_t batch_ylen,
  const size_t batch_ysize
) const {

  int device = -1;
  cudaGetDevice(&device);
  checkCuda(cudaMemPrefetchAsync(rowsY[0], sizeof(bool) * batch_size, device, NULL));
  checkCuda(cudaMemPrefetchAsync(rowsY[1], sizeof(bool) * batch_size, device, NULL));

  dim3 threads(16, 16, 1);
  cudaStream_t stream_for_graph;
  cudaGraph_t graph;

  std::vector<std::vector<cudaGraphNode_t> >infer_dependencies(num_buff);
  std::vector<std::vector<cudaGraphNode_t> >cpy_dependencies(num_buff);
  std::vector<cudaGraphNode_t> w_memcpy_nodes(num_buff);
  std::vector<cudaGraphNode_t> infer_nodes(num_buff);
  std::vector<cudaGraphNode_t> memset_nodes(num_buff);

  checkCuda(cudaStreamCreate(&stream_for_graph));

  checkCuda(cudaGraphCreate(&graph, 0));

  cudaMemcpy3DParms w_memcpy_params = {0};
  cudaKernelNodeParams infer_params = {0};
  cudaMemsetParams memset_params = {0};

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
  infer_params.func           = (void*)wo_host_inference_test<T>;
  infer_params.gridDim        = dim3(batch_size, 1, 1);
  infer_params.blockDim       = threads;
  infer_params.sharedMemBytes = sizeof(T) * _COL_BLK;
  infer_params.extra          = NULL;
  
  //memset
  memset_params.value         = 0;
  memset_params.pitch         = 0;
  memset_params.elementSize   = sizeof(float); // elementSize can be max 4 bytes
  memset_params.width         = batch_ylen * (sizeof(T) / sizeof(float)); 
  memset_params.height        = 1;

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
      infer_dependencies[k].push_back(infer_nodes[k]);

      memset_params.dst = (void*)Y[k % 2];
      checkCuda(cudaGraphAddMemsetNode(
        &memset_nodes[k],
        graph,
        infer_dependencies[k].data(),
        infer_dependencies[k].size(),
        &memset_params)
      );

      infer_dependencies[k].clear();

      if(k != num_buff - 1) {
        infer_dependencies[k + 1].push_back(memset_nodes[k]);
      }

      cpy_dependencies[k].push_back(infer_nodes[k]);
    }
    infer_dependencies[0].push_back(memset_nodes[num_buff - 1]);
  }

  size_t num_edges = 0;
  cudaGraphNode_t *nodes = NULL;
  size_t num_nodes = 0;
  checkCuda(cudaGraphGetNodes(graph, nodes, &num_nodes));
  std::cout << "\nNum of nodes in the graph created manually = "
            << num_nodes
            << std::endl;

  checkCuda(cudaGraphGetEdges(graph, nodes, nodes, &num_edges));
  std::cout << "Num of edges in the graph created manually = " 
            << num_edges
            << std::endl;


  cudaGraphExec_t exec;

  checkCuda(cudaGraphInstantiate(&exec, graph, NULL, NULL, 0));

  cudaStream_t stream_for_D2H_cpy;
  checkCuda(cudaStreamCreate(&stream_for_D2H_cpy));
  for(size_t batch = 0; batch < num_inputs; batch += batch_size) {
    checkCuda(cudaMemcpy(
      Y[0],
      h_Y + batch * _num_neurons_per_layer,
      batch_ysize,
      cudaMemcpyHostToDevice
      )
    );
    checkCuda(cudaGraphLaunch(exec, stream_for_graph));
    checkCuda(cudaStreamSynchronize(stream_for_graph));
    checkCuda(cudaMemcpy(
      h_Y + batch * _num_neurons_per_layer,
      Y[num_buff % 2],
      batch_ysize,
      cudaMemcpyDeviceToHost
      )
    );
    checkCuda(cudaMemset(rowsY[0], 1, sizeof(bool) * batch_size));
  }

  checkCuda(cudaGraphExecDestroy(exec));
  checkCuda(cudaGraphDestroy(graph));
  checkCuda(cudaStreamDestroy(stream_for_graph));

}

}// end of namespace sparse_dnn ----------------------------------------------
