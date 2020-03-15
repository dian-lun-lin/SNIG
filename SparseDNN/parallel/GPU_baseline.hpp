#include <Eigen/Dense>
#include <SparseDNN/utility/matrix_format.h>
#include <SparseDNN/utility/cuda_error.hpp>
#include <SparseDNN/parallel/task.hpp>
#include <SparseDNN/utility/scoring.hpp>

#include <typeinfo>

namespace std {
  namespace fs = experimental::filesystem;  
}

namespace sparse_dnn{

template <typename T>  
class GPUBaseline {

  
  private:
    int* _h_pinned_weight;
    T _bias;
    int _num_neurons_per_layer;
    int _num_layers;
    int _COL_BLK;
    int _max_nnz_per_layer;

    void _non_empty_rows(
      const int num_inputs,
      int* rlenY,
      int* rowsY,
      int& nnz
    ) const;

  public:
    GPUBaseline(
      const std::fs::path& weight_path,
      const T bias = -.3f,
      const int num_neurons_per_layer = 1024,
      const int num_layers = 120,
      const int COL_BLK = 8192
    );

    ~GPUBaseline();

    int num_neurons_per_layer() const { return _num_neurons_per_layer; };
    int num_layers() const { return _num_layers; };
    T bias() const { return _bias; };
    Eigen::Matrix<int, Eigen::Dynamic, 1> infer(
      const std::fs::path& input_path,
      const int num_inputs
    ) const;

};

// ----------------------------------------------------------------------------
// Definition of GPUBaseline
// ----------------------------------------------------------------------------

template <typename T>
GPUBaseline<T>::GPUBaseline(
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
  std::memset(_h_pinned_weight, 0, (sizeof(int) * (num_neurons_per_layer * N_SLAB + 1 + _max_nnz_per_layer) + sizeof(T) * _max_nnz_per_layer) * num_layers);

  read_weight<T>(weight_path, num_neurons_per_layer, _max_nnz_per_layer, num_layers, _COL_BLK, N_SLAB, _h_pinned_weight);

  std::cout << "Done\n";
}

template <typename T>
GPUBaseline<T>:: ~GPUBaseline(){

  checkCuda(cudaFreeHost(_h_pinned_weight));
}

template <typename T>
Eigen::Matrix<int, Eigen::Dynamic, 1> GPUBaseline<T>::infer(
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

  cudaStream_t stream[2];
  checkCuda(cudaStreamCreateWithFlags(&stream[0], cudaStreamNonBlocking));
  checkCuda(cudaStreamCreateWithFlags(&stream[1], cudaStreamNonBlocking));
//issue: how many threads
  dim3 threads(32, 32, 1);
  for(int k = 0; k < _num_layers - 1; ++k){

    checkCuda(cudaMemcpyAsync(
      d_W[(k + 1) % 2],
      _h_pinned_weight + (k + 1) * (_num_neurons_per_layer * N_SLAB + 1 + _max_nnz_per_layer + ((sizeof(T) / sizeof(int)) * _max_nnz_per_layer)),
      sizeof(int) * (_num_neurons_per_layer * N_SLAB + 1 + _max_nnz_per_layer) + sizeof(T) * (_max_nnz_per_layer),
      cudaMemcpyHostToDevice,
      stream[0]
    ));

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
      reinterpret_cast<T*>(d_W[k % 2] + _num_neurons_per_layer * N_SLAB + 1 + _max_nnz_per_layer),
      _bias,
      Y[(k + 1) % 2],
      rlenY[(k + 1) % 2]
    );

    checkCuda(cudaMemsetAsync(Y[k % 2], 0, sizeof(T) * num_inputs * _num_neurons_per_layer, stream[1]));
    checkCuda(cudaStreamSynchronize(stream[1]));
    _non_empty_rows(num_inputs, rlenY[(k + 1) % 2], rowsY[(k + 1) % 2], nerowsY);
    checkCuda(cudaStreamSynchronize(stream[0]));
  }

  baseline_inference<T><<<nerowsY, threads, sizeof(T) * _COL_BLK, stream[1]>>>(
    Y[(_num_layers - 1) % 2],
    nerowsY,
    rowsY[(_num_layers - 1) % 2],
    rlenY[(_num_layers - 1) % 2],
    _COL_BLK,
    N_SLAB,
    _num_neurons_per_layer,
    d_W[(_num_layers - 1) % 2],
    d_W[(_num_layers - 1) % 2] + _num_neurons_per_layer * N_SLAB + 1,
    reinterpret_cast<T*>(d_W[(_num_layers - 1) % 2] + _num_neurons_per_layer * N_SLAB + 1 + _max_nnz_per_layer),
    _bias,
    Y[(_num_layers) % 2],
    rlenY[(_num_layers) % 2]
  );
  checkCuda(cudaStreamSynchronize(stream[1]));

  std::cout << "Done" << std::endl;

  auto score = get_score<T>(Y[_num_layers % 2], num_inputs, _num_neurons_per_layer);

  checkCuda(cudaStreamDestroy(stream[0]));
  checkCuda(cudaStreamDestroy(stream[1]));
  
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
void GPUBaseline<T>::_non_empty_rows(
  const int num_inputs,
  int* rlenY,
  int* rowsY,
  int& nnz
) const {
  
  nnz = 0;

  for(int i = 0; i < num_inputs; ++i){
    if(rlenY[i] > 0){
      rowsY[nnz++] = i;
    }
  }
}


}// end of namespace sparse_dnn ----------------------------------------------
