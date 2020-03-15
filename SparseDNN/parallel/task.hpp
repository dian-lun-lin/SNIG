#pragma once
#include <Eigen/Sparse>
#include <SparseDNN/utility/matrix_operation.hpp>
#include <SparseDNN/utility/matrix_format.h>
#include <cusparse_v2.h>
#include <algorithm>
#include <thrust/scan.h>

namespace sparse_dnn{

struct HostFuncArgs{
  int num_inputs;
  int* cur_layer;
  int* nerowsY;
  int** rlenY;
  int** rowsY;
};

inline
void cusparse_mutiplication(
  const CSRMatrix<float>& a,
  const CSRMatrix<float>& b,
  int a_row,
  int a_col,
  int b_col,
  int nnz_a,
  int nnz_b,
  CSRMatrix<float>& c
);

inline
void cusparse_mutiplication(
  const CSRMatrix<double>& a,
  const CSRMatrix<double>& b,
  int a_row,
  int a_col,
  int b_col,
  int nnz_a,
  int nnz_b,
  CSRMatrix<double>& c
);

template <typename T>
__global__
void add_bias(T* arr, int nnz, T bias);

template <typename T>
void resize_CPU(CSRMatrix<T>& target, int rows);

template <typename T>
void add_bias_relu_CPU(T* arr, T bias, int rows);
template <typename T>
__global__ 
void baseline_inference(
  const T* Y0,
  const int nerowsY,
  const int* rowsY0,
  int* rlenY0,
  const int COL_BLK,
  const int N_SLAB,
  const int num_neurons_per_layer,
  const int* roffW,
  const int* colsW,
  const T* valsW,
  const T bias,
  T* Y1,
  int* rlenY1
);


//-----------------------------------------------------------------------------
//Definition of task function
//-----------------------------------------------------------------------------

template <typename T>
void CUDART_CB non_empty_rows(
  void* data
) {
  HostFuncArgs* args = (HostFuncArgs*)(data);
  int num_inputs = args->num_inputs;
  int* cur_layer = args->cur_layer;
  int* nnz = args->nerowsY;
  int* rlen = args->rlenY[((*cur_layer) + 1) % 2];
  int* nnz_rows = args->rowsY[((*cur_layer) + 1) % 2];

  *nnz = 0;

  for(int i = 0; i < num_inputs; ++i){
    if(rlen[i] != 0){
      nnz_rows[(*nnz)++] = i;
    }
  }
  ++(*cur_layer);
}

inline
void cusparse_mutiplication(
  const CSRMatrix<float>& a,
  const CSRMatrix<float>& b,
  int a_row,
  int a_col,
  int b_col,
  int nnz_a,
  int nnz_b,
  CSRMatrix<float>& c
) {

  cusparseHandle_t handle;
  cusparseMatDescr_t descr_a, descr_b, descr_c;
  cusparseMatDescr_t descr_d = 0;
  cusparseCreate(&handle);
  cusparseCreateMatDescr(&descr_a);
  cusparseCreateMatDescr(&descr_b);
  cusparseCreateMatDescr(&descr_d);
  cusparseCreateMatDescr(&descr_c);
  cusparseSetMatType(descr_a, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr_a, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(descr_b, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr_b, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(descr_c, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr_c, CUSPARSE_INDEX_BASE_ZERO);

  int base_c, nnz_c;
  csrgemm2Info_t info = NULL;
  size_t buffer_size;
  void *buffer = NULL;
  int *nnz = &nnz_c;
  float alpha = 1.0;
  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
  cusparseCreateCsrgemm2Info(&info);
  cusparseScsrgemm2_bufferSizeExt(
    handle, a_row, a_col, b_col, &alpha,
    descr_a, nnz_a, a.row_array, a.col_array,
    descr_b, nnz_b, b.row_array, b.col_array,
    NULL,
    descr_d, 0, NULL, NULL,
    info,
    &buffer_size
  );
  cudaMalloc(&buffer, buffer_size);

  cusparseXcsrgemm2Nnz(
    handle, a_row, a_col, b_col,
    descr_a, nnz_a, a.row_array, a.col_array,
    descr_b, nnz_b, b.row_array, b.col_array,
    descr_d, 0, NULL, NULL,
    descr_c, c.row_array, nnz,
    info, 
    buffer
  );

  if (NULL != nnz){
      nnz_c = *nnz;
  }else{
      cudaMemcpy(&nnz_c, c.row_array + a_row, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&base_c, c.row_array, sizeof(int), cudaMemcpyDeviceToHost);
      nnz_c -= base_c;
  }

  cudaMalloc(&c.col_array, sizeof(int) * nnz_c);
  cudaMalloc(&c.data_array, sizeof(float) * nnz_c);
  cusparseScsrgemm2(
    handle, a_row, a_col, b_col, &alpha,
    descr_a, nnz_a, a.data_array, a.row_array, a.col_array,
    descr_b, nnz_b, b.data_array, b.row_array, b.col_array,
    NULL,
    descr_d, 0, NULL, NULL, NULL,
    descr_c, c.data_array, c.row_array, c.col_array,
    info, 
    buffer
  );

  cudaDeviceSynchronize();

  cusparseDestroyCsrgemm2Info(info);
  cudaFree(buffer);

}

inline
void cusparse_mutiplication(
  const CSRMatrix<double>& a,
  const CSRMatrix<double>& b,
  int a_row,
  int a_col,
  int b_col,
  int nnz_a,
  int nnz_b,
  CSRMatrix<double>& c
) {

  cusparseHandle_t handle;
  cusparseMatDescr_t descr_a, descr_b, descr_c;
  cusparseMatDescr_t descr_d = 0;
  cusparseCreate(&handle);
  cusparseCreateMatDescr(&descr_a);
  cusparseCreateMatDescr(&descr_b);
  cusparseCreateMatDescr(&descr_d);
  cusparseCreateMatDescr(&descr_c);
  cusparseSetMatType(descr_a, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr_a, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(descr_b, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr_b, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(descr_c, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr_c, CUSPARSE_INDEX_BASE_ZERO);

  int base_c, nnz_c;
  csrgemm2Info_t info = NULL;
  size_t buffer_size;
  void *buffer = NULL;
  int *nnz = &nnz_c;
  double alpha = 1.0;
  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
  cusparseCreateCsrgemm2Info(&info);
  cusparseDcsrgemm2_bufferSizeExt(
    handle, a_row, a_col, b_col, &alpha,
    descr_a, nnz_a, a.row_array, a.col_array,
    descr_b, nnz_b, b.row_array, b.col_array,
    NULL,
    descr_d, 0, NULL, NULL,
    info,
    &buffer_size
  );
  cudaMalloc(&buffer, buffer_size);

  cusparseXcsrgemm2Nnz(
    handle, a_row, a_col, b_col,
    descr_a, nnz_a, a.row_array, a.col_array,
    descr_b, nnz_b, b.row_array, b.col_array,
    descr_d, 0, NULL, NULL,
    descr_c, c.row_array, nnz,
    info, 
    buffer
  );

  if (NULL != nnz){
      nnz_c = *nnz;
  }else{
      cudaMemcpy(&nnz_c, c.row_array + a_row, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&base_c, c.row_array, sizeof(int), cudaMemcpyDeviceToHost);
      nnz_c -= base_c;
  }

  cudaMalloc(&c.col_array, sizeof(int) * nnz_c);
  cudaMalloc(&c.data_array, sizeof(double) * nnz_c);
  cusparseDcsrgemm2(
    handle, a_row, a_col, b_col, &alpha,
    descr_a, nnz_a, a.data_array, a.row_array, a.col_array,
    descr_b, nnz_b, b.data_array, b.row_array, b.col_array,
    NULL,
    descr_d, 0, NULL, NULL, NULL,
    descr_c, c.data_array, c.row_array, c.col_array,
    info, 
    buffer
  );

  cudaDeviceSynchronize();

  cusparseDestroyCsrgemm2Info(info);
  cudaFree(buffer);

}

template <typename T>
void add_bias_relu_CPU(T* arr, T bias, int rows){

  for(int k = 0; k < rows; ++k){
    arr[k] += bias;
    if(arr[k] < 0){
      arr[k] = 0;
    }
    else if(arr[k] > 32){
      arr[k] = 32;
    }
  }
}

template <typename T>
__global__
void add_bias(T* arr, int* nnz, T bias){
  int batch = (*nnz + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x);
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  for(int i = index; i < index + blockDim.x * batch; i += blockDim.x){
    if(i < *nnz){
      arr[index] += bias;
    }
  }
} 

template <typename T>
void resize_CPU(CSRMatrix<T>& target, int rows) {

  int nnz = target.row_array[rows - 1];
  int reduce_arr[rows];
  std::memset(reduce_arr, 0, sizeof(int) * rows);

  for(int i = 0; i < nnz; ++i){
    if(target.data_array[i] == 0){

      auto it = std::lower_bound(
        target.row_array, 
        target.row_array + rows,
        i + 1
      );
      ++reduce_arr[it - target.row_array]; 

      target.col_array[i] = -1;
    }
  }

  thrust::inclusive_scan(reduce_arr, reduce_arr + rows, reduce_arr);
  for(int k = 0; k < rows; ++k){
    target.row_array[k] -= reduce_arr[k];
  }

  std::remove(target.data_array, target.data_array + nnz, 0);
  std::remove(target.col_array, target.col_array + nnz, -1);
}

template <typename T>
__global__ 
void baseline_inference(
  const T* Y0,
  const int nerowsY,
  const int* rowsY0,
  int* rlenY0,
  const int COL_BLK,
  const int N_SLAB,
  const int num_neurons_per_layer,
  const int* roffW,
  const int* colsW,
  const T* valsW,
  const T bias,
  T* Y1,
  int* rlenY1
) {

  if(blockIdx.x >= nerowsY){
    return;
  }

  extern  __shared__ T shRow[];

  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int rid = rowsY0[blockIdx.x];
  __syncthreads();
  if(tid == 0){
    rlenY0[rid] = 0;
    rlenY1[rid] = 0;
  }
  for(int i = 0; i < N_SLAB; i++){
    __syncthreads();
    for(int j = threadIdx.x; j < COL_BLK; j++){
      shRow[j] = 0;  
    }
    __syncthreads();
    for(int j = threadIdx.y; j < num_neurons_per_layer; j += blockDim.y){
      T valY = Y0[rid * num_neurons_per_layer + j];
      if(valY == 0){
        continue;
      }
      int begOffW = roffW[i * num_neurons_per_layer + j] + threadIdx.x;
      int endOffW = roffW[i * num_neurons_per_layer + j + 1];
      for(int k = begOffW; k < endOffW; k += blockDim.x){
        int colW = colsW[k];
        T valW = valsW[k];
        atomicAdd(&shRow[colW - i * COL_BLK], valY * valW);
      }
    }
    __syncthreads();
    int count = 0;
    for(int j = 0; j < COL_BLK; j += blockDim.x * blockDim.y){
      if(j + tid < COL_BLK){
        T v = shRow[j + tid] + bias;
        count += __syncthreads_count(v > 0);
        Y1[rid * num_neurons_per_layer + i * COL_BLK + j + tid] = min(T(32), max(T(0), v));
      }
    }
    if(tid == 0){
      rlenY1[rid] += count;
    }
  }

}

}// end of namespace sparse_dnn ----------------------------------------------
