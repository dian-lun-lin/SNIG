#pragma once
#include <Eigen/Sparse>
#include <SparseDNN/utility/matrix_operation.hpp>
#include <vector>

namespace sparse_dnn{


template<typename T>
__global__
void CSR_mutiply_CSC(
  const size_t* y_n_rows,
  const size_t* y_row_array,
  const size_t* y_col_array,
  const T* y_data_array,
  const size_t* w_n_cols,
  const size_t* w_col_array,
  const size_t* w_row_array,
  const T* w_data_array,
  size_t* result_row_array,
  size_t* result_col_array,
  T* result_data_array,
  const T* bias
);

template<typename T>
__global__
void check_nnz(
  const size_t* y_n_rows,
  const size_t* y_row_array,
  const size_t* y_col_array,
  const size_t* y_data_array,
  const size_t* w_n_cols,
  const size_t* w_col_array,
  const size_t* w_row_array,
  const size_t* w_data_array,
  size_t* result_row_array,
  const T* bias
);

//-----------------------------------------------------------------------------
//Definition of task function
//-----------------------------------------------------------------------------


template<typename T>
__global__
void CSR_mutiply_CSC(
  const size_t* y_n_rows,
  const size_t* y_row_array,
  const size_t* y_col_array,
  const T* y_data_array,
  const size_t* w_n_cols,
  const size_t* w_col_array,
  const size_t* w_row_array,
  const T* w_data_array,
  size_t* result_row_array,
  size_t* result_col_array,
  T* result_data_array,
  const T* bias
) {
  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if(row < *y_n_rows){
    const size_t row_start = y_row_array[row];
    const size_t row_end = y_row_array[row + 1];
    T sum = 0;
    size_t current_col;
    size_t nnz_count = result_row_array[row];
    for(size_t w_col = 0; w_col < *w_n_cols; ++w_col){
      current_col = w_col_array[w_col];
      for(size_t y = row_start; y < row_end; ++y){
        for(size_t w = current_col; w < w_col_array[w_col + 1]; ++w){
          if(y_col_array[y] > w_row_array[w]){
            continue;
          }
          else if(y_col_array[y] < w_row_array[w]){
            break;
          }
          else{
            sum += y_data_array[y] * w_data_array[w];
            current_col = w;
            break;
          }
        }
      }
      if(sum == 0){
        continue;
      }
      sum += *bias;
      if(sum <= 0){
        sum = 0;
        continue;
      }
      if(sum > 32){
        sum = 32;
      }
      result_data_array[nnz_count] = sum;
      result_col_array[nnz_count++] = w_col;
      sum = 0;
    }
  }
}

template<typename T>
__global__
void check_nnz(
  const size_t* y_n_rows,
  const size_t* y_row_array,
  const size_t* y_col_array,
  const T* y_data_array,
  const size_t* w_n_cols,
  const size_t* w_col_array,
  const size_t* w_row_array,
  const T* w_data_array,
  size_t* result_row_array,
  const T* bias
) {
  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if(row < *y_n_rows){
    size_t nnz = 0;
    T sum = 0;
    size_t current_col;
    const size_t row_start = y_row_array[row];
    const size_t row_end = y_row_array[row + 1];
    for(size_t w_col = 0; w_col < *w_n_cols; ++w_col){
      current_col = w_col_array[w_col];
      for(size_t y = row_start; y < row_end; ++y){
        for(size_t w = current_col; w < w_col_array[w_col + 1];++w){

          if(y_col_array[y] > w_row_array[w]){
            continue;
          }
          else if(y_col_array[y] < w_row_array[w]){
            break;
          }
          else{
            sum += y_data_array[y] * w_data_array[w];
            current_col = w;
            break;
          }
        }
      }
      if(sum == 0){
        continue;
      }
      sum += *bias;
      if(sum <= 0){
        sum = 0;
        continue;
      }
      ++nnz;
      sum = 0;
  }
  result_row_array[row + 1] = nnz;
  }
}

}// end of namespace sparse_dnn ----------------------------------------------
