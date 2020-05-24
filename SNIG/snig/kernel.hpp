#pragma once

namespace snig{

template <typename T>
__global__ 
void snig_inference(
  const T* Y0,
  const bool* rowsY0,
  const size_t COL_BLK,
  const size_t N_SLAB,
  const size_t num_neurons_per_layer,
  const int* roffW,
  const int* colsW,
  const T* valsW,
  const T bias,
  bool* rowsY1,
  T* Y1
);

//-----------------------------------------------------------------------------
//Definition of kernel function
//-----------------------------------------------------------------------------

template <typename T>
__global__ 
void snig_inference(
  const T* Y0,
  const bool* rowsY0,
  const size_t COL_BLK,
  const size_t N_SLAB,
  const size_t num_neurons_per_layer,
  const int* col_W,
  const int* row_W,
  const T* val_W,
  const T bias,
  bool* rowsY1,
  T* Y1
) {
  int tid = threadIdx.y * blockDim.x + threadIdx.x;

  //N_SLAB is small enough to compute by each single thread
  bool is_all_empty = true;
  for(size_t k = 0; k < N_SLAB; ++k) {
    is_all_empty &= !rowsY0[blockIdx.x * N_SLAB + k];
  }

  if(is_all_empty) {
    //memory reset here
    //avoid calling cudaMemset
    // threads in the first two ifs are in the same warp
    //Hence it doesn't effect performance
    if(rowsY1[blockIdx.x * N_SLAB + blockIdx.y]) {
      for(size_t j = blockIdx.y * COL_BLK + tid; j < (blockIdx.y + 1) * COL_BLK; j += blockDim.x * blockDim.y) {
        Y1[blockIdx.x * num_neurons_per_layer + j] = T(0);
      }
      __syncthreads();
      if(tid == 0) {
        rowsY1[blockIdx.x * N_SLAB + blockIdx.y] = false;
      } 
    }
    return;
  }

  extern __shared__ T shRow[];

  //use 2 length bool array in share memory(is_empty[]) to avoid synchronization
  //is_empty[0] represents whether this row is empty
  //if is_empty[0] is true, this row is empty
  //rowsY1[blockIdx.x] will then be caculated at next iteration.
  __shared__ bool is_empty[2];
  if(tid == 0) {
    is_empty[1] = true;
  }

  //use stride to reset shRow effectively
  //set shRow to bias directly
  //divide N_SLAB to blockIdx.y
  for(size_t k = tid; k < COL_BLK; k += blockDim.x * blockDim.y) {
    shRow[k] = bias;  
  }
  __syncthreads();

  for(size_t k = 0; k < N_SLAB; ++k) {
    if(!rowsY0[blockIdx.x * N_SLAB + k]) {
      continue;
    }
    for(size_t j = threadIdx.y + k * COL_BLK; j < (k + 1) * COL_BLK; j += blockDim.y) {
      T valY = Y0[blockIdx.x * num_neurons_per_layer + j];
      if(valY == 0) {
        continue;
      }
      int beg_W = col_W[blockIdx.y * num_neurons_per_layer + j] + threadIdx.x;
      int end_W = col_W[blockIdx.y * num_neurons_per_layer + j + 1];
      for(int i = beg_W; i < end_W; i += blockDim.x) {
        int rowW = row_W[i];
        T valW = val_W[i];
        atomicAdd(&shRow[rowW - blockIdx.y * COL_BLK], valY * valW);
      }
    }
  }
  __syncthreads();
  for(size_t j = tid; j < COL_BLK; j += blockDim.x * blockDim.y) {
    //use j = tid directly
    T v = min(T(32), max(shRow[j], T(0)));
    Y1[blockIdx.x * num_neurons_per_layer + blockIdx.y * COL_BLK + j] = v;
    is_empty[v > 0] = false;
  }
  //if no one set is_non_emtpy[1] to true
  //it means this row is empty
  //set rowsY1[this row] = false then
  __syncthreads();
  if(tid == 0) {
    rowsY1[blockIdx.x * N_SLAB + blockIdx.y] = !is_empty[1];
  }
}

}// end of namespace snig ----------------------------------------------
