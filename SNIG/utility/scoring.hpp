#pragma once
#include <thrust/scan.h>
#include <Eigen/SparseCore>
#include <Eigen/Dense>
#include <SNIG/utility/matrix_format.h>

namespace snig {

template<typename T>
__global__
void identify(
  T* target_arr,
  const size_t batch_size,
  const size_t num_neurons_per_layer,
  int* result_arr
);

template<typename T>
Eigen::Matrix<int, Eigen::Dynamic, 1> get_score(
  const Eigen::SparseMatrix<T>& target
);


template<typename T>
Eigen::Matrix<int, Eigen::Dynamic, 1> get_score(
  const CSRMatrix<T>& target,
  const size_t rows
);

template<typename T>
Eigen::Matrix<int, Eigen::Dynamic, 1> get_score(
  const T* arr,
  const size_t rows,
  const size_t cols
);

inline
bool is_passed(
  const Eigen::Matrix<int, Eigen::Dynamic, 1>& output,
  const Eigen::Matrix<int, Eigen::Dynamic, 1>& golden
);


//-----------------------------------------------------------------------------
//Definition of scoring function
//-----------------------------------------------------------------------------

template<typename T>
__global__
void identify(
  T* target_arr,
  const size_t batch_size,
  const size_t num_neurons_per_layer,
  int* result_arr
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < batch_size; i += gridDim.x * blockDim.x) {
    T sum = thrust::reduce(
      thrust::device,
      target_arr + i * num_neurons_per_layer,
      target_arr + (i + 1) * num_neurons_per_layer,
      0,
      thrust::plus<T>()
    );
    result_arr[i] = sum > 0 ? 1 : 0;
  }
};


template<typename T>
Eigen::Matrix<int, Eigen::Dynamic, 1> get_score(
  const Eigen::SparseMatrix<T>& target
) {

  Eigen::Matrix<T, Eigen::Dynamic, 1> result = target * Eigen::Matrix<T, Eigen::Dynamic, 1>::Ones(target.cols());

  Eigen::Matrix<int, Eigen::Dynamic, 1> score = result.unaryExpr([] (T a) {
    if(a > 0) return 1;
    else return 0;
  });
  
  return score;
}

template<typename T>
Eigen::Matrix<int, Eigen::Dynamic, 1> get_score(
  const CSRMatrix<T>& target,
  const size_t rows
) {

  Eigen::Matrix<int, Eigen::Dynamic, 1> score(rows, 1);
  for(size_t i = 0; i < rows; ++i) {
    int beg = target.row_array[i];
    int end = target.row_array[i + 1];
    T sum = std::accumulate(target.data_array + beg, target.data_array + end, 0);
    score(i, 0) = sum > 0 ? 1 : 0;
  }
  return score;
}

template<typename T>
Eigen::Matrix<int, Eigen::Dynamic, 1> get_score(
  const T* arr,
  const size_t rows,
  const size_t cols
) {

  Eigen::Matrix<int, Eigen::Dynamic, 1> score(rows, 1);
  for(size_t i = 0; i < rows; ++i) {
    T sum = std::accumulate(arr + i * cols, arr + (i+1) * cols, 0);
    score(i, 0) = sum > 0 ? 1 : 0;
  }
  return score;
}

inline
bool is_passed(
  const Eigen::Matrix<int, Eigen::Dynamic, 1>& output,
  const Eigen::Matrix<int, Eigen::Dynamic, 1>& golden
) {
  int check = output.rows() - output.cwiseEqual(golden).count();
  std::cout << "\nNumber of different categories: " << check << std::endl;
  return (check == 0);
}

}// end of namespace snig ----------------------------------------------
