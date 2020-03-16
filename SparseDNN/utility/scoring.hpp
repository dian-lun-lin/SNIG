#pragma once
#include <Eigen/SparseCore>
#include <Eigen/Dense>
#include <SparseDNN/utility/matrix_format.h>

namespace sparse_dnn {

template<typename T>
Eigen::Matrix<int, Eigen::Dynamic, 1> get_score(const Eigen::SparseMatrix<T>& target);


template<typename T>
Eigen::Matrix<int, Eigen::Dynamic, 1> get_score(
  const CSRMatrix<T>& target,
  const int rows
);

template<typename T>
Eigen::Matrix<int, Eigen::Dynamic, 1> get_score(
  const T* arr,
  const int rows,
  const int cols
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
Eigen::Matrix<int, Eigen::Dynamic, 1> get_score(const Eigen::SparseMatrix<T>& target){

  Eigen::Matrix<T, Eigen::Dynamic, 1> result = (target * Eigen::Matrix<T, Eigen::Dynamic, 1>::Ones(target.cols()));

  Eigen::Matrix<int, Eigen::Dynamic, 1> score = result.unaryExpr([] (T a) {
    if(a > 0) return 1;
    else return 0;
  });
  
  return score;
}

template<typename T>
Eigen::Matrix<int, Eigen::Dynamic, 1> get_score(const CSRMatrix<T>& target, const int rows) {

  Eigen::Matrix<int, Eigen::Dynamic, 1> score(rows, 1);
  for(int i = 0; i < rows; ++i){
    int beg = target.row_array[i];
    int end = target.row_array[i + 1];
    T sum = std::accumulate(target.data_array + beg, target.data_array + end, 0);
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
  std::cout << "Number of different categories: " << check << std::endl;

  return (check == 0);

}

template<typename T>
Eigen::Matrix<int, Eigen::Dynamic, 1> get_score(
  const T* arr,
  const int rows,
  const int cols
){

  Eigen::Matrix<int, Eigen::Dynamic, 1> score(rows, 1);
  for(int i = 0; i < rows; ++i){
    T sum = std::accumulate(arr + i * cols, arr + (i+1) * cols, 0);
    score(i, 0) = sum > 0 ? 1 : 0;
  }
  return score;
}

}// end of namespace sparse_dnn ----------------------------------------------
