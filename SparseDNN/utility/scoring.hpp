#pragma once
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <SparseDNN/utility/matrix_format.h>

namespace sparse_dnn {

template<typename T>
Eigen::Matrix<int, Eigen::Dynamic, 1> get_score(const Eigen::SparseMatrix<T>& target);

template<typename T>
Eigen::Matrix<int, Eigen::Dynamic, 1> get_score(const CSRMatrix<T>& target, const int rows);


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

  int tmp{0};
  Eigen::Matrix<int, Eigen::Dynamic, 1> score(rows, 1);
  for(int i = 0; i < rows; ++i){
    for(int j = target.row_array[i]; j < target.row_array[i + 1]; ++j){
      tmp += target.data_array[j];
    }
    score(i, 0) = tmp > 0 ? 1 : 0;
  }
  return score;
}

inline
bool is_passed(
    const Eigen::Matrix<int, Eigen::Dynamic, 1>& output,
    const Eigen::Matrix<int, Eigen::Dynamic, 1>& golden
) {
  return (output.cwiseEqual(golden).count());
}

}// end of namespace sparse_dnn ----------------------------------------------
