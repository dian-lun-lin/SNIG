#pragma once
#include <Eigen/SparseCore>
#include <vector>
#include <SNIG/utility/matrix_format.h>
#include <Eigen/Dense>

namespace snig {

template<typename T>
std::vector<Eigen::SparseMatrix<T> > slice_by_row(
  const Eigen::SparseMatrix<T, Eigen::RowMajor>& target,
  const size_t num_slices
);

template<typename T>
std::vector<T*> slice_by_row(
  const T*,
  const size_t num_slices
);

inline
Eigen::Matrix<int, Eigen::Dynamic, 1> concatenate_by_row (
  const std::vector<Eigen::Matrix<int, Eigen::Dynamic, 1> >& targets
);


template<typename T>
Eigen::SparseMatrix<T> CSR_matrix_to_eigen_sparse(
  const CSRMatrix<T>& mat,
  const size_t rows,
  const size_t cols
);

template<typename T>
void eigen_sparse_to_CSR_matrix(
  const Eigen::SparseMatrix<T, Eigen::RowMajor>& target,
  CSRMatrix<T>& mat 
);

template<typename T>
void eigen_sparse_to_CSC_matrix(
  const Eigen::SparseMatrix<T>& target,
  CSRMatrix<T>& mat
);

template<typename T>
void eigen_sparse_to_CSR_array(
  const Eigen::SparseVector<T>& target,
  SparseArray<T>& arr
);

inline
Eigen::Matrix<int, Eigen::Dynamic, 1> arr_to_Eigen_int(
  const int* arr,
  const size_t arr_len
);


//-----------------------------------------------------------------------------
//Definition of reader function
//-----------------------------------------------------------------------------

template<typename T>
std::vector<Eigen::SparseMatrix<T> > slice_by_row(
  const Eigen::SparseMatrix<T, Eigen::RowMajor>& target,
  const size_t num_slices
) {

  size_t rows_per_slice = target.rows() / num_slices;
  size_t remain = target.rows() % num_slices;

  std::vector<Eigen::SparseMatrix<T> > slices;
  slices.reserve(num_slices + 1);
  typedef Eigen::Triplet<T> E;
  std::vector<E> triplet_list;

  triplet_list.reserve((rows_per_slice * target.cols()) / 1000);
  int counter = 0;
  Eigen::SparseMatrix<T> tmp(rows_per_slice, target.cols());
  for (int k = 0; k<target.outerSize(); ++k) {
    for (typename Eigen::SparseMatrix<T, Eigen::RowMajor>::InnerIterator it(target, k); it;) {
      if(it.row() < rows_per_slice * (counter + 1)) {
        triplet_list.emplace_back(
              it.row() - (rows_per_slice*(counter)),
              it.col(),
              it.value()
        );
        ++it;
      }
      else{
        tmp.reserve(triplet_list.size());
        tmp.setFromTriplets(triplet_list.begin(), triplet_list.end());
        slices.push_back(std::move(tmp));
        ++counter;
        triplet_list.clear();
      }
    }
  }
  //last one
  tmp.reserve(triplet_list.size());
  tmp.setFromTriplets(triplet_list.begin(), triplet_list.end());
  slices.push_back(std::move(tmp));

  //issue: not test remain yet
  if(remain){
    slices.back().conservativeResize(remain, target.cols());
    slices.back().makeCompressed();
  }
  return slices;

}

template<typename T>
std::vector<T* > slice_by_row(
  const T*,
  const size_t num_slices
) {
}

inline
Eigen::Matrix<int, Eigen::Dynamic, 1> concatenate_by_row(
  const std::vector<Eigen::Matrix<int, Eigen::Dynamic, 1> >& targets
) {
  Eigen::Matrix<int, Eigen::Dynamic, 1> score((targets.size() - 1) * targets[0].rows() + targets.back().rows() , 1);

  score.block(0, 0, targets[0].rows(), targets[0].cols())  = targets[0]; 
  for(size_t i = 1; i < targets.size(); ++i){
    score.block(i * targets[i - 1].rows(), 0, targets[i].rows(), targets[i].cols())  = targets[i]; 
  } 

  return score;
}

template<typename T>
void eigen_sparse_to_CSR_matrix(
  const Eigen::SparseMatrix<T, Eigen::RowMajor>& target,
  CSRMatrix<T>& mat
){
  std::copy(target.outerIndexPtr(), target.outerIndexPtr() + target.rows() + 1, mat.row_array);
  std::copy(target.innerIndexPtr(), target.innerIndexPtr() + target.nonZeros(), mat.col_array);
  std::copy(target.valuePtr(), target.valuePtr() + target.nonZeros(), mat.data_array);
}

template<typename T>
void eigen_sparse_to_CSC_matrix(
  const Eigen::SparseMatrix<T>& target,
  CSRMatrix<T>& mat
) {
  std::copy(target.outerIndexPtr(), target.outerIndexPtr() + target.cols() + 1, mat.col_array);
  std::copy(target.innerIndexPtr(), target.innerIndexPtr() + target.nonZeros(), mat.row_array);
  std::copy(target.valuePtr(), target.valuePtr() + target.nonZeros(), mat.data_array);
}

template<typename T>
void eigen_sparse_to_sparse_array(
  const Eigen::SparseVector<T>& target,
  SparseArray<T>& arr
) {
  std::copy(target.innerIndexPtr(), target.innerIndexPtr() + target.nonZeros(), arr.index_array);
  std::copy(target.valuePtr(), target.valuePtr() + target.nonZeros(), arr.data_array);
}

template<typename T>
Eigen::SparseMatrix<T> CSR_matrix_to_eigen_sparse(
  const CSRMatrix<T>& mat,
  const size_t rows,
  const size_t cols
) {
  Eigen::SparseMatrix<T> result(rows, cols);
  result.reserve(rows*cols / 1000);
  for(size_t i = 0; i < rows; ++i) {
    for(size_t j = mat.row_array[i]; j < mat.row_array[i + 1]; ++j) {
      result.coeffRef(i, mat.col_array[j]) =  mat.data_array[j];
    }
  }
  result.makeCompressed();
  return result;
}

inline
Eigen::Matrix<int, Eigen::Dynamic, 1> arr_to_Eigen_int(
  const int* arr,
  const size_t arr_len
) {
  Eigen::Matrix<int, Eigen::Dynamic, 1> result(arr_len, 1);
  for(size_t i = 0; i < arr_len; ++i) {
    result(i, 1) = arr[i];
  }
  return result;
};


}// end of namespace snig ----------------------------------------------
