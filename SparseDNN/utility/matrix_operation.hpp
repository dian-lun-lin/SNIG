#pragma once
#include <Eigen/Sparse>
#include <vector>
#include <SparseDNN/utility/matrix_format.h>


namespace sparse_dnn {

template<typename T>
std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor> > slice_by_row(
    const Eigen::SparseMatrix<T, Eigen::RowMajor>& target,
    const size_t num_slices
);

template<typename T>
Eigen::SparseVector<T> concatenate_by_row (
    const std::vector<Eigen::SparseVector<T> >& targets
);

template<typename T>
void eigen_sparse_to_CSR_matrix(
    const Eigen::SparseMatrix<T, Eigen::RowMajor>& target,
    CSRMatrix<T>& mat 
);

template<typename T>
void eigen_sparse_to_CSR_array(
    const Eigen::SparseVector<T>& target,
    SparseArray<T>& arr
);

template<typename T>
Eigen::SparseVector<T> get_score(const Eigen::SparseMatrix<T>& target);

//-----------------------------------------------------------------------------
//Definition of reader function
//-----------------------------------------------------------------------------

template<typename T>
std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor> > slice_by_row(
    const Eigen::SparseMatrix<T, Eigen::RowMajor>& target,
    const size_t num_slices
) {

  size_t rows_per_slice = target.rows() / num_slices;
  size_t remain = target.rows() % num_slices;

  std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor> > slices;
  slices.reserve(num_slices + 1);
  typedef Eigen::Triplet<T> E;
  std::vector<E> triplet_list;
  Eigen::SparseMatrix<T, Eigen::RowMajor> tmp(rows_per_slice, target.cols());

  triplet_list.reserve(rows_per_slice / 20);
  int counter = 0;
  for (int k = 0; k<target.outerSize(); ++k){
    //issue T:float
    for (typename Eigen::SparseMatrix<T, Eigen::RowMajor>::InnerIterator it(target, k); it;){
      if(it.row() < rows_per_slice * (counter + 1)){
        triplet_list.push_back(E(
              it.row() - (rows_per_slice*(counter)),
              it.col(),
              it.value()
              )
        );
        ++it;
      }
      else{
        tmp.reserve(triplet_list.size());
        tmp.setFromTriplets(triplet_list.begin(), triplet_list.end());
        tmp.makeCompressed();
        slices.push_back(tmp);
        ++counter;
        tmp.resize(rows_per_slice, target.cols());
        tmp.data().squeeze();
        triplet_list.clear();
      }
    }
  }
  //last one
  tmp.reserve(triplet_list.size());
  tmp.setFromTriplets(triplet_list.begin(), triplet_list.end());
  tmp.makeCompressed();
  slices.push_back(tmp);

  //issue: not test remain yet
  if(remain){
    slices.back().conservativeResize(remain, target.cols());
    slices.back().makeCompressed();
  }
  return slices;

}

template<typename T>
Eigen::SparseVector<T> concatenate_by_row(
    const std::vector<Eigen::SparseVector<T> >& targets
) {

  size_t num_nonZeros{0};
  size_t total_rows{0};
  for(const auto& r:targets){
    num_nonZeros += r.nonZeros();
    total_rows += r.rows();
  }
  Eigen::SparseVector<T> score(total_rows);
  score.reserve(num_nonZeros);

  size_t lump_sum_rows{0};
  for(size_t j = 0; j < targets.size(); ++j){
    for(typename Eigen::SparseVector<T>::InnerIterator it(targets[j]); it; ++it){
      score.coeffRef(lump_sum_rows + it.index()) =  it.value();
    }
    lump_sum_rows += targets[j].rows();
  }

  return score;
}

template<typename T>
void eigen_sparse_to_CSR_matrix(
    const Eigen::SparseMatrix<T, Eigen::RowMajor>& target,
    CSRMatrix<T>& mat
){

  size_t count = 0;
  mat.row_array[0] = 0;

  for(size_t j = 0; j < target.outerSize(); ++j){
    for(typename Eigen::SparseMatrix<T, Eigen::RowMajor>::InnerIterator it(target, j); it; ++it){
      mat.col_array[count] = it.col();
      mat.data_array[count++] = it.value();
    }
  mat.row_array[j + 1] = count;
  }
  return;
}

template<typename T>
void eigen_sparse_to_sparse_array(
    const Eigen::SparseVector<T>& target,
    SparseArray<T>& arr
) {

  int count = 0;
  for(typename Eigen::SparseVector<T>::InnerIterator it(target); it; ++it){
    arr.index_array[count] = it.index();
    arr.data_array[count++] = it.value();
  }
  return;
}

template<typename T>
Eigen::SparseMatrix<T> CSR_matrix_to_eigen_sparse(
  const CSRMatrix<T>& mat,
  const size_t rows,
  const size_t cols
) {
  Eigen::SparseMatrix<T> result(rows, cols);
  result.reserve(rows / 200);
  for(size_t i = 0; i < rows; ++i){
    for(size_t j = mat.row_array[i]; j < mat.row_array[i + 1]; ++j){
      result.coeffRef(i, mat.col_array[j]) =  mat.data_array[j];
    }
  }
  result.makeCompressed();
  return result;
}

template<typename T>
Eigen::SparseVector<T> get_score(const Eigen::SparseMatrix<T>& target){

  //Eigen::SparseVector<T> score = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
    //(target).rowwise().sum().sparseView();
  Eigen::SparseVector<T> score = (target * Eigen::Matrix<T, Eigen::Dynamic, 1>::Ones(target.cols())).sparseView();

  score = score.unaryExpr([] (T a) {
    if(a > 0) return 1;
    else return 0;
  });
  return score;
}

}// end of namespace sparse_dnn ----------------------------------------------
