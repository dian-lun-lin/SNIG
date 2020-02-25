#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include<doctest.h>
#include<doctest.h>
#include<SparseDNN/utility/matrix_format.h>
#include<SparseDNN/utility/matrix_operation.hpp>
#include<Eigen/Sparse>

TEST_CASE("CSR_matrix_to_eigen_sparse"){
  /*
    1 0 0
    0 1 0
    0 0 1 
  */
  sparse_dnn::CSRMatrix<float> y;
  size_t* result;
  y.row_array = new size_t[4];
  y.col_array = new size_t[3];
  y.data_array = new float[3];
  for(int i=1; i<4; ++i){
    y.row_array[i] = i;
  }
  for(int i=0; i<3; ++i){
    y.col_array[i] = i;
  }
  for(int i=0; i<3; ++i){
    y.data_array[i] = 1;
  }
  auto mat = Eigen::Matrix<float, 3, 3>::Identity();
  CHECK(mat ==
    Eigen::Matrix<float,3 ,3>(
    sparse_dnn::CSR_matrix_to_eigen_sparse<float>(y, size_t(3), size_t(3))
  ));
}
