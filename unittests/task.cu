#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include<doctest.h>

#include<SparseDNN/parallel/task.hpp>
#include<SparseDNN/utility/matrix_format.h>
#include<memory>
#include<numeric>

//TEST_CASE("check_nnz"){
  //[>
    //1 0 0        1 0 0
    //0 1 0        0 1 0 
    //0 0 1        0 0 1
  //*/
  //sparse_dnn::CSRMatrix<float> y;
  //sparse_dnn::CSCMatrix<float> x;
  //size_t* result;
  //cudaMallocManaged(&y.row_array, sizeof(size_t)*4);
  //cudaMallocManaged(&y.col_array, sizeof(size_t)*3);
  //cudaMallocManaged(&x.row_array, sizeof(size_t)*3);
  //cudaMallocManaged(&x.col_array, sizeof(size_t)*4);
  //cudaMallocManaged(&result, sizeof(float)*4);
  //for(int i=0; i<4; ++i){
    //y.row_array[i] = i;
  //}
  //for(int i=0; i<3; ++i){
    //y.col_array[i] = i;
  //}
  //for(int i=0; i<4; ++i){
    //x.col_array[i] = i;
  //}
  //for(int i=0; i<3; ++i){
    //x.row_array[i] = i;
  //}
  //sparse_dnn::check_nnz<float><<<8, 32>>>(3, y.row_array, y.col_array, y.data_array, 3, x.col_array, x.row_array, x.data_array, result, 0);
  //cudaDeviceSynchronize();
  //std::partial_sum(result, result+4, result);
  //for(int i=0; i<4; ++i){
     //CHECK(result[i]==i);
  //}
  

//}

//TEST_CASE("task_GPU"){
  //[>
    //1 0 0        1 0 0
    //0 1 0        0 1 0 
    //0 0 1        0 0 1
  //*/
  //sparse_dnn::CSRMatrix<float> y;
  //sparse_dnn::CSCMatrix<float> x;
  //sparse_dnn::CSRMatrix<float> result;
  //cudaMallocManaged(&y.row_array, sizeof(size_t)*4);
  //cudaMallocManaged(&y.col_array, sizeof(size_t)*3);
  //cudaMallocManaged(&y.data_array, sizeof(float)*3);
  //cudaMallocManaged(&x.row_array, sizeof(size_t)*3);
  //cudaMallocManaged(&x.col_array, sizeof(size_t)*4);
  //cudaMallocManaged(&x.data_array, sizeof(float)*3);
  //cudaMallocManaged(&result.row_array, sizeof(size_t)*4);
  //cudaMallocManaged(&result.col_array, sizeof(size_t)*3);
  //cudaMallocManaged(&result.data_array, sizeof(float)*3);
  //for(int i=0; i<4; ++i){
    //y.row_array[i] = i;
    ////result.row_array need initailize
    //result.row_array[i] = i;
  //}
  //for(int i=0; i<3; ++i){
    //y.col_array[i] = i;
  //}
  //for(int i=0; i<3; ++i){
    //y.data_array[i] = 1;
  //}
  //for(int i=0; i<4; ++i){
    //x.col_array[i] = i;
  //}
  //for(int i=0; i<3; ++i){
    //x.row_array[i] = i;
  //}
  //for(int i=0; i<3; ++i){
    //x.data_array[i] = 1;
  //}
  //sparse_dnn::task_GPU<float><<<8, 32>>>(3, y.row_array, y.col_array, y.data_array, 3, x.col_array, x.row_array, x.data_array, result.row_array, result.col_array, result.data_array, 0.0f);
  //cudaDeviceSynchronize();
  //for(int i=0; i<3; ++i){
     //CHECK(result.col_array[i] == i);
  //}
  //for(int i=0; i<3; ++i){
     //CHECK(result.data_array[i] == 1);
  //}

//}
