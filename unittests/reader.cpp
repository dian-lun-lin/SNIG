#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include<doctest.h>

#include<SparseDNN/utility/reader.hpp>
#include <experimental/filesystem>
#include <fstream>

#include <Eigen/Sparse>
#include <vector>
#include <string>

//issue size limited
namespace std {
	namespace fs = experimental::filesystem;
}

// TODO
TEST_CASE("tsv_string_to_matrix") {
  std::string t1("1\t1\t1\n2\t2\t1\n3\t3\t1");
  Eigen::Matrix<float, 3, 3> mat1;
   mat1 << 1, 0, 0,
           0, 1, 0,
           0, 0, 1;
  CHECK(Eigen::Matrix<float, 3, 3>(sparse_dnn::tsv_string_to_matrix<float>(t1, 3, 3)) == mat1);

  std::string t2("100\t100\t1");
  Eigen::Matrix<double, 100, 100> mat2;
  mat2.setZero();
  mat2(99, 99) = 1;
  CHECK(Eigen::Matrix<double, 100, 100>(sparse_dnn::tsv_string_to_matrix<double>(t1, 100, 100)) == mat2);
}

//TEST_CASE("read weight"){

  //std::ostringstream oss;
  //oss << "1\t1\t1\n
          //2\t2\t1\n
          //3\t3\t1\n
          //4\t4\t1\n";

  //Eigen::SparseMatrix<float> mat(4, 3);
  //for(int i=0; i<4;++i){
    //for(int j=0; j<3; ++j){
      //if(i==j){
        //mat.coeffRef(i,j) += 1;
      //}
    //}
  //}

  //std::vector<std::fs::path> files;
  //filenames.reserve(100);
  //std::vector<Eigen::SparseMatrix<float> mats;
  //mats.reserve(100);
  //for(int i=0; i<100;++i){
    //filenames.push_back(std::fs::temp_directory_path() / "n3-l" + std::to_string(i));
    //mats.push_back(mat);
  //}

  //std::ofstream f;
  //for(auto name:filenames){
    //f.open(name);
    //f << oss.str();
    //f.close();
  //}

  //auto results = read_weight<float>(std::fs::temp_directory_path(), 3, 100);
  //CHECK(std::equal(results.begin(), results.end(), mats));

  //for(auto name:filenames){
    //std::fs::remove(name);
  //}

//}
//Test_CASE("read input"){
//Eigen::SparseMatrix<T> read_input(
    //const std::fs::path& input_path,
    //const size_t num_inputs,
    //const size_t num_features
//) {

  //std::ostringstream oss;
  //oss << "1\t1\t1\n
          //2\t2\t1\n
          //3\t3\t1\n
          //4\t4\t1\n";

  //Eigen::SparseMatrix<float> mat(4, 3);
  //for(int i=0; i<4;++i){
    //for(int j=0; j<3; ++j){
      //if(i==j){
        //mat.coeffRef(i,j) += 1;
      //}
    //}
  //}
  //auto filename = std::fs::temp_directory_path() / "input_test";
  //std::ofstream f{filename};
  //f << oss.str();
  //CHECK(read_input<float>(filename, 4, 3)==mat);
  //std::fs::remove(filename);

//}

//TEST_CASE("read golden"){

//template <typename T>
//Eigen::SparseVector<T> read_golden(
    //const std::fs::path& golden_path,
    //const size_t num_inputs
//) {

  //auto vec = Eigen::Vector<float, 65536>::Constant(65536, 1);
  //Eigen::SparseVector<float> sparse_vec = vec.SparseView();

  //auto filename = std::fs::temp_directory_path() / "input_test";
  //std::ofstream f{filename};
  //std::stringstream ss;
  //ss << vec;
  //f << ss.str();
  //CHECK(read_golden(filename, 65536)==vec);
  //std::fs::remove(filename);
//}
