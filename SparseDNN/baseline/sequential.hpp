#pragma once

#include <iostream>
#include <SparseDNN/utility/file.hpp>
#include <Eigen/Sparse>
#include <vector>

namespace std {
  namespace fs = experimental::filesystem;
}
namespace sparse_dnn {
template <typename T>
class Sequential {

  // T is the floating point type, either float or double
  static_assert(
    std::is_same<T, float>::value || std::is_same<T, double>::value,
    "data type must be either float or double"
  );

  private:

    const std::fs::path _weight_path;
    const size_t _num_inputs;
    const size_t _num_neurons;
    const size_t _num_layers;
    const T _bias;

    std::vector<Eigen::SparseMatrix<T> > _read_weight_and_input (
      const std::fs::path& input_path
    ) const;

    Eigen::SparseMatrix<T> _read_golden (const std::fs::path& path) const;
    Eigen::SparseMatrix<T> _str_to_matrix(
      const std::string&s, const int& rows, const int& cols
    ) const;

  public:
    
    // TODO: restyle
    Sequential(
      const std::fs::path weight_path,
    const size_t num_inputs,
      const size_t num_neurons,
      const size_t num_layers,
      T bias
    );

    ~Sequential();

    const std::fs::path& weight_path() const { return _weight_path; };

    size_t num_neurons() const { return _num_neurons; };
    size_t num_layers() const { return _num_layers; };
    T bias() const { return _bias; };

    void infer(const std::fs::path& input, const std::fs::path& golden) const;
    void check_passed(const Eigen::SparseMatrix<T>& output_mat, const std::fs::path& golden) const;
};

// ----------------------------------------------------------------------------
// Definition of Sequential
// ----------------------------------------------------------------------------

template <typename T>
Sequential<T>::Sequential(
  const std::fs::path weight_path,
  const size_t num_inputs,
  const size_t num_neurons,
  const size_t num_layers,
  T bias
) :
  _weight_path {weight_path},
  _num_inputs {num_inputs},
  _num_neurons {num_neurons},
  _num_layers {num_layers},
  _bias {bias}
{
  std::cout << "Constructing a sequential baseline.\n";
}

template <typename T>
Sequential<T>::~Sequential() {
}



template <typename T>
std::vector<Eigen::SparseMatrix<T> > Sequential<T>::_read_weight_and_input (const std::fs::path& input_path) const {
  std::vector<Eigen::SparseMatrix<T> > mats;
  mats.reserve(_num_layers + 1);
  //read input
  std::string input_str = read_file_to_string(input_path);
  mats.push_back(_str_to_matrix(input_str, _num_inputs, _num_neurons));
  //read weights
  for(int i = 0; i < _num_layers; ++i){
    std::fs::path p = _weight_path;
    p /= "n" + std::to_string(_num_neurons) + "-l" + std::to_string(i + 1) + ".tsv";
    std::string data_str = read_file_to_string(p);
    mats.push_back(_str_to_matrix(data_str, _num_neurons, _num_neurons));
  }
  return mats;
}

template <typename T>
Eigen::SparseMatrix<T> Sequential<T>::_str_to_matrix(const std::string& s, const int &rows, const int &cols) const {
  typedef Eigen::Triplet<T> E;
  std::string line;
  std::vector<E> tripletList;
  Eigen::SparseMatrix<T> mat(rows, cols);
  std::istringstream read_s(s);
  while(std::getline(read_s, line)){
    std::istringstream lineStream(line);
    std::string token;
    std::vector<T> tokens;
    while(std::getline(lineStream, token, '\t'))  tokens.push_back(std::stof(token));
    tripletList.push_back(E(tokens[0] - 1, tokens[1] - 1, tokens[2]));
  }
  mat.reserve(tripletList.size());
  mat.setFromTriplets(tripletList.begin(), tripletList.end());
  mat.makeCompressed();
  return mat;
}

template <typename T>
inline 
Eigen::SparseMatrix<T> Sequential<T>::_read_golden(const std::fs::path& path) const {
  std::string line;
  std::istringstream read_s(read_file_to_string(path));
  Eigen::SparseMatrix<T> v(_num_inputs, 1);
  v.reserve(_num_inputs / 200);
  while(std::getline(read_s, line)) v.insert(std::stoi(line) - 1, 1) = 1;
  return v;
}


template <typename T>
void Sequential<T>::infer(
  const std::fs::path& input, const std::fs::path& golden
) const {
  std::cout << "Loading the weight and input..............\n";
  Eigen::SparseMatrix<T> input_mat(_num_inputs, _num_neurons);
  auto mats = _read_weight_and_input(input);
  std::cout << "Start inference............................\n";
  Eigen::SparseMatrix<T> Y(mats[0]);
  Eigen::SparseMatrix<T> Z(Y.rows(), Y.cols());
  Z.reserve(Y.nonZeros());
  for(auto w = mats.begin() + 1; w < mats.end(); ++w){
    Z = (Y * (*w)).pruned();
    Z.coeffs() += _bias;
    Y = Z.unaryExpr([] (T a) {
        if(a < 0) return T(0);
        else if(a > 32) return T(32);
        return a;
        });
  }
  Y.makeCompressed();
  std::cout << "Done.\n";
  check_passed(Y, golden);
}
template <typename T>
inline 
void Sequential<T>::check_passed(const Eigen::SparseMatrix<T>& output_mat, const std::fs::path& golden) const {
  std::cout << "Checking correctness........................\n";
  auto golden_mat = _read_golden(golden);
  Eigen::SparseMatrix<T> final_output_mat(_num_neurons, 1);
  final_output_mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> (output_mat).rowwise().sum().sparseView();
  //final_output_mat = output_mat * Eigen::Matrix<T, output_mat.cols(), 1>::Ones();
  final_output_mat = final_output_mat.unaryExpr([] (T a) {
      if(a > 0) return T(1);
      else return T(0);
      });
  Eigen::SparseMatrix<T> diff_mat = golden_mat - final_output_mat;
  diff_mat = diff_mat.pruned();
  if(diff_mat.nonZeros())
    std::cout << "Challenge FAILED\n";
  else
    std::cout << "Challenge PASSED\n" ;
}

}  // end of namespace sparse_dnn ---------------------------------------------



