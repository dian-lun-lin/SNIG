#pragma once
#include <iostream>
#include <Eigen/Sparse>
#include <SparseDNN/utility/reader.hpp>
#include <vector>

#include<thread>
#include <ThreadPool/ThreadPool.h>

namespace std {
  namespace fs = experimental::filesystem;
}

namespace sparse_dnn {
template <typename T>
class DataParallel {

  static_assert(
  std::is_same<T, float>::value||std::is_same<T, double>::value,
  "data type must be either float or double"
  );

  private:

    std::vector<Eigen::SparseMatrix<T> > _weights;
    const size_t _num_neurons_per_layer;
    const size_t _num_layers;
    const T _bias;


    Eigen::SparseVector<T> _infer_CPU(
        const Eigen::SparseMatrix<T, Eigen::RowMajor>& Y, 
        const size_t num_inputs
    ) const;

    //Eigen::SparseVector<T> _infer_GPU(
        //const Eigen::SparseMatrix<T>& Y,
        //const size_t num_inputs
    //) const;
    Eigen::SparseVector<T> _task(Eigen::SparseMatrix<T, Eigen::RowMajor>& Y) const;

    std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor> >_slicing_inputs(
        const Eigen::SparseMatrix<T, Eigen::RowMajor>& input,
        const size_t num_tasks,
        const size_t num_inputs
    ) const;

    Eigen::SparseVector<T> _combine_results(
        const std::vector<Eigen::SparseVector<T> >& get_results,
        const size_t num_inputs,
        const size_t num_inputs_per_task
    ) const;

  public:

    DataParallel(
      const std::fs::path& wieght_path,
      const T bias,
      const size_t num_neurons_per_layer=1024,
      const size_t num_layers=120
      );

    ~DataParallel();

    size_t num_neurons_per_layer() const { return _num_neurons_per_layer; };
    size_t num_layers() const { return _num_layers; };
    T bias() const { return _bias; };

    Eigen::SparseVector<T> infer(
      const std::fs::path& input_path,
      const size_t num_inputs,
      bool is_GPU=false
      ) const;
};

// ----------------------------------------------------------------------------
// Definition of DataParallel
// ----------------------------------------------------------------------------

template<typename T>
DataParallel<T>::DataParallel(
  const std::fs::path& weight_path,
  const T bias,
  const size_t num_neurons_per_layer,
  const size_t num_layers
):
  _bias{bias},
  _num_neurons_per_layer{num_neurons_per_layer},
  _num_layers{num_layers}
{
  std::cout << "Constructing a data parallel network.\n";

  std::cout << "Loading the weight.............." << std::flush;
  _weights = read_weight<T>(weight_path, num_neurons_per_layer, num_layers);
  std::cout << "Done\n";
}

template<typename T>
DataParallel<T>::~DataParallel(){

}

template<typename T>
Eigen::SparseVector<T> DataParallel<T>::infer(
  const std::fs::path& input_path,
  const size_t num_inputs,
  bool is_GPU
  ) const {

  std::cout << "Reading input.............................." << std::flush;
  auto input = read_input<T>(input_path, num_inputs, _num_neurons_per_layer);
  std::cout << "Done\n";

  std::cout << "Start inference............................" << std::flush;
  Eigen::SparseVector<T> result;

  //if(is_GPU){
    //result=_infer_GPU(input, num_inputs);
  //}
  //else{
    result=_infer_CPU(input, num_inputs);
  //}
  return result;
}

template<typename T>
Eigen::SparseVector<T> DataParallel<T>::_infer_CPU(
    const Eigen::SparseMatrix<T, Eigen::RowMajor>& input,
    const size_t num_inputs
) const {

  //issue can I include library here?
  //issue nested function not support
  size_t num_tasks = 32;
  size_t num_threads = std::thread::hardware_concurrency();
  ThreadPool pool(num_threads);

  //slicing
  auto slicing_inputs = _slicing_inputs(input, num_tasks, num_inputs);

  std::vector<std::future<Eigen::SparseVector<T> > > results;
  results.reserve(num_tasks + 1);

  //issue alignment
  //lvalue reference to rvalue is illegal
  //rvalue reference
  for(auto each_input:slicing_inputs){
    results.push_back(pool.enqueue(
        [this] (Eigen::SparseMatrix<T, Eigen::RowMajor>& Y){return this->_task(Y);},
        each_input
      )
    );
  }
  for(auto r=results.begin(); r!=results.end(); ++r){
    r->wait();
  }

  //issue cannot auto r: results
  std::vector<Eigen::SparseVector<T> > get_results;
  get_results.reserve(results.size());
  for(auto r=results.begin(); r!=results.end(); ++r){
    get_results.push_back(r->get());
  }

  auto score = _combine_results(get_results, num_inputs, num_inputs/num_tasks);

  return score;
}

template<typename T>
Eigen::SparseVector<T> DataParallel<T>::_task(Eigen::SparseMatrix<T, Eigen::RowMajor>& Y) const {
   //issue how eigen overload assignment
  Eigen::SparseMatrix<T> Z;
  for(auto w : _weights){
    Z = (Y * w).pruned();
    Z.coeffs() += _bias;
    Y = Z.unaryExpr([] (T a) {
     if(a < 0) return T(0);
     else if(a > 32) return T(32);
     return a;
     });
  }

  Eigen::SparseVector<T> score = 
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(Y)
    .rowwise().sum().sparseView();

  score = score.unaryExpr([] (T a) {
    if(a > 0) return 1;
    else return 0;
  });
  return score;
}

template<typename T>
std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor> > DataParallel<T>::_slicing_inputs(
    const Eigen::SparseMatrix<T, Eigen::RowMajor>& input,
    const size_t num_tasks,
    const size_t num_inputs
) const {

  size_t num_inputs_per_task = num_inputs/num_tasks;
  size_t remain = num_inputs%num_tasks;

  std::vector<Eigen::SparseMatrix<T, Eigen::RowMajor> > slicing_inputs;
  slicing_inputs.reserve(num_tasks + 1);
  typedef Eigen::Triplet<T> E;
  std::vector<E> triplet_list;
  Eigen::SparseMatrix<T, Eigen::RowMajor> tmp(num_inputs_per_task, _num_neurons_per_layer);

  triplet_list.reserve(num_inputs/200);
  int counter=0;
  for (int k=0; k<input.outerSize(); ++k){
    //issue T:float
    for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(input, k); it;){
      if(it.row() < num_inputs_per_task * (counter+1)){
        triplet_list.push_back(E(
              it.row()-(num_inputs_per_task*(counter)),
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
        slicing_inputs.push_back(tmp);
        ++counter;
        tmp.resize(num_inputs_per_task, _num_neurons_per_layer);
        tmp.data().squeeze();
        triplet_list.clear();
      }
    }
  }
  //last one
  tmp.reserve(triplet_list.size());
  tmp.setFromTriplets(triplet_list.begin(), triplet_list.end());
  tmp.makeCompressed();
  slicing_inputs.push_back(tmp);

  //issue: not test remain yet
  if(remain){
    slicing_inputs.back().conservativeResize(remain, _num_neurons_per_layer);
    slicing_inputs.back().makeCompressed();
  }
  return slicing_inputs;

}

template<typename T>
Eigen::SparseVector<T> DataParallel<T>::_combine_results(
    const std::vector<Eigen::SparseVector<T> >& get_results,
    const size_t num_inputs,
    const size_t num_inputs_per_task
) const {

  int num_nonZeros{0};
  for(auto r:get_results){
    num_nonZeros+=r.nonZeros();
  }
  Eigen::SparseVector<T> score(num_inputs);
  score.reserve(num_nonZeros);

  for(int j=0; j<get_results.size(); ++j){
    for(Eigen::SparseVector<float>::InnerIterator it(get_results[j]); it; ++it){
      score.coeffRef(j*num_inputs_per_task + it.index()) =  it.value();
    }
  }

  return score;
}


}// end of namespace sparse_dnn ----------------------------------------------
