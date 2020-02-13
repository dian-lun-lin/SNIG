#include <CLI11/CLI11.hpp>
#include <experimental/filesystem>

#include <SparseDNN/SparseDNN.hpp>
#include <SparseDNN/utility/reader.hpp>

namespace std {
  namespace fs = experimental::filesystem;
}


//#include <thread>

int main(int argc, char* argv[]) {
   
  // TODO: study std::thread
  // https://en.cppreference.com/w/cpp/thread/thread/thread
  
  // TODO: study std::promise and std::future
  // https://en.cppreference.com/w/cpp/thread/promise
  // https://en.cppreference.com/w/cpp/thread/future

  // usage: ./main -m sequential 
  CLI::App app{"SparseDNN"};
  std::string mode = "sequential";
  app.add_option("-m, --mode", 
    mode, 
    "select mode(sequential/), default is sequential");

  std::fs::path weight_path;
  app.add_option("-w, --weight", weight_path, "weight directory path")
    ->check(CLI::ExistingDirectory);

  size_t num_neurons_per_layer=1024;
  app.add_option(
    "--num_neurons_per_layer", 
    num_neurons_per_layer, 
    "total number of neurons per layer, default is 1024"
  );

  size_t num_layers=120;
  app.add_option(
      "--num_layers",
      num_layers, 
      "total number of layers, default is 120"
  );

  float bias = -0.3f;
  app.add_option("-b, --bias", bias, "bias");

  CLI11_PARSE(app, argc, argv);

  sparse_dnn::Sequential<float> sequential(
    weight_path, 
    bias,
    num_neurons_per_layer, 
    num_layers
  );

  std::fs::path input_path("/home/dian-lun/dian/GraphChallenge_SparseDNN/dataset/MNIST/sparse-images-1024.tsv");
  std::fs::path golden_path("/home/dian-lun/dian/GraphChallenge_SparseDNN/dataset/MNIST/neuron1024-l120-categories.tsv");
  //auto issue
  auto result = sequential.infer(input_path, 60000);
  auto golden = sparse_dnn::read_golden<float>(golden_path, 60000);
  if(sparse_dnn::is_passed<float>(result, golden)){
    std::cout << "CHALLENGE PASSED\n";
  }
  else{
    std::cout << "CHALLENGE FAILED\n";
  }
  return 0;
}
