#include <CLI11/CLI11.hpp>
#include <experimental/filesystem>

#include <SparseDNN/SparseDNN.hpp>

namespace std {
  namespace fs = experimental::filesystem;
}
int main(int argc, char* argv[]) {

  // usage: ./main -m sequential 
  CLI::App app{"SparseDNN"};
  std::string mode = "sequential";
  app.add_option("-m, --mode", 
    mode, 
    "select mode(sequential/), default is sequential");


  std::fs::path input_path, weight_path, golden_path;
  app.add_option("-i, --input", input_path, "input file path")
    ->check(CLI::ExistingFile);
  app.add_option("-w, --weight", weight_path, "weight directory path")
    ->check(CLI::ExistingDirectory);
  app.add_option("-g, --golden", golden_path, "golden file path")
    ->check(CLI::ExistingFile);

  size_t num_inputs, num_neurons, num_layers;
  app.add_option("--num_inputs", num_inputs, "total number of inputs");
  app.add_option(
    "--num_neurons", 
    num_neurons, 
    "total number of neurons per layer");
  app.add_option("--num_layers", num_layers, "total number of layers");

  float bias = 0.0f;
  app.add_option("-b, --bias", bias, "bias");



  CLI11_PARSE(app, argc, argv);
  // Non-RAII
  sparse_dnn::Sequential<float> sequential(
    weight_path, 
    input_path, 
    golden_path, 
    num_inputs, 
    num_neurons, 
    num_layers, 
    bias);
  
  // 1. scoring
  // 2. measure the cpu runtime
  sequential.infer();



  return 0;
}
