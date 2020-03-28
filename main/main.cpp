#include <CLI11/CLI11.hpp>
#include <Eigen/Dense>
#include <SparseDNN/SparseDNN.hpp>
#include <SparseDNN/utility/reader.hpp>
#include <SparseDNN/utility/scoring.hpp>

int main(int argc, char* argv[]) {
   
  // usage: ./main -m sequential 
  //        ./main -m CPU_parallel
  
  CLI::App app{"SparseDNN"};
  std::string mode = "sequential";
  app.add_option("-m, --mode", 
    mode, 
    "select mode(sequential/CPU_parallel), default is sequential");

  std::fs::path weight_path("../sample_data/weight/neuron1024/");
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

  //for testing
  std::fs::path input_path("../sample_data/MNIST/sparse-images-1024.tsv");
  app.add_option(
      "--input",
      input_path, 
      "input tsv file path, default is 1024"
  );

  std::fs::path golden_path("../sample_data/MNIST/neuron1024-l120-categories.tsv");
  app.add_option(
      "--golden",
      golden_path, 
      "golden tsv file path, default is 1024/120"
  );

  CLI11_PARSE(app, argc, argv);

  //Data parallel mode
  Eigen::Matrix<int, Eigen::Dynamic, 1> result;

  if(mode == "CPU_parallel"){
    sparse_dnn::CPUParallel<float> cpu_parallel(
        weight_path,
        bias,
        num_neurons_per_layer,
        num_layers
        );
    result = cpu_parallel.infer(input_path, 60000);
  }
  else if(mode == "sequential"){
    sparse_dnn::Sequential<float> sequential(
      weight_path, 
      bias,
      num_neurons_per_layer, 
      num_layers
    );
    result = sequential.infer(input_path, 60000);
  }

  auto golden = sparse_dnn::read_golden(golden_path, 60000);
  if(sparse_dnn::is_passed(result, golden)){
    std::cout << "CHALLENGE PASSED\n";
  }
  else{
    std::cout << "CHALLENGE FAILED\n";
  }
  return 0;
}
