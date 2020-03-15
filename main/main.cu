#pragma once
#include <CLI11/CLI11.hpp>
#include <experimental/filesystem>

#include <SparseDNN/SparseDNN_GPU.hpp>
#include <SparseDNN/utility/reader.hpp>
#include <SparseDNN/utility/scoring.hpp>

namespace std {
  namespace fs = experimental::filesystem;
}


//#include <thread>

int main(int argc, char* argv[]) {
   

  // usage: ./main -m sequential 
  //        ./main -m CPU_parallel
  //        ./main -m GPU_cusparse
  CLI::App app{"SparseDNN"};
  std::string mode = "sequential";
  app.add_option("-m, --mode", 
    mode, 
    "select mode(sequential/GPU), default is sequential");

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
  Eigen::Matrix<int, Eigen::Dynamic, 1> result;
  //Data parallel mode
  if(mode == "GPU_cusparse"){
    sparse_dnn::GPUCusparse<float> GPU_cusparse(
      weight_path, 
      bias,
      num_neurons_per_layer, 
      num_layers
    );
    result = GPU_cusparse.infer(input_path, 60000);
  }
  else if(mode == "GPU_baseline"){

    sparse_dnn::GPUBaseline<float> GPU_baseline(
      weight_path, 
      bias,
      num_neurons_per_layer, 
      num_layers,
      1024
    );
    result = GPU_baseline.infer(input_path, 60000);
  }
  else if(mode == "GPU_cugraph"){
    sparse_dnn::GPUCugraph<float> GPU_cugraph(
      weight_path, 
      bias,
      num_neurons_per_layer, 
      num_layers,
      1024
    );
    result = GPU_cugraph.infer(input_path, 60000);
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
