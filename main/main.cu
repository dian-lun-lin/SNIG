#include <CLI11/CLI11.hpp>
#include <SNIG/SNIG_GPU.hpp>
#include <SNIG/utility/reader.hpp>
#include <SNIG/utility/scoring.hpp>
#include <iostream>

int main(int argc, char* argv[]) {

  //  All files should be converted to binary first

  // usage: 
  //        --mode(-m)
  //        --weight path of weight
  //        --bias(-b) bias
  //        --num_neurons_per_layer 1024, 4096, 16384, or 65536
  //        --num_device 1, 2, 3, 4
  //        --num_layers num_layers 120, 480, or 1920
  //        --input_path path of input
  //        --golden_path path of golden

  //example1:  
  //        ./main_cu
  //example2:  
  //        ./main_cu  -m BF_one_gpu --weight ../sample_data/weight/neuron1024/ --num_neurons_per_layer 1024 --num_layers 120 --input_path ../sample_data/MNIST/sparse-images-1024.b --golden_path ../sample_data/MNIST/neuron1024-l120-categories.b

  CLI::App app{"SNIG"};
  std::string mode = "BF_one_gpu";
  app.add_option(
    "-m, --mode", 
    mode, 
    "select mode(BF_one_gpu, BF_one_gpu_cudagraph, BF_multiple_gpus, SNIG_cudagraph, or SNIG_taskflow), default is bf_one_gpu"
  );

  std::fs::path weight_path("../sample_data/weight/neuron1024/");
  app.add_option("-w, --weight", weight_path, "weight directory path")
    ->check(CLI::ExistingDirectory);

  size_t num_neurons_per_layer = 1024;
  app.add_option(
    "--num_neurons_per_layer", 
    num_neurons_per_layer, 
    "total number of neurons per layer, default is 1024"
  );

  size_t num_dev = 1;
  app.add_option(
    "--num_device", 
    num_dev,
    "number of GPUs, default is 1"
  );

  size_t num_layers = 120;
  app.add_option(
      "--num_layers",
      num_layers, 
      "total number of layers, default is 120"
  );

  float bias = -0.3f;
  app.add_option("-b, --bias", bias, "bias");


  std::fs::path input_path("../sample_data/MNIST/sparse-images-1024.b");
  app.add_option(
      "--input",
      input_path, 
      "input binary file path, default is 1024"
  );

  std::fs::path golden_path("../sample_data/MNIST/neuron1024-l120-categories.b");
  app.add_option(
      "--golden",
      golden_path, 
      "golden binary file path, default is 1024/120"
  );
  CLI11_PARSE(app, argc, argv);
  Eigen::Matrix<int, Eigen::Dynamic, 1> result;
  std::cout << "Current mode: " << mode << std::endl;

  //binary format is not completed yet.
  if(mode == "BF_one_gpu") {
    snig::BFOneGpu<float> bf_one(
      weight_path, 
      bias,
      num_neurons_per_layer, 
      num_layers
    );
    result = bf_one.infer(input_path, 60000);
  }

  if(mode == "BF_one_gpu_cudagraph") {
    snig::BFOneGpuCudaGraph<float> bf_one_cudagraph(
      weight_path, 
      bias,
      num_neurons_per_layer, 
      num_layers
    );
    result = bf_one_cudagraph.infer(input_path, 60000);
  }
  if(mode == "BF_multiple_gpus") {
    snig::BFMultiGpu<float> bf_multi(
      weight_path, 
      bias,
      num_neurons_per_layer, 
      num_layers
    );
    result = bf_multi.infer(input_path, 60000, num_dev);
  }
  else if(mode == "SNIG_cudagraph") {
    snig::SNIGCudaGraph<float> snig_cudagraph(
      weight_path, 
      bias,
      num_neurons_per_layer, 
      num_layers
    );
    result = snig_cudagraph.infer(input_path, 60000, 5000, 10, num_dev);
  }
  else if(mode == "SNIG_taskflow") {
    snig::SNIGTaskflow<float> snig_taskflow(
      weight_path, 
      bias,
      num_neurons_per_layer, 
      num_layers
    );
    result = snig_taskflow.infer(input_path, 60000, 5000, 10, num_dev);
  }
  auto golden = snig::read_golden_binary(golden_path);
  if(snig::is_passed(result, golden)) {
    std::cout << "CHALLENGE PASSED\n";
  }
  else{
    std::cout << "CHALLENGE FAILED\n";
  }
  return 0;
}
