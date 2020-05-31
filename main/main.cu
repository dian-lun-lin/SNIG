#include <CLI11/CLI11.hpp>
#include <SNIG/SNIG.hpp>
#include <SNIG/utility/reader.hpp>
#include <SNIG/utility/scoring.hpp>
#include <iostream>

int main(int argc, char* argv[]) {

  //  ***All files should be converted to binary first***

  // usage: 
  //        --mode(-m)
  //        --weight(-w)                 :  directory path of weight
  //        --input(-i)                  :  file path of input
  //        --golden(-g)                 :  file path of golden
  //        --num_neurons_per_layer(-n)  :  number of neurons 1024, 4096, 16384, or 65536
  //        --num_layers(-l)             :  number of layers 120, 480, or 1920
  //        --bias(-b)                   :  bias
  //        --num_gpus                   :  number of GPUs 1, 2, 3, 4, ...
  //        --input_batch_size           :  input batch size
  //        --num_weight_buffers         :  number of weight buffers, must be even and factor of 120

  //example1:  
  //        ./main_cu

  //example2:  
  //        ./main_cu  -m BF_one_gpu -w ../sample_data/weight/neuron1024/ -n 1024 -l 120 -i ../sample_data/MNIST/sparse-images-1024.b -g ../sample_data/MNIST/neuron1024-l120-categories.b

  CLI::App app{"SNIG"};

  std::string mode = "SNIG";
  app.add_option(
    "-m, --mode", 
    mode, 
    "select mode(BF_one_gpu, BF_multiple_gpus, SNIG, or SNIG_pipeline), default is SNIG"
  );

  std::fs::path weight_path("../sample_data/weight/neuron1024/");
  app.add_option(
    "-w, --weight",
    weight_path,
    "weight directory path"
  )->check(CLI::ExistingDirectory);

  std::fs::path input_path("../sample_data/MNIST/sparse-images-1024.b");
  app.add_option(
      "-i, --input",
      input_path, 
      "input binary file path, default is ../sample_data/MNIST/sparse-images-1024.b"
  )->check(CLI::ExistingFile);

  std::fs::path golden_path("../sample_data/MNIST/neuron1024-l120-categories.b");
  app.add_option(
      "-g, --golden",
      golden_path, 
      "golden binary file path, default is ../sample_data/MINIST/neuron1024-l120-categories.b"
  );
  

  size_t num_neurons_per_layer = 1024;
  app.add_option(
    "-n, --num_neurons_per_layer", 
    num_neurons_per_layer, 
    "total number of neurons per layer, default is 1024"
  );

  size_t num_layers = 120;
  app.add_option(
    "-l, --num_layers",
    num_layers, 
    "total number of layers, default is 120"
  );

  float bias = -0.3f;
  app.add_option(
    "-b, --bias",
    bias,
    "bias, default is -0.3"
  );

  size_t num_gpus = 1;
  app.add_option(
    "--num_gpus", 
    num_gpus,
    "number of GPUs, default is 1"
  );
  
  size_t num_weight_buffers = 2;
  app.add_option(
    "--num_weight_buffers", 
    num_weight_buffers,
    "number of weight buffers, default is 2"
  );
  
  size_t batch_size = 5000;
  app.add_option(
    "--input_batch_size", 
    batch_size,
    "number of input bath size, default is 5000"
  );

  CLI11_PARSE(app, argc, argv);
  Eigen::Matrix<int, Eigen::Dynamic, 1> result;
  std::cout << "Current mode: " << mode << std::endl;

  if(mode == "SNIG") {
    snig::SNIG<float> snig(
      weight_path, 
      bias,
      num_neurons_per_layer, 
      num_layers
    );
    result = snig.infer(input_path, 60000, batch_size, num_weight_buffers, num_gpus);
  }
  else if(mode == "SNIG_pipeline") {
    snig::SNIGPipeline<float> snig_pipeline(
      weight_path, 
      bias,
      num_neurons_per_layer, 
      num_layers
    );
    result = snig_pipeline.infer(input_path, 60000, batch_size, num_gpus);
  }
  else if(mode == "BF") {
    //only perform initial partition since we don't have NVLink 
    snig::BF<float> bf(
      weight_path, 
      bias,
      num_neurons_per_layer, 
      num_layers
    );
    result = bf.infer(input_path, 60000, num_gpus);
  }
  else {
    using namespace std::literals::string_literals;
    throw std::runtime_error("Error mode. Please correct your mode name"s);
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
