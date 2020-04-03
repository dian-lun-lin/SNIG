#include <CLI11/CLI11.hpp>
#include <SparseDNN/SparseDNN_GPU.hpp>
#include <SparseDNN/utility/reader.hpp>
#include <SparseDNN/utility/scoring.hpp>

int main(int argc, char* argv[]) {

  //  All files should be converted to binary first

  // usage: 
  //        --mode(-m) GPU_baseline, or GPU_cugraph
  //        --weight path of weight
  //        --bias(-b) bias
  //        --num_neurons_per_layer 1024, 4096, 16384, or 65536
  //        --num_layers num_layers 120, 480, or 1920
  //        --input_path path of input
  //        --golden_path path of golden

  //example1:  
  //        ./main_cu
  //example2:  
  //        ./main_cu  -m GPU_cugraph --weight ../sample_data/weight/neuron1024/ --num_neurons_per_layer 1024 --num_layers 120 --input_path ../sample_data/MNIST/sparse-images-1024.b --golden_path ../sample_data/MNIST/neuron1024-l120-categories.b

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

  //binary format is not completed yet.
  if(mode == "GPU_cusparse") {
    sparse_dnn::GPUCusparse<float> GPU_cusparse(
      weight_path, 
      bias,
      num_neurons_per_layer, 
      num_layers
    );
    result = GPU_cusparse.infer(input_path, 60000);
  }
  if(mode == "GPU_baseline") {
    sparse_dnn::GPUBaseline<float> GPU_baseline(
      weight_path, 
      bias,
      num_neurons_per_layer, 
      num_layers
    );
    result = GPU_baseline.infer(input_path, 60000);
  }
  else if(mode == "GPU_cugraph") {
    sparse_dnn::GPUCugraph<float> GPU_cugraph(
      weight_path, 
      bias,
      num_neurons_per_layer, 
      num_layers
    );
    result = GPU_cugraph.infer(input_path, 60000, true);
  }
  else if(mode == "GPU_decompose") {
    sparse_dnn::GPUDecomp<float> GPU_decompose(
      weight_path, 
      bias,
      num_neurons_per_layer, 
      num_layers
    );
    result = GPU_decompose.infer(input_path, 60000, 5000, 10);
  }
  else if(mode == "GPU_taskflow") {
    sparse_dnn::GPUTaskflow<float> GPU_taskflow(
      weight_path, 
      bias,
      num_neurons_per_layer, 
      num_layers
    );
    result = GPU_taskflow.infer(input_path, 60000, 5000, 10);
  }
  else if(mode == "GPU_decompose_multiple") {
    sparse_dnn::GPUDecompMulti<float> GPU_decomp_multi(
      weight_path, 
      bias,
      num_neurons_per_layer, 
      num_layers
    );
    result = GPU_decomp_multi.infer(input_path, 60000, 5000, 10, 1);
  }
  auto golden = sparse_dnn::read_golden_binary(golden_path);
  if(sparse_dnn::is_passed(result, golden)) {
    std::cout << "CHALLENGE PASSED\n";
  }
  else{
    std::cout << "CHALLENGE FAILED\n";
  }
  return 0;
}
