#include <CLI11/CLI11.hpp>
#include <SNIG/utility/reader.hpp>
#include <SNIG/utility/utility.hpp>
#include <vector>

void convert_to_binary(
  const std::fs::path& weight_path,
  const std::fs::path& input_path,
  const std::fs::path& golden_path,
  const size_t num_neurons,
  const size_t sec_size,
  const size_t num_secs,
  const size_t num_layers=1920
);

int main(int argc, char* argv[]) {

  // usage: ./to_binary
  //          --neurons(-n) :  1024, 4096, or 16384
  //          --convert_all :  convert all files (true, false)
  //          --sample_data :  use sample_data (true, false)

  // example1:
  //        ./to_binary --sample_data true
  // example2:
  //        ./to_binary -n 1024
  // example3:
  //        ./to_binary -convert_all true

  // sec_size, num_secs would be caculated automatically based on GPU architecture.

  CLI::App app{"Converter"};

  size_t num_neurons = 1024;
  app.add_option(
    "-n, --num_neurons", 
    num_neurons, 
    "select number of neurons, default is 1024"
  );

  bool convert_all = false;
  app.add_option(
    "--convert_all", 
    convert_all, 
    "convert all files, default is false"
  );

  bool sample_data = false;
  app.add_option(
    "--sample_data", 
    sample_data, 
    "convert sample data to binary file, default is false"
  );

  std::fs::path weight_path;

  std::fs::path input_path;

  std::fs::path golden_path;

  CLI11_PARSE(app, argc, argv);

  size_t sec_size;
  size_t num_secs;

  std::cout << "Benchmark Converter\n";

  //convert sample data
  if(sample_data) {
    size_t neuron{1024};
    input_path  = "../sample_data/MNIST/";
    golden_path = "../sample_data/MNIST/";
    weight_path = "../sample_data/weight/neuron1024/";
    sec_size = snig::get_sec_size<float>(neuron);
    num_secs = neuron / sec_size; 

    convert_to_binary(
      weight_path,
      input_path,
      golden_path,
      neuron,
      sec_size,
      num_secs,
      120
    );
    return;
  }

  //convert all benchmarks
  if(convert_all) {
    std::vector<size_t> neurons_vec{1024, 4096, 16384, 65536};
    input_path = "../dataset/MNIST/";
    golden_path = "../dataset/MNIST/";
    for(auto& neuron : neurons_vec) {
      sec_size = snig::get_sec_size<float>(neuron);
      num_secs = neuron / sec_size; 
      weight_path = "../dataset/weight/neuron" + std::to_string(neuron) + "/";
      convert_to_binary(
        weight_path,
        input_path,
        golden_path,
        neuron,
        sec_size,
        num_secs
      );
    }
    return;
  }

  //convert benchmarks with num_neurons neruons
  sec_size = snig::get_sec_size<float>(num_neurons);
  num_secs = num_neurons / sec_size; 
  input_path = "../dataset/MNIST/";
  golden_path = "../dataset/MNIST/";
  weight_path = "../dataset/weight/neuron" + std::to_string(num_neurons) + "/";

  convert_to_binary(
    weight_path,
    input_path,
    golden_path,
    num_neurons,
    sec_size,
    num_secs
  );



}

void convert_to_binary(
  const std::fs::path& weight_path,
  const std::fs::path& input_path,
  const std::fs::path& golden_path,
  const size_t num_neurons,
  const size_t sec_size,
  const size_t num_secs,
  const size_t num_layers
) {

  std::cout << "num_neurons : " << num_neurons << std::endl;

  std::cout << "Transforming weight files... \n";
  snig::tsv_file_to_binary_file<float>(
    weight_path,
    num_layers,
    num_neurons,
    num_neurons,
    sec_size,
    num_secs,
    num_neurons * 32
  ); 

  std::cout << "Transforming input files...\n";

  snig::tsv_file_to_binary_file<float>(
    input_path,
    60000,
    num_neurons
  );

  std::cout << "Transforming golden files...\n";

  if(num_layers == 1920) {
    std::vector<int> layers_vec{120, 480, 1920};
    for(int i = 0; i < 3; ++i) {
      snig::tsv_file_to_binary_file(
        golden_path,
        num_neurons,
        layers_vec[i],
        60000
      );
    }
  }
  else {
    snig::tsv_file_to_binary_file(
      golden_path,
      num_neurons,
      num_layers,
      60000
    );
  }
}
