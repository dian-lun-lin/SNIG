#pragma once
#include <CLI11/CLI11.hpp>
#include <iostream>
#include <SparseDNN/utility/reader.hpp>


int main(int argc, char* argv[]) {

  // usage: ./tsv_file_to_binary --neurons(-n) 1024 --layers(-l) 1920 
  //          --weight_dir(-w) ../sample_data/weight/neuron1024/
  //          --input_path(-i)  ../sample_data/MNIST/sparse-images-1024.tsv
  //          --golden_path(-g)  ../sample_data/MNIST/neuron1024-l120-categories.tsv

  //          --golden_all true  Convert all golden files

  // COL_BLK, N_SLAB would be caculated automatically, based on GPU architecture.

  CLI::App app{"Converter"};

  int num_neurons_per_layer = 1024;
  app.add_option("-n, --neurons", 
    num_neurons_per_layer, 
    "select number of neurons, default is 1024");

  int num_layers = 120;
  app.add_option("-l, --layers", 
    num_layers, 
    "select number of layers, default is 120");

  bool golden_all = true;
  app.add_option("--golden_all", 
    golden_all, 
    "this would convert all golden files with the same neurons. Otherwise only specific num_layers and num_neurons would be converted. Default is true");

  std::fs::path weight_path("../sample_data/weight/neuron1024/");
  app.add_option("-w, --weight_path", 
    weight_path, 
    "select directory of weights. Output binary files would also be generated here. Default is ../sample_data/weight/neuron1024/");

  std::fs::path input_path("../sample_data/MNIST/");
  app.add_option("-i, --input_path", 
    input_path, 
    "select input path. Output binary files would also be generated here. Default is ../sample_data/MNIST/");

  std::fs::path golden_path("../sample_data/MNIST/");
  app.add_option("-g, --golden_path", 
    golden_path, 
    "select golden path. Output binary files would also be generated here. Default is ../sample_data/MNIST/");

  CLI11_PARSE(app, argc, argv);

  int COL_BLK;
  int N_SLAB;

  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  int max_num_per_block = props.sharedMemPerBlock / sizeof(float);

  if(num_neurons_per_layer <= max_num_per_block){
    COL_BLK = num_neurons_per_layer;
  }
  else{
    int max_divisor = 2;
    while((num_neurons_per_layer % max_divisor != 0) || (max_num_per_block < (num_neurons_per_layer / max_divisor))){
      ++max_divisor;
    }
    COL_BLK = num_neurons_per_layer / max_divisor;
  }

  N_SLAB = num_neurons_per_layer / COL_BLK; 

  sparse_dnn::tsv_file_to_binary_file<float>(
    weight_path,
    num_layers,
    num_neurons_per_layer,
    num_neurons_per_layer,
    COL_BLK,
    N_SLAB,
    10000
  ); 

  sparse_dnn::tsv_file_to_binary_file<float>(
    input_path,
    60000,
    num_neurons_per_layer
  );

  if(!golden_all){
    sparse_dnn::tsv_file_to_binary_file<float>(
      golden_path,
      num_neurons_per_layer,
      num_layers,
      60000
    );
  }
  else{
    for(int i = 120; i <= 1920; i *= 4){
      sparse_dnn::tsv_file_to_binary_file<float>(
        golden_path,
        num_neurons_per_layer,
        i,
        60000
      );
    }
  }
}
