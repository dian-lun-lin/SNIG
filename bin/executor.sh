#usage: $1 select mode(BF_one_gpu, BF_one_gpu_cudagraph, BF_multiple_gpus, SNIG_cudagraph, or SNIG_taskflow), default is BF_one_gpu"
#       $2 neurons_per_layer
#       $3 num_layers
#       $4 num_device
#       $5 is_test_data 0, 1

get_bias() {

  if [[ "$1" == "1024" ]]; then
    bias="-0.3"
  elif [[ "$1" == "4096" ]]; then
    bias="-0.35"
  elif [[ "$1" == "16384" ]]; then
    bias="-0.4"
  elif [[ "$1" == "65536" ]]; then
    bias="-0.45"
  fi

}



get_command() {
  get_bias $2
  if [[ $5 == 1 ]]; then
    if [[ "$1" == "CPU_parallel" || "$1" == "sequential" ]]; then
         ./main -m $1 -w ../dataset/weight/neuron$2/ --num_neurons_per_layer $2 --num_layers $3 --input ../dataset/MNIST/sparse-images-$2.b --golden ../dataset/MNIST/neuron$2-l$3-categories.b --bias $bias
    else
         ./main_cuda -m $1 -w ../dataset/test/weight/neuron$2/ --num_neurons_per_layer $2 --num_layers $3 --input ../dataset/test/MNIST/sparse-images-$2.b --golden ../dataset/test/MNIST/neuron$2-l$3-categories.b --bias $bias --num_gpus $4

    fi

  else
    if [[ "$1" == "CPU_parallel" || "$1" == "sequential" ]]; then
         ./main -m $1 -w ../dataset/weight/neuron$2/ --num_neurons_per_layer $2 --num_layers $3 --input ../dataset/MNIST/sparse-images-$2.b --golden ../dataset/MNIST/neuron$2-l$3-categories.b --bias $bias
    else
         ./main_cuda -m $1 -w ../dataset/weight/neuron$2/ --num_neurons_per_layer $2 --num_layers $3 --input ../dataset/MNIST/sparse-images-$2.b --golden ../dataset/MNIST/neuron$2-l$3-categories.b --bias $bias --num_gpus $4

    fi
    
  fi
}

get_command $1 $2 $3 $4 $5
