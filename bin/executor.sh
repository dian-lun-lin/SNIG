#usage: $1 mode (BF,  SNIG, or GPipe), default is SNIG"
#       $2 num_neurons
#       $3 num_layers
#       $4 num_gpus
#       $5 batch_size
#       $6 num_weight_buffers
#       $7, $8, $9 thread_dimension

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

  if [[ "$1" == "-h" ]]; then
    echo "usage : ./excuator.sh mode num_neurons num_layers num_gpus input_batch_size num_weight_buffers thread_dimension"
    echo ""
    echo "\"./executor.sh SNIG 65536 1920 4\" use SNIG to peform the benchmark with 65536 neurons and 1920 layers under 4 GPUs"
    echo "\"./executor.sh BF 4096 1920 2\" use BF to perform the benchmark with 4096 neurons and 1920 layers under 2 GPUs"
    exit
  fi

  default_mode="SNIG"
  default_num_neurons=1024
  default_layers=120
  default_num_gpus=1
  default_input_batch_size=5000
  default_num_weight_buffers=2
  default_threads=(2 512 1)

  mode=${1:-$default_mode}
  num_neurons=${2:-$default_num_neurons}
  num_layers=${3:-$default_layers}
  num_gpus=${4:-$default_num_gpus}
  input_batch_size=${5:-$default_input_batch_size}
  num_weight_buffers=${6:-$default_num_weight_buffers}
  threads_dim0=${7:-${default_threads[0]}}
  threads_dim1=${8:-${default_threads[1]}}
  threads_dim2=${9:-${default_threads[2]}}

  get_bias $num_neurons
  ./snig -m $mode -w ../dataset/weight/neuron$num_neurons/ --num_neurons $num_neurons --num_layers $num_layers --input ../dataset/MNIST/sparse-images-$num_neurons.b --golden ../dataset/MNIST/neuron$num_neurons-l$num_layers-categories.b --bias $bias --num_gpus $num_gpus --input_batch_size $input_batch_size --num_weight_buffers $num_weight_buffers -t $threads_dim0 $threads_dim1 $threads_dim2 

}

get_command $1 $2 $3 $4 $5 $6 $7 $8 $9
