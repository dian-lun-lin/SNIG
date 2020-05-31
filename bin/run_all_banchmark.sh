# running all layers and all features
# for GPU
# usage: $1 mode (BF_one_gpu, BF_one_gpu_cudagraph, BF_multiple_gpus, SNIG_taskflow, SNIG_cudagraph)
#        $2 num_device (1, 2, 3, 4)
#        $3 run_times (1, 2,..., 10)
#        $4 output_directory
mode=$1
num_dev=$2
run_times=$3
output_file=$4
bias=(-.3 -.35 -.4 -.45)
Nneuron=(1024 4096 16384 65536)

#for (( dev = num_dev; dev >= 2; --dev))
#for (( bs = 0; bs < 4; ++bs))
  #do
  for (( K = 0; K < run_times; ++K ))
  do
    for (( I = 0; I < 4; ++I ))
    do
      for maxLayer in 1920
      do
        ./main_cuda -m $mode -w "../dataset/weight/neuron${Nneuron[I]}" --num_neurons_per_layer ${Nneuron[I]} --num_layers $maxLayer -b ${bias[I]} --input "../dataset/MNIST/sparse-images-${Nneuron[I]}.b" --golden "../dataset/MNIST/neuron${Nneuron[I]}-l$maxLayer-categories.b" --num_input_batch_size 5000 --num_device $num_dev >> $output_file/run_time$K.out
    
      done
    done
  done
#done
