# running all layers and all features
# for GPU
# usage: $1 mode $2 num_device $3 output_dir
mode=$1
num_dev=$2
output_dir=$3
bias=(-.3 -.35 -.4 -.45)
Nneuron=(1024 4096 16384 65536)

for (( I = 0; I < 4; ++I ))
do
  for maxLayer in 120 480 1920
  do
    /usr/bin/time -f "Elapsed real time : %E (hr/min/sec)\nCPU spent in kernel mode : %S sec\nCPU spent in user mode : %U sec\nPercentage of the CPU that this job got : %P\nMax RSS : %M Kbytes" -o ./$output_dir/$mode-${Nneuron[I]}-$maxLayer.out ./main_cuda -m $mode -w "../dataset/weight/neuron${Nneuron[I]}" --num_neurons_per_layer ${Nneuron[I]} --num_layers $maxLayer -b ${bias[I]} --input "../dataset/MNIST/sparse-images-${Nneuron[I]}.b" --golden "../dataset/MNIST/neuron${Nneuron[I]}-l$maxLayer-categories.b" --num_device $num_dev
  done
done
