bias=(-.3 -.35 -.4 -.45)
Nneuron=(1024 4096 16384 65536)
mode="GPU_taskflow"
for (( I = 0; I < 4; ++I ))
do
  for maxLayer in 120 480 1920
  do
    /usr/bin/time -f "Elapsed real time : %E (hr/min/sec)\nCPU spent in kernel mode : %S sec\nCPU spent in user mode : %U sec\nPercentage of the CPU that this job got : %P\nMax RSS : %M Kbytes" -o ./result_0325/$mode-${Nneuron[I]}-$maxLayer.txt ./main_cuda -m $mode -w "/home/dian-lun/dian/GraphChallenge_SparseDNN/dataset/weight/neuron${Nneuron[I]}" --num_neurons_per_layer ${Nneuron[I]} --num_layers $maxLayer -b ${bias[I]} --input "/home/dian-lun/dian/GraphChallenge_SparseDNN/dataset/MNIST/sparse-images-${Nneuron[I]}.b" --golden "/home/dian-lun/dian/GraphChallenge_SparseDNN/dataset/MNIST/neuron${Nneuron[I]}-l$maxLayer-categories.b"
  done
done
