#usage: $1 for mode {GPU_cusparse, GPU_baseline, CPU_parallel, sequential}
#       $2 neurons_per_layer
#       $3 num_layers
if [[ "$1" == "GPU_cusparse" || "$1" == "GPU_baseline" ]]; then
     ./main_cuda -m $1 -w ../dataset/weight/neuron$2/ --num_neurons_per_layer $2 --num_layers $3 --input ../dataset/MNIST/sparse-images-$2.tsv --golden ../dataset/MNIST/neuron$2-l$3-categories.tsv

else
     ./main -m $1 -w ../dataset/weight/neuron$2/ --num_neurons_per_layer $2 --num_layers $3 --input ../dataset/MNIST/sparse-images-$2.tsv --golden ../dataset/MNIST/neuron$2-l$3-categories.tsv

fi
