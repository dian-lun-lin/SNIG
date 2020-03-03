#usage: $1 for mode {GPU, CPU_parallel, sequential}
#       $2 neurons_per_layer
#       $3 num_layers
if [ "$1" = "GPU" ]; then
     ./main_cuda -w ../dataset/weight/neuron$2/ --num_layers $3 --input ../dataset/MNIST/sparse-images-$2.tsv --golden ../dataset/MNIST/neuron$2-l$3-categories.tsv

else
     ./main -m $1 -w ../dataset/weight/neuron$2/ --num_layers $3 --input ../dataset/MNIST/sparse-images-$2.tsv --golden ../dataset/MNIST/neuron$2-l$3-categories.tsv

fi
