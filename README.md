# SNIG

Accelerated Large Sparse Neural Network Inference using Task Graph Parallelism 

# Problem Statement

SNIG is an inference engine for the [MIT/Amazon/IEEE HPEC Sparse Deep Neural Network Graph Challenge](./https://graphchallenge.mit.edu/challenges). 

We develop highly optimized inference kernels and leverage the power of CUDA Graphs to enable efficient decomposition of model and data parallelisms.

# Step 1 : Compile SNIG

```bash
~$ mkdir build
~$ cd build
~$ make
```
You will see executable files ('snig' and 'to_binary') under `bin/`.
To run SNIG with smallest benchmark under 1 GPU, you can simply type :

```
cd bin/
~$ ./to_binary --sample_data true
~$ ./snig
```

To run other benchmarks, you need to download the dataset from MIT/IEEE/Amazon Graph Challenge.

# Step 2: Download the Model, Input Dataset, and Golden Reference

The dataset is available at https://graphchallenge.mit.edu/data-sets

First, create directories to store the dataset :
```
~$ mkdir ./dataset
~$ mkdir ./dataset/MNIST
~$ mkdir ./dataset/weight
```
After downloading and extracting the dataset, 
you need to move the input dataset and golden reference to ```./dataset/MNIST``` and the model to ```./dataset/weight/```, respectively.

The file paths should be like :

```
./dataset/weight/neuron1024/{tsv files}
./datast/MNIST/neuron1024-l120-categories.tsv
./dataset/MNIST/sparse-images-1024.tsv
```

# Step 3: Transform the Input Dataset to Binary Format

Computing the raw dataset is extremely time-consuming.
To execute SNIG, you need to transform the input dataset to binary format first.
**Make sure all the data is stored in** ```./dataset```

First, 
``` 
  ~$ cd bin/ 
```
To convert one bencmark :
```
~$ ./to_binary --num_neurons --num_layers
```
For example, ``` ~$ ./to_binary 16384 1920 ``` would convert benchmark with 16384 neurons and 1920 layers to binary file.

To convert all benchmarks :
```
~$ ./to_binary --conver_all true
```
Note that converting all benchmarks would take some time.

# Step 4 : Run SNIG on a Specific Benchmark
First, 
```
  cd bin/
```

You can either use ```~$ ./snig ``` for setting details or our srcipt ```~$ ./executor.sh``` with tuned parameters
## For ```~$ ./executor.sh``` :
```
  ~$ ./execuator.sh -mode (SNIG, BF, SNIG_pipeline) -num_neurons -num_layers -num_gpus
```
For example, ```~$ ./executor.sh SNIG 65536 1920 4``` use SNIG to peform benchmark with 65536 neurons and 1920 layers under 4 GPUs.

Check ``` ~$ ./executor.sh -h``` for more details

## For ```~$ ./snig``` :
```
  ~$ ./snig -mode -weight -input -golden -num_neurons -num_layers -bias --num_gpus --num_weight_buffers --input_batch_size -thread_dimension
```
Check ```~$ ./snig -h ``` for more detials

## Command Options for ```~$./snig```
```
  -h,--help                   Print this help message and exit
  -m,--mode                   select mode(SNIG, SNIG_pipeline, or BF), default is SNIG
  -w,--weight                 weight directory path, default is ../sample_data/weight/neuron1024/
  -i,--input                  input binary file path, default is ../sample_data/MNIST/sparse-images-1024.b
  -g,--golden                 golden binary file path, default is ../sample_data/MINIST/neuron1024-l120-categories.b
  -n,--num_neurons            total number of neurons, default is 1024
  -l,--num_layers             total number of layers, default is 120
  -b,--bias                   bias, default is -0.3
  --num_gpus                  number of GPUs, default is 1
  --num_weight_buffers        number of weight buffers, default is 2
  --input_batch_size          number of input bath size, default is 5000
  -t,--thread_dimension       thread dimension for inference kernel, need 3 parameters, default is 2 512 1
```

# Results

The table below summarizes our result on a machine (mention explicitly the GPU specification) ??

## Baseline Implementation

You can refere to [bf.hpp](./SNIG/bf/bf.hpp) and [kernel.hpp](./SNIG/bf/kernel.hpp) for our implementation of the 
[A GPU Implementation of the Sparse Deep Neural Network Graph Challenge](https://doi.org/10.1109/HPEC.2019.8916223),
and [snig_pipeline](./SNIG/snig/snig_pipeline.hpp) and [kernel.hpp](./SNIG/snig/kernel.hpp) for our implementation of the [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://papers.nips.cc/paper/8305-gpipe-efficient-training-of-giant-neural-networks-using-pipeline-parallelism).

# Reference

+ [A GPU Implementation of the Sparse Deep Neural Network Graph Challenge](https://doi.org/10.1109/HPEC.2019.8916223)
+ [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://papers.nips.cc/paper/8305-gpipe-efficient-training-of-giant-neural-networks-using-pipeline-parallelism)


TO DL: you may draw a table of benchmar statistics (Table I in the paper)
