# SNIG

Accelerated Large Sparse Neural Network Inference using Task Graph Parallelism 

# Problem Statement

SNIG is an inference engine for the [MIT/Amazon/IEEE HPEC Sparse Deep Neural Network Graph Challenge](./https://graphchallenge.mit.edu/challenges). 

We develop highly optimized inference kernels and leverage the power of CUDA Graphs to enable efficient decomposition of model and data parallelisms.

# Compile and Run SNIG

```bash
~$ mkdir build
~$ cd build
~$ make
```
You will see executable files 'snig', 'to_binary' under `bin/`.
To run SNIG with smallest benchmark under 1 GPU, you can simply type :

```
cd bin/
~$ ./to_binary
~$ ./snig
```

To run other benchmarks, you need to download the dataset from MIT/IEEE/Amazon Graph Challenge.

# Download the Model, Input Dataset, and Golden Reference

The dataset is available at https://graphchallenge.mit.edu/data-sets

After downloading and extracting the dataset, 
you need to move the input dataset and golden reference to ```./dataset/MNIST``` and the model to ```./dataset/weight/```

The file paths would be like:

```
./dataset/weight/neuron1024/{tsv files}
./datast/MNIST/neuron1024-l120-categories.tsv
./dataset/MNIST/sparse-images-1024.tsv
```





## Transform the Input Dataset to Binary Format

Computing the raw dataset is extremely time-consuming.
To execute SNIG, you need to transform the input dataset to binary format first.


## Run SNIG on a Specific Benchmark

???

```bash
~$ ???
```

## Additional Command Options

# Results

The table below summarizes our result on a machine (mention explicitly the GPU specification) ??

## Baseline Implementation

You can refere to [?.hpp](./file/to/bf/method) and ?.hpp for our implementation of the [BF method](...) and the [GPipe](...).

# Reference

+ Link to the BF paper
+ Link to the GPipe paper


TO DL: you may draw a table of benchmar statistics (Table I in the paper)
