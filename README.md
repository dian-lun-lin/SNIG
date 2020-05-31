# SNIG

Accelerated Large Sparse Neural Network Inference using Task Graph Parallelism 

# Problem Statement

SNIG is an inference engine for the [MIT/Amazon/IEEE HPEC Sparse Deep Neural Network Graph Challenge](./https://graphchallenge.mit.edu/challenges). 
We develop highly optimized inference kernels and leverage the power of CUDA Graphs to enable efficient decomposition of model and data parallelisms.

# Download the Input Dataset and Golden Reference

The dataset is available at https://graphchallenge.mit.edu/data-sets

TO DL: you may draw a table of benchmar statistics (Table I in the paper)

# Compile and Run SNIG

This is the **most** important section!!!

```bash
~$ mkdir build
~$ cd build
~$ make
```
You will see the executable file 'main_cuda', '' under `bin/`.

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

