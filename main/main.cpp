#include <SparseDNN/SparseDNN.hpp>

int main(int argc, char* argv[]) {

  // TODO:
  // usage: ./main -m sequential 
  

  // Non-RAII
  sparse_dnn::Sequential<float> sequential("/home/dian-lun/dian/GraphChallenge_SparseDNN/dataset/weight/neuron1024", 60000, 1024, 120, -.3f);
  
  // TODO:
  // 1. scoring
  // 2. measure the cpu runtime
  sequential.infer("/home/dian-lun/dian/GraphChallenge_SparseDNN/dataset/MNIST/sparse-images-1024.tsv", 
		  "/home/dian-lun/dian/GraphChallenge_SparseDNN/dataset/MNIST/neuron1024-l120-categories.tsv"
		  );



  return 0;
}
