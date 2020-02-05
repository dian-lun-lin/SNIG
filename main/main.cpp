#include <SparseDNN/SparseDNN.hpp>

int main(int argc, char* argv[]) {

  // TODO:
  // usage: ./main -m sequential 
  

  // Non-RAII
  sparse_dnn::Sequential<float> sequential("gg", 10, 10, -1.0f);
  
  // TODO:
  // 1. scoring
  // 2. measure the cpu runtime
  //sequential.infer("path-to-input", "path-to-golden");



  return 0;
}
