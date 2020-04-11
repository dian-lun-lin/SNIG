#pragma once
#include <assert.h>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", 
            cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

inline
std::string checkType(const cudaGraphNodeType& type) {
  std::string str;
  if(type == 0) {
    str = "kenel";
  }
  else if(type == 1) {
    str = "memcpy";
  }
  else if(type == 2) {
    str = "memset";
  }
  else if(type == 3) {
    str = "host";
  }
  else if(type == 4) {
    str = "Node which executes an embedded graph";
  }
  else if(type == 5) {
    str = "empty";
  }
  return str;
}
