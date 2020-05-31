#pragma once
#include <functional>
#include <algorithm>

namespace snig {

template<typename T>
size_t get_sec_size(const size_t num_neurons);

inline
float average_zero_percent_in_non_empty_rows(
  int* rlenY,
  int* rowsY,
  size_t num_features,
  size_t nerowsY
);

inline
void num_nonzero_row_percent(std::vector<size_t>& nerows);

inline
void num_nonzero_row(std::vector<size_t>& nerows);

//-----------------------------------------------------------------------------
//Definition of utility function
//-----------------------------------------------------------------------------

template<typename T>
size_t get_sec_size(const size_t num_neurons) {

  //only for the same GPUs
  //
  //get tuned shared memory size
  //num_neurons must be divisible by shared memory (a.k.a. sec_size)
  //only for double float
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  size_t sec_size{0};

  size_t max_num_per_block = props.sharedMemPerBlock / sizeof(T);
  if(num_neurons <= max_num_per_block) {
    sec_size = num_neurons;
  }
  else{
    int max_divisor = 2;
    while((num_neurons % max_divisor != 0) || 
          (max_num_per_block < (num_neurons / max_divisor))) {
      ++max_divisor;
    }
    sec_size = num_neurons / max_divisor;
  }
  return sec_size;
}

inline
float average_zero_percent_in_non_empty_rows(
  int* rlenY,
  int* rowsY,
  size_t num_features,
  size_t nerowsY
) {
  int total_zero{0};
  for(int i = 0; i < nerowsY; ++i) {
    total_zero += num_features - rlenY[rowsY[i]];
  }
  return (100 * (total_zero / float(num_features * nerowsY)));

};

inline
void num_nonzero_row_percent(std::vector<size_t>& nerows)
{
  std::cout << "\nPerencetage of number of nonzero rows of each GPU : ";
  size_t total = std::accumulate(nerows.begin(), nerows.end(), 0);
  for(auto& num : nerows) {
    std::cout << (100 * (float(num) / total)) << "% ";
  }
}

inline
void num_nonzero_row(std::vector<size_t>& nerows)
{
  std::cout << "\nNumber of nonzero rows of each GPU : ";
  size_t total = std::accumulate(nerows.begin(), nerows.end(), 0);
  for(auto& num : nerows) {
    std::cout << num << " ";
  }
}

}// end of namespace snig ----------------------------------------------
