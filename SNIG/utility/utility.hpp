#pragma once
#include <functional>
#include <algorithm>

namespace snig {

inline
float average_zero_percent_in_non_empty_rows(
  int* rlenY,
  int* rowsY,
  size_t num_features,
  size_t nerowsY
);

//-----------------------------------------------------------------------------
//Definition of utility function
//-----------------------------------------------------------------------------

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
