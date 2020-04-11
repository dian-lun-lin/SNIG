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

}// end of namespace snig ----------------------------------------------
