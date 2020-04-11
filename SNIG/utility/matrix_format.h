#pragma once

namespace snig{

  template<typename T>
  struct CSRMatrix{
    int* row_array;
    int* col_array;
    T* data_array;
  };

  template<typename T>
  struct CSCMatrix{
    int* col_array;
    int* row_array;
    T* data_array;
  };

  template<typename T>
  struct SparseArray{
    int* index_array;
    T* data_array;
  };

  template <typename T>
  struct Triplet{
    int row;
    int col;
    T value;

    bool operator < (const Triplet& b) const {
      return row < b.row;
    }

    Triplet(int r, int c, T v)
    : row{r},
      col{c},
      value{v}
    {
    }

  };

}// end of namespace snig ----------------------------------------------


