#pragma once

namespace sparse_dnn{
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

}


