#pragma once

namespace sparse_dnn{
  template<typename T>
  struct CSRMatrix{
    size_t* row_array;
    size_t* col_array;
    T* data_array;
  };

  template<typename T>
  struct CSCMatrix{
    size_t* col_array;
    size_t* row_array;
    T* data_array;
  };

  template<typename T>
  struct SparseArray{
    size_t* index_array;
    T* data_array;
  };

}


