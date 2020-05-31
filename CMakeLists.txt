# CMake version
cmake_minimum_required (VERSION 3.9)
MESSAGE(STATUS "CMAKE_ROOT: " ${CMAKE_ROOT})

# Project name
project(SNIG VERSION 1.0.0 LANGUAGES CXX CUDA)

# Turn on the verbose
set(CMAKE_VERBOSE_MAKEFILE ON)

# defult release build
set(SNIG_DEFAULT_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${SNIG_DEFAULT_BUILD_TYPE}'")
  set(
    CMAKE_BUILD_TYPE "${SNIG_DEFAULT_BUILD_TYPE}" 
    CACHE
    STRING "Choose the type of build." 
    FORCE
  )
  # Set the possible values of build type for cmake-gui
  set_property(
    CACHE 
    CMAKE_BUILD_TYPE 
    PROPERTY STRINGS
    "Debug" "Release"
  )
endif()

#not test yet
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "7.3")
	  message(FATAL_ERROR "\n SNIG requires g++ at least v7.3")
  endif()
## clang++
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "6.0")
	  message(FATAL_ERROR "\n SNIG requires clang++ at least v6.0")
  endif() 
endif()

# Args
option(SDNN_BUILD_TESTS "Enables build of tests" ON)

# installation path
set(SDNN_UTEST_DIR ${PROJECT_SOURCE_DIR}/unittests)
set(SDNN_3RD_PARTY_DIR ${PROJECT_SOURCE_DIR}/3rd-party)

# include test
include(CTest)

#compiler setting 
#add_library(error_settings INTERFACE)
#add_library(snig::error_settings ALIAS error_settings)

#target_compile_options(
  #error_settings
  #INTERFACE
  #$<$<COMPILE_LANGUAGE:Clang>:-Wall -Wextra -Wfatal-errors>
  #$<$<COMPILE_LANGUAGE:GNU>:-Wall -Wextra -Wfatal-errors>
#)

##optmization
#add_library(optimize_settings INTERFACE)
#add_library(snig::optimize_settings ALIAS optimize_settings)

#target_compile_options(
  #optimize_settings INTERFACE
  #$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:Clang>>:-O2 -march=native>
  #$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:GNU>>:-O2 -march=native>
  #$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:Clang>>:-O0 -g>
  #$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:GNU>>:-O0 -g>
#)
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}" "-Wall -Wextra -Wfatal-errors")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}" "-O2 -march=native")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-Xcompiler -fopenmp " )
set(CUDA_NVCC_FLAGS_DEBUG "${CUDA_NVCC_FLAGS_DEBUG}" "-lineinfo")
set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE}" "-O2 -w ")

#default
add_library(default_settings INTERFACE)
add_library(snig::default_settings ALIAS default_settings)
target_link_libraries(
  default_settings 
  INTERFACE 
  snig::error_settings 
  snig::optimize_settings 
)


# CXX target properties
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(CUDA QUIET REQUIRED)
# Thread
find_package(Threads REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)
#OpenMP
find_package(OpenMP REQUIRED)
set(OpenMP_CXX_FLAGS "-fopenmp")

# message
message(STATUS "CMAKE_HOST_SYSTEM: ${CMAKE_HOST_SYSTEM}")
message(STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})
message(STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER})
message(STATUS "CMAKE_CXX_COMPILER_VERSION: " ${CMAKE_CXX_COMPILER_VERSION})
message(STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})
message(STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS "CMAKE_CURRENT_BINARY_DIR: " ${CMAKE_CURRENT_BINARY_DIR})
message(STATUS "CMAKE_EXE_LINKER_FLAGS: " ${CMAKE_EXE_LINKER_FLAGS})
message(STATUS "CMAKE_INSTALL_PREFIX: " ${CMAKE_INSTALL_PREFIX})
message(STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})
message(STATUS "CMAKE_PREFIX_PATH: " ${CMAKE_PREFIX_PATH})
message(STATUS "PROJECT_NAME: " ${PROJECT_NAME})

#include directories
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/3rd-party/)

#cuda
include_directories(/usr/local/cuda/include/)
link_directories(/usr/local/cuda/lib64/)


#--------interface------
add_library(${PROJECT_NAME} INTERFACE)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_14)
target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include/> 
)
#-----------------------

# test
#if(${SDNN_BUILD_TESTS})
 
#enable_testing()
#message(STATUS "Building unit tests ...")
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${SDNN_UTEST_DIR})

#add_executable(reader ${SDNN_UTEST_DIR}/reader.cpp)
#target_link_libraries(reader stdc++fs)
#target_include_directories(reader PRIVATE ${SDNN_3RD_PARTY_DIR}/doctest)
#add_test(tsv_string_to_matrix ${SDNN_UTEST_DIR}/reader -tc=tsv_string_to_matrix)

#add_executable(matrix_operation ${SDNN_UTEST_DIR}/matrix_operation.cpp)
#target_include_directories(matrix_operation PRIVATE ${SDNN_3RD_PARTY_DIR}/doctest)
#add_test(CSR_matrix_to_eigen_sparse ${SDNN_UTEST_DIR}/matrix_operation -tc=CSR_matrix_to_eigen_sparse)

#cuda_add_executable(task ${SDNN_UTEST_DIR}/task.cu)
#target_include_directories(task PRIVATE ${SDNN_3RD_PARTY_DIR}/doctest)
#add_test(GPU_check_nnz ${SDNN_UTEST_DIR}/task -tc=check_nnz)
#add_test(GPU_task ${SDNN_UTEST_DIR}/task -tc=task_GPU)

#add_executable(thread_pool ${SDNN_UTEST_DIR}/thread_pool.cpp)
#target_include_directories(thread_pool PRIVATE ${SDNN_3RD_PARTY_DIR}/doctest)
#target_link_libraries(thread_pool Threads::Threads)
#add_test(ThreadPool_sum ${SDNN_UTEST_DIR}/thread_pool -tc=sum)
#add_test(ThreadPool_create ${SDNN_UTEST_DIR}/thread_pool -tc=create_pool)
#add_test(ThreadPool_enqueue_type ${SDNN_UTEST_DIR}/thread_pool -tc=enque_type)
#add_test(ThreadPool_enqueue_large_size ${SDNN_UTEST_DIR}/thread_pool -tc=enque_large_size)

#endif()


#find -arch
include(FindCUDA)
set(CUDA_ARCH_LIST Auto CACHE STRING
    "List of CUDA architectures (e.g. Pascal, Volta, etc) or \
compute capability versions (6.1, 7.0, etc) to generate code for. \
Set to Auto for automatic detection (default)."
)
cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_ARCH_LIST})
list(APPEND CUDA_NVCC_FLAGS ${CUDA_ARCH_FLAGS})

# add executables
message(STATUS "building executables ...")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

cuda_add_executable(main_cuda ${PROJECT_SOURCE_DIR}/main/main.cu)
target_link_libraries(main_cuda ${PROJECT_NAME} stdc++fs OpenMP::OpenMP_CXX)


#cuda_add_executable(tsv_file_to_binary ${PROJECT_SOURCE_DIR}/main/tsv_file_to_binary.cu)
#target_link_libraries(tsv_file_to_binary ${PROJECT_NAME} stdc++fs)

#CPU parallel. Not support yet.
#cuda_add_executable(diagonal_to_binary ${PROJECT_SOURCE_DIR}/main/diagonal_to_binary.cu)
#target_link_libraries(diagonal_to_binary ${PROJECT_NAME} stdc++fs snig::default_settings)
#target_link_libraries(main ${PROJECT_NAME} Threads::Threads stdc++fs snig::default_settings)

