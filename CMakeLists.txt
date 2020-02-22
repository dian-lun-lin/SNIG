# CMake version
cmake_minimum_required (VERSION 3.9)
MESSAGE(STATUS "CMAKE_ROOT: " ${CMAKE_ROOT})

# Project name
project(SparseDNN VERSION 1.0.0 LANGUAGES CXX)

# Turn on the verbose
set(CMAKE_VERBOSE_MAKEFILE ON)


#not test yet
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "7.3")
	  message(FATAL_ERROR "\n SparseDNN requires g++ at least v7.3")
  endif()
## clang++
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "6.0")
	  message(FATAL_ERROR "\n SparseDNN requires clang++ at least v6.0")
  endif() 
endif()

# Args
option(SDNN_BUILD_TESTS "Enables build of tests" ON)
option(SDNN_ENABLE_GPU "turn on GPU" OFF)

# installation path
set(SDNN_UTEST_DIR ${PROJECT_SOURCE_DIR}/unittests)
set(SDNN_3RD_PARTY_DIR ${PROJECT_SOURCE_DIR}/3rd-party)

# include test
include(CTest)

#compiler setting 
add_library(error_settings INTERFACE)

target_compile_options(
  error_settings
  INTERFACE
  $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wfatal-errors>
  $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wfatal-errors>
)

#optmization
add_library(optimize_settings INTERFACE)
target_compile_options(
  optimize_settings INTERFACE
  $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:Clang>>:-O2 -march=native>
  $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU>>:-O2 -march=native>
  $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU>>:-O0 -g>
  $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:Clang>>:-O0 -g>
)

# CXX target properties
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Thread
find_package(Threads REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)

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

#--------interface------
add_library(${PROJECT_NAME} INTERFACE)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)
target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include/> 
)
#-----------------------

# test
if(${SDNN_BUILD_TESTS})
 
enable_testing()
message(STATUS "Building unit tests ...")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${SDNN_UTEST_DIR})

add_executable(reader ${SDNN_UTEST_DIR}/reader.cpp)
target_link_libraries(reader stdc++fs)
target_include_directories(reader PRIVATE ${SDNN_3RD_PARTY_DIR}/doctest)
add_test(tsv_string_to_matrix ${SDNN_UTEST_DIR}/reader -tc=tsv_string_to_matrix)

add_executable(thread_pool ${SDNN_UTEST_DIR}/thread_pool.cpp)
target_include_directories(thread_pool PRIVATE ${SDNN_3RD_PARTY_DIR}/doctest)
target_link_libraries(thread_pool Threads::Threads)
add_test(ThreadPool_sum ${SDNN_UTEST_DIR}/thread_pool -tc=sum)
add_test(ThreadPool_create ${SDNN_UTEST_DIR}/thread_pool -tc=create_pool)
add_test(ThreadPool_enqueue_type ${SDNN_UTEST_DIR}/thread_pool -tc=enque_type)
add_test(ThreadPool_enqueue_large_size ${SDNN_UTEST_DIR}/thread_pool -tc=enque_large_size)

endif()

# add executables
message(STATUS "building executables ...")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

if(SDNN_ENABLE_GPU)
  find_package(CUDA QUIET REQUIRED)
  cuda_add_executable(main_cuda ${PROJECT_SOURCE_DIR}/main/main.cu)
  target_link_libraries(main_cuda stdc++fs)
endif()

add_executable(main ${PROJECT_SOURCE_DIR}/main/main.cpp)
target_link_libraries(main ${PROJECT_NAME} Threads::Threads stdc++fs)
