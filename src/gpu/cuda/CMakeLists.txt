#
#   Binary Neurons Network
#   created by Ilya Shishkin
#   cortl@8iter.ru
#   https://github.com/cortl0/binary_neurons_network
#   licensed by GPL v3.0
#

cmake_minimum_required(VERSION 3.17)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(bnn-cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES
    cuda.cu
    ../../common/logger.cpp
    )

add_library(${PROJECT_NAME} STATIC ${SOURCES})
