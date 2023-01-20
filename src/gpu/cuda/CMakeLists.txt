#
#   Binary Neurons Network
#   created by Ilya Shishkin
#   cortl@8iter.ru
#   https://github.com/cortl0/binary_neurons_network
#   licensed by GPL v3.0
#

cmake_minimum_required(VERSION 2.8)
find_package(CUDA 9.0 REQUIRED)
project(bnn_cuda LANGUAGES CUDA CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    ../../bnn
    external
    )
include_directories(../../ ../../bnn)

set(HEADERS
    ${CMAKE_SOURCE_DIR}/settings.hpp
    ${CMAKE_SOURCE_DIR}/statistic.hpp
)

set(SOURCES
    ${CMAKE_SOURCE_DIR}/main.cpp
)

add_library(bnn_cuda STATIC main.cpp ${SOURCES} ${HEADERS})
target_link_libraries(bnn_cuda ${CUDA_LIBRARIES})

