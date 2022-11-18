#
#   Binary Neurons Network
#   created by Ilya Shishkin
#   cortl@8iter.ru
#   https://github.com/cortl0/binary_neurons_network
#   licensed by GPL v3.0
#

cmake_minimum_required(VERSION 2.8)

project(bnn_cuda LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#set(LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/usr/local/cuda/lib64})
#target_link_libraries(${CMAKE_PROJECT_NAME} ${LD_LIBRARY_PATH})

set(HEADERS
    ${HEADERS}
    ${CMAKE_SOURCE_DIR}/settings.hpp
    ${CMAKE_SOURCE_DIR}/statistic.hpp
)

set(SOURCES
    ${CMAKE_SOURCE_DIR}/bnn_cuda_test.cpp
)

include_directories(./external)

add_library(${CMAKE_PROJECT_NAME} STATIC ${SOURCES} ${HEADERS})

target_link_libraries(${CMAKE_PROJECT_NAME} -pthread)
