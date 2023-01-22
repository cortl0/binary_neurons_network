#
#   Binary Neurons Network
#   created by Ilya Shishkin
#   cortl@8iter.ru
#   https://github.com/cortl0/binary_neurons_network
#   licensed by GPL v3.0
#

cmake_minimum_required(VERSION 2.8)

project(bnn-cpu LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BNN_ARCHITECTURE_CPU)
include_directories(../)

include_directories(${CMAKE_SOURCE_DIR}/bnn)

add_library(${PROJECT_NAME} STATIC cpu.cpp)
#add_library(${PROJECT_NAME} STATIC cuda.cpp)

target_link_libraries(${PROJECT_NAME} -pthread)
