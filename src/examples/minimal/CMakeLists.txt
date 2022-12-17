#
#   Binary Neurons Network
#   created by Ilya Shishkin
#   cortl@8iter.ru
#   https://github.com/cortl0/binary_neurons_network
#   licensed by GPL v3.0
#

cmake_minimum_required(VERSION 2.8)

project(minimal LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(../..)
include_directories(${CMAKE_SOURCE_DIR})
#include_directories(${CMAKE_SOURCE_DIR}/cpu)

add_executable(${PROJECT_NAME} main.cpp ../../cpu/brain.cpp)

target_link_libraries(${PROJECT_NAME} -pthread)
#${CMAKE_SOURCE_DIR}/../../build/cpu/libbnn-cpu.a
