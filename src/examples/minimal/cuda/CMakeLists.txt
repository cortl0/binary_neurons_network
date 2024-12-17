#
#   Binary Neurons Network
#   created by Ilya Shishkin
#   cortl@8iter.ru
#   https://github.com/cortl0/binary_neurons_network
#   licensed by GPL v3.0
#

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(minimal-cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_definitions(BNN_ARCHITECTURE_CUDA)

add_executable(${PROJECT_NAME} ../main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE pthread bnn-cuda)
