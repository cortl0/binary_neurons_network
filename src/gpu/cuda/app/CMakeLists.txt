#
#   Binary Neurons Network
#   created by Ilya Shishkin
#   cortl@8iter.ru
#   https://github.com/cortl0/binary_neurons_network
#   licensed by GPL v3.0
#

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(bnn-cuda-app LANGUAGES CXX CUDA)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    .
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
    ../..)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib .)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib)

add_executable(${PROJECT_NAME} main.cpp)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${PROJECT_NAME} PRIVATE bnn-cuda)
