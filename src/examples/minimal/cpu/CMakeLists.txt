#
#   Binary Neurons Network
#   created by Ilya Shishkin
#   cortl@8iter.ru
#   https://github.com/cortl0/binary_neurons_network
#   licensed by GPL v3.0
#

cmake_minimum_required(VERSION 2.8)

project(minimal-cpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_definitions(BNN_ARCHITECTURE_CPU)

include_directories(../../..)

add_executable(${PROJECT_NAME} ../main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE pthread bnn-cpu)
