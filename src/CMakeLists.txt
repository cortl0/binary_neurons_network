#
#   Binary Neurons Network
#   created by Ilya Shishkin
#   cortl@8iter.ru
#   https://github.com/cortl0/binary_neurons_network
#   licensed by GPL v3.0
#

cmake_minimum_required(VERSION 3.17)

include_directories(.)

add_subdirectory(cpu)
add_subdirectory(gpu/cuda)
add_subdirectory(examples/minimal/cpu)
add_subdirectory(examples/minimal/cuda)
add_subdirectory(unit_tests)
