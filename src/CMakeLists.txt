#
#   Binary Neurons Network
#   created by Ilya Shishkin
#   cortl@8iter.ru
#   https://github.com/cortl0/binary_neurons_network
#   licensed by GPL v3.0
#

cmake_minimum_required(VERSION 2.8)

add_subdirectory(cpu)
add_subdirectory(gpu)
add_subdirectory(examples/minimal)

#execute_process(COMMAND make --makefile=${PROJECT_SOURCE_DIR}/gpu/cuda/Makefile
#    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/gpu/cuda
#    )
