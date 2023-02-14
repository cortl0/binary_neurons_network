#
#   Binary Neurons Network
#   created by Ilya Shishkin
#   cortl@8iter.ru
#   https://github.com/cortl0/binary_neurons_network
#   licensed by GPL v3.0
#

cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(..)
include(CTest)
enable_testing()

set(SOURCES
    bnn_memory_test.cpp
    ../common/logger.cpp
    )

add_executable(bnn_memory_test ${SOURCES})
add_test(NAME bnn_memory_test COMMAND bnn_memory_test)
