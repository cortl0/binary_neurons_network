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

set(BNN_TEST_BIN_NAME bnn_test)

set(SOURCES
    ../bnn/${BNN_TEST_BIN_NAME}.cpp
    ../common/logger.cpp
    )

add_executable(${BNN_TEST_BIN_NAME} ${SOURCES})
add_test(NAME ${BNN_TEST_BIN_NAME} COMMAND ${BNN_TEST_BIN_NAME})
