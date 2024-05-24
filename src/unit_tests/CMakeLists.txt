#
#   Binary Neurons Network
#   created by Ilya Shishkin
#   cortl@8iter.ru
#   https://github.com/cortl0/binary_neurons_network
#   licensed by GPL v3.0
#

cmake_minimum_required(VERSION 3.17)

project(bnn_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)
enable_testing()

set(SOURCES
    ../bnn/${PROJECT_NAME}.cpp
    ../common/logger.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})
add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
