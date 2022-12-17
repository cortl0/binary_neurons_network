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

include_directories(../)

set(HEADERS
#    ${HEADERS}
#    ../bnn/bnn.h
#    ${CMAKE_SOURCE_DIR}/../bnn/m_sequence.h
#    ${CMAKE_SOURCE_DIR}/../common/headers/state.hpp
#    ${CMAKE_SOURCE_DIR}/../bnn/neurons/neuron.h
#    ${CMAKE_SOURCE_DIR}/../bnn/neurons/binary.h
#    ${CMAKE_SOURCE_DIR}/../bnn/neurons/motor.h
#    ${CMAKE_SOURCE_DIR}/../bnn/neurons/sensor.h
#    ${CMAKE_SOURCE_DIR}/../bnn/neurons/storage.h
#    ${CMAKE_SOURCE_DIR}/../bnn/random.h
#    ${CMAKE_SOURCE_DIR}/../cpu/thread.h
)

set(SOURCES
#    ${CMAKE_SOURCE_DIR}/bnn.cpp
#    ${CMAKE_SOURCE_DIR}/bnn/neurons/neuron.cpp
#    ${CMAKE_SOURCE_DIR}/bnn/neurons/binary.cpp
#    ${CMAKE_SOURCE_DIR}/bnn/neurons/motor.cpp
#    ${CMAKE_SOURCE_DIR}/bnn/neurons/sensor.cpp
#    ${CMAKE_SOURCE_DIR}/bnn/random.cpp
#    ${CMAKE_SOURCE_DIR}/
    brain.cpp
)

include_directories(${CMAKE_SOURCE_DIR}/bnn)

add_library(${PROJECT_NAME} STATIC ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME} -pthread)
