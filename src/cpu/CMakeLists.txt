#
#   Binary Neurons Network
#   created by Ilya Shishkin
#   cortl@8iter.ru
#   https://github.com/cortl0/binary_neurons_network
#   licensed by GPL v3.0
#

cmake_minimum_required(VERSION 2.8)

project(bnn_cpu LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(HEADERS
    ${HEADERS}
    ${CMAKE_SOURCE_DIR}/brain.h
    ${CMAKE_SOURCE_DIR}/../common/headers/config.hpp
    ${CMAKE_SOURCE_DIR}/../common/headers/m_sequence.h
    ${CMAKE_SOURCE_DIR}/../common/headers/simple_math.hpp
    ${CMAKE_SOURCE_DIR}/../common/headers/state.hpp
    ${CMAKE_SOURCE_DIR}/neurons/neuron.h
    ${CMAKE_SOURCE_DIR}/neurons/binary.h
    ${CMAKE_SOURCE_DIR}/neurons/motor.h
    ${CMAKE_SOURCE_DIR}/neurons/sensor.h
    ${CMAKE_SOURCE_DIR}/neurons/storage.hpp
    ${CMAKE_SOURCE_DIR}/random.h
    ${CMAKE_SOURCE_DIR}/thread.h
)

set(SOURCES
    ${CMAKE_SOURCE_DIR}/brain.cpp
    ${CMAKE_SOURCE_DIR}/../common/sources/m_sequence.cpp
    ${CMAKE_SOURCE_DIR}/neurons/neuron.cpp
    ${CMAKE_SOURCE_DIR}/neurons/binary.cpp
    ${CMAKE_SOURCE_DIR}/neurons/motor.cpp
    ${CMAKE_SOURCE_DIR}/neurons/sensor.cpp
    ${CMAKE_SOURCE_DIR}/random.cpp
    ${CMAKE_SOURCE_DIR}/thread.cpp
)

add_library(${CMAKE_PROJECT_NAME} STATIC ${SOURCES} ${HEADERS})

target_link_libraries(${CMAKE_PROJECT_NAME}  -pthread)
