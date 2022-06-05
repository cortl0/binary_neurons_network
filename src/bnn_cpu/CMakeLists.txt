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
    ${CMAKE_SOURCE_DIR}/headers/brain.h
    ${CMAKE_SOURCE_DIR}/../common/headers/config.hpp
    ${CMAKE_SOURCE_DIR}/../common/headers/m_sequence.h
    ${CMAKE_SOURCE_DIR}/../common/headers/simple_math.hpp
    ${CMAKE_SOURCE_DIR}/../common/headers/state.hpp
    ${CMAKE_SOURCE_DIR}/headers/neurons/neuron.h
    ${CMAKE_SOURCE_DIR}/headers/neurons/binary.h
    ${CMAKE_SOURCE_DIR}/headers/neurons/motor.h
    ${CMAKE_SOURCE_DIR}/headers/neurons/sensor.h
    ${CMAKE_SOURCE_DIR}/headers/neurons/storage.hpp
    ${CMAKE_SOURCE_DIR}/headers/random.h
    ${CMAKE_SOURCE_DIR}/headers/thread.h
)

set(SOURCES
    ${CMAKE_SOURCE_DIR}/sources/brain.cpp
    ${CMAKE_SOURCE_DIR}/../common/sources/m_sequence.cpp
    ${CMAKE_SOURCE_DIR}/sources/neurons/neuron.cpp
    ${CMAKE_SOURCE_DIR}/sources/neurons/binary.cpp
    ${CMAKE_SOURCE_DIR}/sources/neurons/motor.cpp
    ${CMAKE_SOURCE_DIR}/sources/neurons/sensor.cpp
    ${CMAKE_SOURCE_DIR}/sources/random.cpp
    ${CMAKE_SOURCE_DIR}/sources/thread.cpp
)

add_library(${CMAKE_PROJECT_NAME} STATIC ${SOURCES} ${HEADERS})

target_link_libraries(${CMAKE_PROJECT_NAME}  -pthread)
