#
#   Binary Neurons Network
#   created by Ilya Shishkin
#   cortl@8iter.ru
#   https://github.com/cortl0/binary_neurons_network
#   licensed by GPL v3.0
#

cmake_minimum_required(VERSION 2.8)

project(bnn)

set(SOURCE
	brain.cpp
	m_sequence.cpp
	neurons/neuron.cpp
	neurons/binary.cpp
	neurons/motor.cpp
	neurons/sensor.cpp
	random/random.cpp
	storage.cpp
	thread.cpp)

add_library(${CMAKE_PROJECT_NAME} SHARED ${SOURCE})

target_link_libraries(${CMAKE_PROJECT_NAME} pthread)
