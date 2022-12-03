#
#   Binary Neurons Network
#   created by Ilya Shishkin
#   cortl@8iter.ru
#   https://github.com/cortl0/binary_neurons_network
#   licensed by GPL v3.0
#

cmake_minimum_required(VERSION 2.8)
#find_package(CUDA 9.0 REQUIRED)
project(bnn_cuda LANGUAGES CUDA CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(external)

set(HEADERS
    ${CMAKE_SOURCE_DIR}/settings.hpp
    ${CMAKE_SOURCE_DIR}/statistic.hpp
)

set(SOURCES
    ${CMAKE_SOURCE_DIR}/main.cpp
)

#set(LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/usr/local/cuda/lib64})
add_executable(bnn_cuda ${SOURCES} ${HEADERS})
target_link_libraries(bnn_cuda ${CUDA_LIBRARIES})
#target_link_libraries(${CMAKE_PROJECT_NAME} /usr/local/cuda/lib64 -pthread)







#execute_process(COMMAND ${CMAKE_SOURCE_DIR}/n.sh
#OUTPUT_FILE file.txt)
#set(CMAKE_INCLUDE_CURRENT_DIR ON)
#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)

#${CMAKE_SOURCE_DIR}/gpu/cuda/
#include_directories(external)

#add_library(${PROJECT_NAME} STATIC ${SOURCES} ${HEADERS})

#target_link_libraries(${PROJECT_NAME} -pthread)
