cmake_minimum_required(VERSION 3.16.0)
project(QRes)

include_directories(include)
add_executable(QRes_3b1d src/QRes_3b1d.cpp src/timer.cpp include/header.hpp include/tensor3b1d.hpp include/timer.hpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fopenmp")

find_package(MPI REQUIRED)
#include_directories(SYSTEM ${MPI_INCLUDE_PATH})
target_link_libraries(QRes_3b1d PUBLIC MPI::MPI_CXX)


