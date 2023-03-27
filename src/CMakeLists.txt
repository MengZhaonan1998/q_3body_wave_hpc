cmake_minimum_required(VERSION 3.16.0)
project(QRes)
add_executable(QRes_3b1d QRes_3b1d.cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fopenmp")

find_package(MPI REQUIRED)
#include_directories(SYSTEM ${MPI_INCLUDE_PATH})
target_link_libraries(QRes_3b1d PUBLIC MPI::MPI_CXX)
