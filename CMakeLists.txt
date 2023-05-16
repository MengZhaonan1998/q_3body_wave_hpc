cmake_minimum_required(VERSION 3.16.0)
project(QRes)

include_directories(include)

add_executable(QRes_3b1d src/QRes_3b1d.cpp src/timer.cpp src/buildMatrix.cpp src/JacobiDavidson.cpp src/modifiedGS.cpp src/operations.cpp src/gmres_solver.cpp)
add_executable(run_tests src/run_tests.cpp src/test_operations.cpp src/operations.cpp src/gtest_mpi.cpp src/buildMatrix.cpp src/modifiedGS.cpp src/gmres_solver.cpp)
add_executable(lapack_test src/testlapack.c)

target_include_directories(QRes_3b1d PUBLIC include/)
target_include_directories(run_tests PUBLIC include/)
target_include_directories(lapack_test PUBLIC include/)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fopenmp -lm")

find_package(MPI REQUIRED)
#include_directories(SYSTEM ${MPI_INCLUDE_PATH})
target_link_libraries(run_tests PUBLIC MPI::MPI_CXX)
target_link_libraries(QRes_3b1d PUBLIC MPI::MPI_CXX)
target_link_libraries(lapack_test ${CMAKE_SOURCE_DIR}/lapacklib/liblapacke.a ${CMAKE_SOURCE_DIR}/lapacklib/liblapack.a ${CMAKE_SOURCE_DIR}/lapacklib/libcblas.a ${CMAKE_SOURCE_DIR}/lapacklib/librefblas.a gfortran m)

