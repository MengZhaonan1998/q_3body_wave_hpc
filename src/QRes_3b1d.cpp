#include <iostream>
#include <cstdlib>
#include <mkl.h>
#include "mpi.h"

#include "header.hpp"
#include "timer.hpp"
#include "tensor3b1d.hpp"

int main(int argc, char* argv[])
{
  if (argc < 6){
    fprintf(stderr, "Usage %s <nx> <ny> <number of eigenvalues> <max iterations> <potential type>",argv[0]);
    return 1;
  }

  std::cout << "Welcome to QRes_3b1d!\n" 
	    << "This program aims at computing resonances of a quantum three body problem in parallel.\n" << std::endl;
 
  int nx = atoi(argv[1]);  // x grid size
  int ny = atoi(argv[2]);  // y grid size
  if (nx <= 2 || ny <= 2) throw std::runtime_error("need at least two grid points in each dimension to implement boundary conditions");

  int numeigs = atoi(argv[3]); // number of eigenvalues desired
  int maxiter = atoi(argv[4]); // maximum number of iterations
  char pot = argv[5][0];       // potential type

  int proc_rank; // processor rank
  int proc_numb; // number of processors

  MPI_Init(&argc, &argv); // initialize mpi

  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank); // get processor rank
  MPI_Comm_size(MPI_COMM_WORLD, &proc_numb); // get processor number

  resultJD *results; // use to store the results of eigenvalues, eigenvectors and convergence history
  results = (resultJD*)malloc(sizeof(resultJD));
  results->eigval = (double*)malloc(numeigs * sizeof(double));
  results->eigvec = (double*)malloc(numeigs * nx * ny * sizeof(double));
  results->cvg_hist = (double*)malloc(maxiter * sizeof(double));


  double* D1;  // Chebyshev diff mat x
  double* D2;  // Chebyshev diff mat y
  double* V;   // Potential matrix/or vector

  QRes::Kron2D<double> Koperator(1, D1, 1, D2, V, 1.0, 1.0);
  Koperator.apply();

  std::cout << "finish!\n" << std::endl;
 
  free(results->eigval);
  free(results->eigvec);
  free(results->cvg_hist);
  free(results);

  MPI_Finalize();
  return 0;
}
