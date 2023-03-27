#include <iostream>
#include "mpi.h"

int main(int argc, char* argv[])
{
  if (argc < 5){
    fprintf(stderr, "Usage %s <nx> <ny> <number of eigenvalues> <potential type>",argv[0]);
    return 1;
  }

  std::cout << "Welcome to QRes_3b1d!i\n" 
	    << "This program aims at computing resonances of a quantum three body problem in parallel." << std::endl;
 
  int nx = atoi(argv[1]);  // x grid size
  int ny = atoi(argv[2]);  // y grid size
  if (nx <= 2 || ny <= 2) throw std::runtime_error("need at least two grid points in each dimension to implement boundary conditions");

  int numeigs = atoi(argv[3]); // number of eigenvalues desired
  char pot = argv[4][0];          // potential type

  int proc_rank; // processor rank
  int proc_numb; // number of processors

  MPI_Init(&argc, &argv); // initialize mpi

  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank); // get processor rank
  MPI_Comm_size(MPI_COMM_WORLD, &proc_numb); // get processor number

  

  MPI_Finalize();
  return 0;
}
