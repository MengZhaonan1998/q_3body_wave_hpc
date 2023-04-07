#include "mpi.h"
#include "header.hpp"
#include "tensor3b1d.hpp"

std::unique_ptr<resultJD> JacobiDavidson(int nR,int nr,double LR,double Lr,std::map<std::string, std::string> jdopts)
{
  //--- some settings ---//
  using namespace std::complex_literals; 
  int proc_rank;               // processor rank
  int proc_numb;               // number of processors
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank); // get processor rank
  MPI_Comm_size(MPI_COMM_WORLD, &proc_numb); // get processor number

  int numeigs = stoi(jdopts["numeigs"]);        // number of eigenvalues desired
  int mindim = stoi(jdopts["mindim"]);          // minimum dimension of search space V
  int maxdim = stoi(jdopts["maxdim"]);          // maximum dimension of search space V
  int maxiter = stoi(jdopts["maxiter"]);        // maximum number of iterations 
  double tolerance = stod(jdopts["tolerance"]); // tolerance of residual norm
  std::complex<double> sigma(stod(jdopts["target_real"]),stod(jdopts["target_imag"])); // target (shift) (complex)
 
  //--- memory allocation ---//
  // there are three things stored in the result: eigenvalue, eigenvector and convergence history
  std::unique_ptr<resultJD> result_ptr(new resultJD);
  result_ptr->eigval = (std::complex<double>*)malloc(sizeof(std::complex<double>)*numeigs);
  result_ptr->eigvec = (std::complex<double>*)malloc(sizeof(std::complex<double>)*numeigs*(nR+1)*(nr+1));
  result_ptr->cvg_hist = (double*)malloc(sizeof(double)*maxiter);

  // search space V (stored by column major)
 // std::complex<double>* V = (std::complex<double>*)malloc(sizeof(std::complex<double>)*(nR+1)*(nr+1)*maxdim);
  
  
  int iter = 0;     // iteration number
  int detected = 0; // number of eigenpairs found
  

  for (int i=0; i< numeigs; i++)  result_ptr->eigval[i] = 1i;

// proc_rank=1;
  // result_ptr->eigvec = (double*)malloc(5 * (nR+1) * (nr+1) * sizeof(double));
 

  return result_ptr;
}
