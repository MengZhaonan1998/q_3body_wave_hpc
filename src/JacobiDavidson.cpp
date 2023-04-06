#include "header.hpp"

std::unique_ptr<resultJD> JacobiDavidson(int proc_numb, int proc_rank, std::map<std::string, std::string> jdopts)
{
  int numeigs = stoi(jdopts["numeigs"]);        // number of eigenvalues desired
  int mindim = stoi(jdopts["mindim"]);          // minimum dimension of search space V
  int maxdim = stoi(jdopts["maxdim"]);          // maximum dimension of search space V
  int maxiter = stoi(jdopts["maxiter"]);        // maximum number of iterations 
  double tolerance = stod(jdopts["tolerance"]); // tolerance of residual norm
  std::complex<double> sigma(stod(jdopts["target_real"]),stod(jdopts["target_imag"])); // target (shift) (complex)
 
  //--- Todo ---//
  std::unique_ptr<resultJD> result_ptr(new resultJD);
  result_ptr->eigval = (double*)malloc( numeigs * sizeof(double));
 
  for (int i=0; i< 5; i++)  result_ptr->eigval[i] = 1.0;


  // result_ptr->eigvec = (double*)malloc(5 * (nR+1) * (nr+1) * sizeof(double));
 


  return result_ptr;
}
