#include <iostream>

#include "header.hpp"

#define N 5 

int main()
{

  double *Dn = new double[(N+1)*(N+1)];

  ChebyshevDiffMatrix2(N, 1, Dn);

  SparseMatrix *pot;
  pot = buildGaussianPotential2B1D(N, 1, 1);

  MatrixShow(N+1,N+1,(N+1)*(N+1),Dn);
 
 
  delete [] Dn;

  return 0;
}
