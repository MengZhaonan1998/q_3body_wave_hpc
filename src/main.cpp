#include <iostream>

#include "header.hpp"

#define N 5 

int main()
{

  double *Dn = new double[(N+1)*(N+1)];

  ChebyshevDiffMatrix2(N, Dn);

  MatrixShow(N+1,N+1,(N+1)*(N+1),Dn);
 
 
  delete [] Dn;

  return 0;
}
