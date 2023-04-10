#include "header.hpp"
#include "operations.hpp"
/*
 *
 * */
void modifiedGS(std::complex<double>* V, int m, int n)
{
  int j,k;
  std::complex<double> norm;
  std::complex<double> vdot;
  for (j=0; j<n; j++)
  {
    norm = complex_dot(m, j*m, j*m, V, V); 
    norm = sqrt(norm);
    complex_0xpby(m, j*m, std::complex<double>(1.0,0.0)/norm, V);
    for (k=j+1; k<n; k++)
    {
       vdot = complex_dot(m, j*m, k*m, V, V);
       complex_axpby(m, j*m, k*m, -vdot, V, std::complex<double>(1.0,0.0), V);
    }
  }
}


