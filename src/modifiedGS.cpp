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
    norm = complex_dot(m, V+j*m, V+j*m); 
    norm = sqrt(norm);
    vec_update(m, std::complex<double>(1.0,0.0)/norm, V+j*m, V+j*m);
    for (k=j+1; k<n; k++)
    {
       vdot = complex_dot(m, V+j*m, V+k*m);
       complex_axpby(m, -vdot, V+j*m, std::complex<double>(1.0,0.0), V+k*m);
    }
  }
}


