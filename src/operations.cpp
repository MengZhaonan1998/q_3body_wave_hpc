#include "operations.hpp"
#include "header.hpp"

void init(int n, double* x, double value)
{
  
  #pragma omp parallel for 
  for (int i=0; i<n; i++)
     x[i] = value;   // assign value to every entry of x
  
  return;
}


void complex_init(int n, std::complex<double>* x, std::complex<double> value)
{
 
  #pragma omp parallel for 
  for (int i=0; i<n; i++)
     x[i] = value;   // assign value to every entry of x
  
  return;
}


std::complex<double> dot(int n, std::complex<double> const* x, std::complex<double> const* y)
{
  std::complex<double> res=0.0;
  
  /* It's already 2023, and OpenMP still doesn't support complex number operations?! 
   * No... We can define the reduction using 'declare' */
  #pragma omp declare reduction(+: std::complex<double>: omp_out += omp_in) initializer (omp_priv = omp_orig) 
  #pragma omp parallel for reduction(+:res) 
  for (int i=0; i<n; i++)
     res += x[i]*y[i];
  
  return res;  
}


std::complex<double> complex_dot(int n, std::complex<double> const* x, std::complex<double> const* y)
{
  std::complex<double> res(0.0,0.0);
  
  #pragma omp declare reduction(+: std::complex<double>: omp_out += omp_in) initializer (omp_priv = omp_orig)
  #pragma omp parallel for reduction(+:res)
  for (int i=0; i<n; i++)
     res += std::conj(x[i]) * y[i];

  return res;
}


void axpby(int n, double a, double const* x, double b, double* y)
{
  #pragma omp parallel for
  for (int i=0; i<n; i++)
     y[i] = (a * x[i] + b * y[i]);
  return;
}


void complex_axpby(int n, std::complex<double> a, std::complex<double>* x, std::complex<double> b, std::complex<double>* y)
{
  #pragma omp parallel for
  for (int i=0; i<n; i++)
     y[i] = (a * x[i] + b * y[i]);
  return;
}


void vec_update(int n, std::complex<double> a, std::complex<double>* x, std::complex<double>* y)
{
  #pragma omp parallel for
  for (int i=0; i<n; i++)
     y[i] = a * x[i];
  return;
}
