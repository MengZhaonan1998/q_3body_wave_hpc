#include "operations.hpp"
#include "tensor3b1d.hpp"
#include "header.hpp"

void init(int n, double* x, double value)
{
  
  #pragma omp parallel for 
  for (int i=0; i<n; i++)
     x[i] = value;   // assign value to every entry of x
  
  return;
}


void init(int n, std::complex<double>* x, std::complex<double> value)
{
  #pragma omp parallel for 
  for (int i=0; i<n; i++)
     x[i] = value;   // assign value to every entry of x
  
  return;
}


std::complex<double> dot(int n, std::complex<double> const* x, std::complex<double> const* y)
{
  std::complex<double> local_res=0.0;
  std::complex<double> global_res;   
  /* It's already 2023, and OpenMP still doesn't support complex number operations?! 
   * No... We can define the reduction using 'declare' */
  #pragma omp declare reduction(+: std::complex<double>: omp_out += omp_in) initializer (omp_priv = omp_orig) 
  #pragma omp parallel for reduction(+:local_res) 
  for (int i=0; i<n; i++)
     local_res += x[i]*y[i];
  
  MPI_Allreduce(&local_res, &global_res, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
  
  return global_res;  
}


std::complex<double> complex_dot(int n, std::complex<double> const* x, std::complex<double> const* y)
{
  std::complex<double> local_res(0.0,0.0);
  std::complex<double> global_res;
  #pragma omp declare reduction(+: std::complex<double>: omp_out += omp_in) initializer (omp_priv = omp_orig)
  #pragma omp parallel for reduction(+:local_res)
  for (int i=0; i<n; i++)
     local_res += std::conj(x[i]) * y[i];
  
  MPI_Allreduce(&local_res, &global_res, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);

  return global_res;
}


void axpby(int n, double a, double const* x, double b, double* y)
{
  #pragma omp parallel for
  for (int i=0; i<n; i++)
     y[i] = (a * x[i] + b * y[i]);
  return;
}


void axpby(int n, std::complex<double> a, std::complex<double> const* x, std::complex<double> b, std::complex<double>* y)
{
  #pragma omp parallel for
  for (int i=0; i<n; i++)
     y[i] = (a * x[i] + b * y[i]);
  return;
}


void vec_update(int n, std::complex<double> a, std::complex<double> const* x, std::complex<double>* y)
{
  #pragma omp parallel for
  for (int i=0; i<n; i++)
     y[i] = a * x[i];
  return;
}


int domain_decomp(int global)
{
  int rank, size, local;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (global % size != 0)
  {
     std::cout << "To balance the workload among processors, it is recommended to choose a processor number that is divisible by A." << std::endl;
     int td = std::round(double(global)/double(size));   
     if (rank != size-1)
        local = td;
     else
	local = global - td*rank;     
  }
  else
  {
     local = global / size;
  }

  return local;
}

