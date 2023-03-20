#define _USE_MATH_DEFINES

#include <cmath>

#include "header.hpp"


void ChebyshevDiffMatrix(int n, double L, double *ChebD1){
  /* Chebyshev differentiation matrix D_N
   *  n -- number of grid points (The true number of grid points is n+1 due to the nature of Chebyshev-Gauss-Labatto points)
   *  L -- domain [-L,L] 
   *  ChebD1 -- Chebyshev diff matrix
   *  */
  double *xCheb = new double[n+1]; // chebyshev-gauss-lobatto grid points (Spectral methods in MATLAB)
  int i,j;
  for (i=0; i<=n; i++)
    xCheb[i] = cos(M_PI * i / n);  // the length of grid is n+1	

  // Dn[0,0]
  ChebD1[0] = (2.0*n*n+1)/6.0/L; 
  // Dn[0,j<n]
  for (j=1; j<n; j++) ChebD1[j] = 2.0*pow(-1.0,j)/(1.0-xCheb[j])/L; 
  // Dn[0,n]
  ChebD1[n] = pow(-1.0,n)/2.0/L;

  // Dn[i,j] 0<i<n 0<=j<=n
  for (i=1; i<n; i++)
  {
    ChebD1[i*(n+1)] = -pow(-1.0,i)/2.0/(1.0-xCheb[i])/L;
    for (j=1; j<n; j++)
    {
      if (i==j) 
        ChebD1[i*(n+1)+j] = -xCheb[j]/2.0/(1.0-xCheb[j]*xCheb[j])/L; 
      else
        ChebD1[i*(n+1)+j] = pow(-1.0,i+j)/(xCheb[i]-xCheb[j])/L; 

    }
    ChebD1[i*(n+1)+n] = pow(-1.0,n+i)/2.0/(1.0+xCheb[i])/L;
  }
 
  // Dn[n,0]
  ChebD1[n*(n+1)] = -pow(-1,n)/2.0/L;
  // Dn[n,j<n]
  for (j=1; j<n; j++) ChebD1[n*(n+1)+j] = -2.0*pow(-1.0,n+j)/(1.0+xCheb[j])/L; 
  // Dn[n,n]
  ChebD1[(n+1)*(n+1)-1] = -(2.0*n*n+1.0)/6.0/L;

  delete [] xCheb;
  return;
}


void ChebyshevDiffMatrix2(int n, double L, double *ChebD2){
  /*  Second-order Chebyshev differentation matrix
   *  n -- number of grid points (The true number of grid points is n+1 due to the nature of Chebyshev-Gauss-Labatto points)
   *  L -- domain [-L,L] 
   *  ChebD1 -- 2nd-order Chebyshev diff matrix
   *  */
  double *ChebD1 = new double[(n+1)*(n+1)];
  ChebyshevDiffMatrix(n, L, ChebD1);
  
  int i,j,k;
  for (i=0; i<=n; i++)
    for (j=0; j<=n; j++)
      for (k=0; k<=n; k++)   
        ChebD2[i*(n+1)+j] += ChebD1[i*(n+1)+k] * ChebD1[k*(n+1)+j];
  
  delete [] ChebD1;
  return;
}


SparseMatrix *buildGaussianPotential2B1D(int n, double L, double v0){
  /*  Sparse matrix of Gaussian potential for 2-body 1-dimensional problem
   *  n -- number of grid points (The true number of grid points is n+1 due to the nature of Chebyshev-Gauss-Labatto points)
   *  L -- domain [-L,L] 
   *  v0 -- potentail parameter
   * */
  double *xCheb = new double[n+1];
  double val;
  int i;
  for (i=0; i<=n; i++)
    xCheb[i] = L * cos(M_PI * i / n); // the length of grid is n+1	

  SparseMatrix* VGauss2B1D = new SparseMatrix;
  VGauss2B1D->dim = n+1;
  VGauss2B1D->nz = 1;

  VGauss2B1D->columnInd = new int[n+1];
  VGauss2B1D->value = new double[n+1];
  
  for (i=0; i<=n; i++)  
  {
    VGauss2B1D->columnInd[i] = i;
    val = pow(xCheb[i], 2);
    VGauss2B1D->value[i] = -exp(-val) * v0;
  }

  delete [] xCheb;
  return VGauss2B1D;
}


/*
double *buildDenseHamiltonian2B1D(int n, double L, SparseMatrix *Vsp){
  if ((n+1) != Vsp->dim){
    printf("dimension of Vsp matrix does not match with the grid.\n");
    return 0; //...
  }
	
  double *Hamilton = new double[(n+1)*(n+1)];
   
  ChebyshevDiffMatrix2(n, Hamilton);
  
  

  return Hamilton;
}

*/





