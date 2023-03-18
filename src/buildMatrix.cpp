#define _USE_MATH_DEFINES

#include <cmath>

#include "header.hpp"


void ChebyshevDiffMatrix(int n, double *ChebD1){
  double *xCheb = new double[n+1]; // chebyshev-gauss-lobatto grid points (Spectral methods in MATLAB)
  int i,j;
  for (i=0; i<=n; i++)
    xCheb[i] = cos(M_PI * i / n); // the length of grid is n+1	

  // Dn[0,0]
  ChebD1[0] = (2.0*n*n+1)/6.0; 
  // Dn[0,j<n]
  for (j=1; j<n; j++) ChebD1[j] = 2.0*pow(-1.0,j)/(1.0-xCheb[j]); 
  // Dn[0,n]
  ChebD1[n] = pow(-1.0,n)/2.0;

  // Dn[i,j] 0<i<n 0<=j<=n
  for (i=1; i<n; i++)
  {
    ChebD1[i*(n+1)] = -pow(-1.0,i)/2.0/(1.0-xCheb[i]);
    for (j=1; j<n; j++)
    {
      if (i==j) 
        ChebD1[i*(n+1)+j] = -xCheb[j]/2.0/(1.0-xCheb[j]*xCheb[j]); 
      else
        ChebD1[i*(n+1)+j] = pow(-1.0,i+j)/(xCheb[i]-xCheb[j]); 

    }
    ChebD1[i*(n+1)+n] = pow(-1.0,n+i)/2.0/(1.0+xCheb[i]);
  }
 
  // Dn[n,0]
  ChebD1[n*(n+1)] = -pow(-1,n)/2.0;
  // Dn[n,j<n]
  for (j=1; j<n; j++) ChebD1[n*(n+1)+j] = -2.0*pow(-1.0,n+j)/(1.0+xCheb[j]); 
  // Dn[n,n]
  ChebD1[(n+1)*(n+1)-1] = -(2.0*n*n+1.0)/6.0;

  delete [] xCheb;
  return;
}


void ChebyshevDiffMatrix2(int n, double *ChebD2){
  double *ChebD1 = new double[(n+1)*(n+1)];
  ChebyshevDiffMatrix(n, ChebD1);
  
  int i,j,k;
  for (i=0; i<=n; i++)
    for (j=0; j<=n; j++)
      for (k=0; k<=n; k++)   
        ChebD2[i*(n+1)+j] += ChebD1[i*(n+1)+k] * ChebD1[k*(n+1)+j];
  
  delete [] ChebD1;
  return;
}




/*
void buildDifferentialMatrixChebD1(int n, double *ChebD1){
	//Interior grid, compare Boyd Appendix F.9
	double *x_Cheb;
	int i,j;
	x_Cheb=(double *)malloc(n*sizeof(double));
	for(i=0;i<n;i++){
		x_Cheb[i]=(1*(2*i+1)/(2*n));
	}
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			if(i==j){
				ChebD1[i*n+j]=0.5*x_Cheb[i]/(1-(x_Cheb[i],2));
			}
			else{
				ChebD1[i*n+j]=(M_PI*(i+j))*((1-(x_Cheb[j],2))/(1-(x_Cheb[i],2)))/(x_Cheb[i]-x_Cheb[j]);
			}
		}
	}
	free(x_Cheb);
	return;
}
*/
