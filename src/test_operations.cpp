#include "header.hpp"
#include "gtest_mpi.hpp"
#include "operations.hpp"
#include "tensor3b1d.hpp"
#include <iostream>


TEST(stencil, bounds_check)
{
  stencil3d S;
  S.nx=5;
  S.ny=3;
  S.nz=2;
  EXPECT_THROW(S.index_c(-1,0,0), std::runtime_error);
  EXPECT_THROW(S.index_c(S.nx,0,0), std::runtime_error);
  EXPECT_THROW(S.index_c(0,-1,0), std::runtime_error);
  EXPECT_THROW(S.index_c(0,S.ny,0), std::runtime_error);
  EXPECT_THROW(S.index_c(0,0,-1), std::runtime_error);
  EXPECT_THROW(S.index_c(0,0,S.nz), std::runtime_error);
}


TEST(stencil, index_order_kji)
{
  stencil3d S;
  S.nx=50;
  S.ny=33;
  S.nz=21;

  int i=10, j=15, k=9;

  EXPECT_EQ(S.index_c(i,j,k), S.index_c(i-1,j,k)+1);
  EXPECT_EQ(S.index_c(i,j,k), S.index_c(i,j-1,k)+S.nx);
  EXPECT_EQ(S.index_c(i,j,k), S.index_c(i,j,k-1)+S.nx*S.ny);
}


TEST(operations, init)
{
  const int n=15;
  double x[n];
  for (int i=0; i<n; i++) x[i]=double(i+1);

  double val=42.0;
  init(n, x, val);

  double err=0.0;
  for (int i=0; i<n; i++) err = std::max(err, std::abs(x[i]-val));

  // note: EXPECT_NEAR uses a tolerance relative to the size of the target,
  // near 0 this is very small, so we use an absolute test instead by 
  // comparing to 1 instead of 0.
  EXPECT_NEAR(1.0+err, 1.0, std::numeric_limits<double>::epsilon());
}


TEST(operations, dot) {
  const int n=150;
  double x[n], y[n];

  for (int i=0; i<n; i++)
  {
    x[i] = double(i+1);
    y[i] = 1.0/double(i+1);
  }

  double res = dot(n, x, y); // The results of dot(x,y) should be equal to the length n
  EXPECT_NEAR(res, (double)n, n*std::numeric_limits<double>::epsilon());
}


TEST(operations, axpby)
{
  const int n=10;
  double x[n], y[n];
  
  double a=2.0; 
  double b=2.0;

  for (int i=0; i<n; i++)
  {
     x[i] = double(i+1)/2.0;
     y[i] = double(n-i-1)/2.0;
  }

  axpby(n, a, x, b, y); // The results of axpby should be the array in which every element is equal to the length n
  
  double err=0.0;
  for (int i=0; i<n; i++) err = std::max(err, std::abs(y[i]-double(n)));
  EXPECT_NEAR(1.0+err, 1.0, std::numeric_limits<double>::epsilon());
}


TEST(operations, tensor_apply)
{
  // test for tensor operator reshape(a2*C2*W+a1*W*C1^T,N1*N2,1)+V*w
  double C1[4]={1,2,1,3};            // C1
  double C2[9]={2,2,3,2,2,4,2,2,5};  // C2
  double V[6]={1,2,3,6,3,2};   // Potential matrix/or vector

  double v_in[6]={1,2,1,3,0,4};
  double v_out[6];
   
  QRes::Kron2D<double> Koperator(2, C1, 3, C2, V, 1.0, 1.0);
  Koperator.apply(v_in, v_out);
  
  double result[6] = {17, 16, 23, 46, 24, 47}; 

  double err=0.0;
  for (int i=0; i<6; i++) err = std::max(err, std::max(err, std::abs(result[i]-v_out[i])));
  EXPECT_NEAR(1.0+err, 1.0, std::numeric_limits<double>::epsilon());  
}


TEST(buildoperator, chebyshevDiffMat)
{
  int n =5;
  double L=1.0;
  
 // double ChebMat[(n+1)*(n+1)];
  double* ChebMat = (double*)malloc((n+1)*(n+1)*sizeof(double));
  ChebyshevDiffMatrix(n, L, ChebMat); // first order chebyshev differentiation matrix

  double true_chebmat[(n+1)*(n+1)] = 
   {8.5000,  -10.4721,    2.8944,   -1.5279,    1.1056,   -0.5000,
    2.6180,   -1.1708,   -2.0000,    0.8944,   -0.6180,    0.2764,
   -0.7236,    2.0000,   -0.1708,   -1.6180,    0.8944,   -0.3820,
    0.3820,   -0.8944,    1.6180,    0.1708,   -2.0000,    0.7236,
   -0.2764,    0.6180,   -0.8944,    2.0000,    1.1708,   -2.6180,
    0.5000,   -1.1056,    1.5279,   -2.8944,   10.4721,   -8.5000}; // true chebyshev diff matrix

  double err = 0.0;
  for (int i=0; i<(n+1)*(n+1); i++)
      err = std::max(err, std::max(err, std::abs(true_chebmat[i]-ChebMat[i])));

  EXPECT_NEAR(1.0+err, 1.0, 0.0001);  
  free(ChebMat);
}


TEST(buildoperator, chebyshevDiffMat2)
{
  int n =5;
  double L=1.0;
  
  // double ChebMat[(n+1)*(n+1)];
  double* ChebMat2 = (double*)malloc((n+1)*(n+1)*sizeof(double));
  ChebyshevDiffMatrix2(n, L, ChebMat2); // second order chebyshev differentiation matrix

  double true_chebmat2[(n+1)*(n+1)] = 
  {41.6000,  -68.3607,   40.8276,  -23.6393,   17.5724,   -8.0000,
   21.2859,  -31.5331,   12.6833,   -3.6944,    2.2111,   -0.9528,
   -1.8472,    7.3167,  -10.0669,    5.7889,   -1.9056,    0.7141,
    0.7141,   -1.9056,    5.7889,  -10.0669,    7.3167,   -1.8472,
   -0.9528,    2.2111,   -3.6944,   12.6833,  -31.5331,   21.2859,
   -8.0000,   17.5724,  -23.6393,   40.8276,  -68.3607,   41.6000};

  double err = 0.0;
  for (int i=0; i<(n+1)*(n+1); i++)
      err = std::max(err, std::max(err, std::abs(true_chebmat2[i]-ChebMat2[i])));

  EXPECT_NEAR(1.0+err, 1.0, 0.0001);  
  free(ChebMat2);
}

















