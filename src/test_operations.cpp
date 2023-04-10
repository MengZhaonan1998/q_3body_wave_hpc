#include "header.hpp"
#include "gtest_mpi.hpp"
#include "operations.hpp"
#include "tensor3b1d.hpp"
#include <iostream>


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


TEST(operations, dot) 
{
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


TEST(operations, complex_dot)
{
  const int n=50;
  std::complex<double> x[n], y[n];

  for (int i=0; i<n; i++)
  {
    x[i] = std::complex<double>(double(i+1), double(i+1));
    y[i] = std::complex<double>(1/double(i+1), 1/double(i+1));
  }
	
  std::complex<double> res = complex_dot(n/2, n/2, n/2, x, y);
  EXPECT_NEAR(res.real(), double(n), n*std::numeric_limits<double>::epsilon());
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


TEST(operations, complex_axpby)
{
  const int n=10;
  std::complex<double> x[n], y[n];

  std::complex<double> a=2.0;
  std::complex<double> b=2.0;
 
  for (int i=0; i<n; i++)
  {
    x[i] = std::complex<double>(double(i+1)/2.0,1.0);
    y[i] = std::complex<double>(double(n-i-1)/2.0,-1.0);
  }

  complex_axpby(n/2, n/2, n/2, a, x, b, y);

  double err=0.0;

  for (int i=n/2; i<n; i++)
  {  
    err = std::max(err, std::abs(y[i].real()-double(n)));
    err = std::max(err, std::abs(y[i].imag()-0.0));
  } 
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


TEST(functions, modified_gramschmidt)
{
  using namespace std::complex_literals;
  std::complex<double> V[16] = { 0.9827 + 0.3127i, 0.7302 + 0.1615i, 0.3439 + 0.1788i, 0.5841 + 0.4229i,
                                 0.1078 + 0.0942i, 0.9063 + 0.5985i, 0.8797 + 0.4709i, 0.8178 + 0.6959i,
                                 0.2607 + 0.6999i, 0.5944 + 0.6385i, 0.0225 + 0.0336i, 0.4253 + 0.0688i,
                                 0.4219 - 0.4712i, 0.9731 + 0.4214i, 0.1289 - 0.5839i, 0.6731 + 0.8571i};
  int m=4; int n=4;
  modifiedGS(V, m, n);

  double err_nr = 0.0;
  double err_ni = 0.0;
  std::complex<double> norm;
  for (int i=0; i<n; i++)
  {
     norm = complex_dot(m, m*i, m*i, V, V);
     err_nr = std::max(err_nr, norm.real());
     err_ni = std::max(err_ni, norm.imag());
  }
  EXPECT_NEAR(err_nr, 1.0, std::numeric_limits<double>::epsilon());
  EXPECT_NEAR(1.0+err_ni, 1.0, std::numeric_limits<double>::epsilon());

  double err_rr = 0.0;
  double err_ri = 0.0;
  for (int i=1; i<n; i++)
     for (int j=0; j<i; j++)
     {
        norm = complex_dot(m, m*j, m*i, V, V);
        err_rr = std::max(err_rr, norm.real());
        err_ri = std::max(err_ri, norm.imag());
     }

  EXPECT_NEAR(1.0+err_rr, 1.0, 10*std::numeric_limits<double>::epsilon());
  EXPECT_NEAR(1.0+err_ri, 1.0, 10*std::numeric_limits<double>::epsilon());
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

















