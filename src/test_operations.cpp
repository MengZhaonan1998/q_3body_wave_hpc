#include "header.hpp"
#include "gtest_mpi.hpp"
#include "operations.hpp"
#include "tensor3b1d.hpp"
#include "gmres_solver.hpp"
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
  int loc_n = domain_decomp(n);
  
  std::complex<double> x[loc_n], y[loc_n];

  for (int i=0; i<loc_n; i++)
  {
    x[i] = std::complex<double>(double(i+1),0.0);
    y[i] = 1.0/std::complex<double>(double(i+1),0.0);
  }

  std::complex<double> res = dot(loc_n, x, y); // The results of dot(x,y) should be equal to the length n
  
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank==0)
     EXPECT_NEAR(res.real(), (double)n, n*std::numeric_limits<double>::epsilon());
}


TEST(operations, complex_dot)
{
  const int n=50;
  int loc_n = domain_decomp(n);

  std::complex<double> x[loc_n], y[loc_n];

  for (int i=0; i<loc_n; i++)
  {
    x[i] = std::complex<double>(double(i+1), double(i+1));
    y[i] = std::complex<double>(1/double(i+1), 1/double(i+1));
  }
	
  std::complex<double> res = complex_dot(loc_n, x, y);
  
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank==0)
     EXPECT_NEAR(res.real(), 2*double(n), n*std::numeric_limits<double>::epsilon());
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

  axpby(n/2, a, x+n/2, b, y+n/2);

  double err=0.0;

  for (int i=n/2; i<n; i++)
  {  
    err = std::max(err, std::abs(y[i].real()-double(n)));
    err = std::max(err, std::abs(y[i].imag()-0.0));
  } 
  EXPECT_NEAR(1.0+err, 1.0, std::numeric_limits<double>::epsilon());
}
 

TEST(tensor_operations, tensor_apply)
{
  // test for tensor operator reshape(a2*C2*W+a1*W*C1^T,N1*N2,1)+V*w
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  // test only for one or two processors
  if (size==1)
  {
     double V[6]={1,2,3,6,3,2};   // Potential matrix/or vector
     double v_in[6]={1,2,1,3,0,4};
     double v_out[6];
     double result[6] = {17, 16, 23, 46, 24, 47}; 
     double C1[4]={1,2,1,3};            // C1 m=2
     double C2[9]={2,2,3,2,2,4,2,2,5};  // C2 n=3
     QRes::Kron2D<double> Koperator(2, C1, 3, C2, V, 1.0, 1.0);
     Koperator.apply(v_in, v_out);
     double err=0.0;
     for (int i=0; i<6; i++) err = std::max(err, std::max(err, std::abs(result[i]-v_out[i])));
  }
  else
  {
     double v_in[3];
     double V[3];
     double result[3];
     double v_out[3];
     double C1[4]={1,2,1,3};            // C1 m=2
     double C2[9]={2,2,3,2,2,4,2,2,5};  // C2 n=3
     if (rank==0)
     {
        double V[3] = {1,2,3}; 
        double v_in[3] = {1,2,1};
        double result[3] = {17, 16, 23};
        QRes::Kron2D<double> Koperator(2, C1, 3, C2, V, 1.0, 1.0);
        Koperator.apply(v_in, v_out);
        double err=0.0;	
        for (int i=0; i<3; i++) err = std::max(err, std::max(err, std::abs(result[i]-v_out[i])));
        EXPECT_NEAR(1.0+err, 1.0, std::numeric_limits<double>::epsilon());  
     }
     else
     {
        double V[3] = {6,3,2};
        double v_in[3] = {3,0,4};
        double result[3] = {46, 24, 47};
        QRes::Kron2D<double> Koperator(2, C1, 3, C2, V, 1.0, 1.0);
        Koperator.apply(v_in, v_out);
        double err=0.0;
        for (int i=0; i<3; i++) err = std::max(err, std::max(err, std::abs(result[i]-v_out[i])));
        EXPECT_NEAR(1.0+err, 1.0, std::numeric_limits<double>::epsilon());  
     }
  }
}


TEST(tensor_operations, correct_apply)
{
  int nR=3;
  int nr=3;
  int N=16;
  double LR=1.0;
  double Lr=1.0;
  
  int loc_n = domain_decomp(nR+1);
  int loc_len = loc_n * (nr+1);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::complex<double> KR[N],CR[N],MR[N],Kr[N],Cr[N],Mr[N],
                       Vp[loc_len],a0[loc_len],v_in[loc_len],v_out[loc_len],vbest[loc_len],z[loc_len];

  buildKmatrix(nR, LR, KR);   // K matrix (nR coordinate)
  buildCmatrix(nR, CR);       // C matrix (nR coordinate)  A problem about C matrix.. complex number? Check!
  buildMmatrix(nR, MR);       // M matrix (nR coordinate)
  buildKmatrix(nr, Lr, Kr);   // K matrix (nr coordinate)
  buildCmatrix(nr, Cr);       // C matrix (nr coordinate)
  buildMmatrix(nr, Mr);       // M matrix (nr coordinate)
  
  buildGaussianPotential3b1d(nR, nr, LR, Lr, 1.0, 1.0, 1.0, Vp);  // (already checked)

  init(loc_len, a0, 0.0);
  QRes::Kron2D<std::complex<double>> Koperator(nR+1, KR, nr+1, Kr, Vp, 1.0, 1.0); // K tensor operator
  QRes::Kron2D<std::complex<double>> Coperator(nR+1, CR, nr+1, Cr, a0, 1.0, 1.0); // C tensor operator
  QRes::Kron2D<std::complex<double>> Moperator(nR+1, MR, nr+1, Mr, a0, 1.0, 1.0); // M tensor operator
  init(loc_len,v_in, 1.0);
  init(loc_len,v_out, 3.1);  // initialize v_out randomly to check the pointer safety
  init(loc_len,vbest, 0.1);
  init(loc_len,z, 1.0);

  QRes::CorrectOp<std::complex<double>> correctOp(loc_len, Koperator, Coperator, Moperator, vbest, z, 3.0);
  correctOp.apply(v_in, v_out);

  std::complex<double> result[N] = {6.6834, -0.0229, -0.0229, 6.6834, 0.1263, -6.7868, -6.7868, 0.1263,
                       	            0.1263, -6.7868, -6.7868, 0.1263, 6.6834, -0.0229, -0.0229, 6.6834};
//if (rank==0){
// for (int i=0; i<loc_len; i++) std::cout << "v_out["<<i<<"]="<<v_out[i] <<std::endl;}
  
  double err=0.0;
  for (int i=0; i<loc_len; i++) err = std::max(err, std::max(err, std::abs(result[rank*loc_len+i]-v_out[i])));
  EXPECT_NEAR(1.0+err, 1.0, 1e-4);

}


TEST(functions, gmres_solver)
{
  int nR=3;
  int nr=3;
  int N=16;
  double LR=1.0;
  double Lr=1.0;

  int loc_n = domain_decomp(nR+1);
  int loc_len = loc_n * (nr+1);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::complex<double> KR[N],CR[N],MR[N],Kr[N],Cr[N],Mr[N],
                       Vp[loc_len],a0[loc_len],x[loc_len],r[loc_len],vbest[loc_len],z[loc_len],b[loc_len];
  using namespace std::complex_literals; 

  buildKmatrix(nR, LR, KR);   // K matrix (nR coordinate)
  buildCmatrix(nR, CR);       // C matrix (nR coordinate)  A problem about C matrix.. complex number? Check!
  buildMmatrix(nR, MR);       // M matrix (nR coordinate)
  buildKmatrix(nr, Lr, Kr);   // K matrix (nr coordinate)
  buildCmatrix(nr, Cr);       // C matrix (nr coordinate)
  buildMmatrix(nr, Mr);       // M matrix (nr coordinate)
  buildGaussianPotential3b1d(nR, nr, LR, Lr, 1.0, 1.0, 1.0, Vp);  // (already checked)
  
  init(loc_len, a0, 0.0);
  QRes::Kron2D<std::complex<double>> Koperator(nR+1, KR, nr+1, Kr, Vp, 1.0, 1.0); // K tensor operator
  QRes::Kron2D<std::complex<double>> Coperator(nR+1, CR, nr+1, Cr, a0, 1.0, 1.0); // C tensor operator	   
  QRes::Kron2D<std::complex<double>> Moperator(nR+1, MR, nr+1, Mr, a0, 1.0, 1.0); // M tensor operator
  init(loc_len, x, 0.0);

  std::complex<double> g_vbest[N] = {0.0644 - 0.1000i, 0.0644 - 0.1000i, 0.0644 - 0.1000i, 0.0644 - 0.1000i,
	      		           0.0644 - 0.1000i, 0.0644 - 0.1000i, 0.0644 - 0.1000i, 0.0644 - 0.1000i,
	      			   0.0644 - 0.1000i, 0.0644 - 0.1000i, 0.0644 - 0.1000i, 0.0644 - 0.1000i,
	     	 		   0.0644 - 0.1000i, 0.0644 - 0.1000i, 0.0644 - 0.1000i, 0.0644 - 0.1000i};
  std::complex<double> g_z[N] = {0.1287 - 0.2001i, -0.1555 - 0.1000i, -0.1555 - 0.1000i, 0.1287 - 0.2001i,
		              -0.1555 - 0.1000i, -0.4398 - 0.0000i, -0.4398 - 0.0000i,-0.1555 - 0.1000i,
	 		      -0.1555 - 0.1000i, -0.4398 - 0.0000i, -0.4398 - 0.0000i,-0.1555 - 0.1000i,
	  		       0.1287 - 0.2001i, -0.1555 - 0.1000i, -0.1555 - 0.1000i, 0.1287 - 0.2001i};
  std::complex<double> g_b[N] = {-0.3592 - 0.1253i, 0.0018 - 0.0027i, 0.0018 - 0.0027i, -0.3592 - 0.1253i,
	  		       -0.0097 + 0.0150i, 0.3671 + 0.1130i, 0.3671 + 0.1130i, -0.0097 + 0.0150i,
	  		       -0.0097 + 0.0150i, 0.3671 + 0.1130i, 0.3671 + 0.1130i, -0.0097 + 0.0150i,
	  		       -0.3592 - 0.1253i, 0.0018 - 0.0027i, 0.0018 - 0.0027i, -0.3592 - 0.1253i};
  for (int i=0; i<loc_len; i++)
  {
     vbest[i] = g_vbest[rank*loc_len +i];
     z[i] = g_z[rank*loc_len +i];
     b[i] = g_b[rank*loc_len +i];
  }

  QRes::CorrectOp<std::complex<double>> correctOp(loc_len, Koperator, Coperator, Moperator, vbest, z, 1.0000+1.5547i); 

  double resNorm;
  int numIter;
  gmres_solver(&correctOp, loc_len, x, b, 1e-8, 100, &resNorm, &numIter, 0);
  
  correctOp.apply(x,r);                    
  axpby(N, 1.0, b, -1.0, r); 

  double err = std::sqrt(complex_dot(loc_len,r,r).real())/std::sqrt(complex_dot(loc_len,b,b).real());
  EXPECT_NEAR(1.0+err, 1.0, 10*std::numeric_limits<double>::epsilon());
}

/*
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
     norm = complex_dot(m, V+m*i, V+m*i);
     err_nr = std::max(err_nr, norm.real());
     err_ni = std::max(err_ni, norm.imag());
  }
  EXPECT_NEAR(err_nr, 1.0, 10*std::numeric_limits<double>::epsilon());
  EXPECT_NEAR(1.0+err_ni, 1.0, 10*std::numeric_limits<double>::epsilon());

  double err_rr = 0.0;
  double err_ri = 0.0;
  for (int i=1; i<n; i++)
     for (int j=0; j<i; j++)
     {
        norm = complex_dot(m, V+m*j, V+m*i);
        err_rr = std::max(err_rr, norm.real());
        err_ri = std::max(err_ri, norm.imag());
     }

  EXPECT_NEAR(1.0+err_rr, 1.0, 10*std::numeric_limits<double>::epsilon());
  EXPECT_NEAR(1.0+err_ri, 1.0, 10*std::numeric_limits<double>::epsilon());
}

*/

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


TEST(buildoperator, Kmatrix)
{
  int n =5;
  double L=1.0;
  
  // double ChebMat[(n+1)*(n+1)];
  std::complex<double>* K = (std::complex<double>*)malloc((n+1)*(n+1)*sizeof(std::complex<double>));
  buildKmatrix(n, L, K);

  double true_K[(n+1)*(n+1)] =
  {8.5000,  -10.4721,  2.8944,  -1.5279,  1.1056,  -0.5000,
  -10.6430,  15.7666, -6.3416,   1.8472, -1.1056,   0.4764,
   0.9236,   -3.6584,  5.0334,  -2.8944,  0.9528,  -0.3570,
  -0.3570,    0.9528, -2.8944,   5.0334, -3.6584,   0.9236,
  0.4764,   -1.1056,  1.8472,  -6.3416,  15.7666, -10.6430,
  -0.5000,    1.1056, -1.5279,   2.8944, -10.4721,  8.5000};
  
   double err = 0.0;
  for (int i=0; i<(n+1)*(n+1); i++)
      err = std::max(err, std::max(err, std::abs(K[i]-true_K[i])));

  EXPECT_NEAR(1.0+err, 1.0, 0.0001);  
  free(K);
}




