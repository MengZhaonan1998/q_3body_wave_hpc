#include "timer.hpp"
#include "operations.hpp"
#include "tensor3b1d.hpp"
#include "gmres_solver.hpp"
#include "header.hpp"

//--- given rotation ---//
void given_rotation(int k, std::complex<double>* h, std::complex<double>* cs, std::complex<double>* sn)
{
  std::complex<double> temp, t, cs_k, sn_k;
  for (int i=0; i<k; i++)
  {
     temp = cs[i] * h[i] + sn[i] * h[i+1];
     h[i+1] = -sn[i] * h[i] + cs[i] * h[i+1];
     h[i] = temp;
  }
  
  // update the next sin cos values for rotation
  t = std::sqrt( h[k]*h[k] + h[k+1]*h[k+1] );
  cs[k] = h[k]/t;
  sn[k] = h[k+1]/t;

  // eliminate H(i+1,i)
  h[k] = cs[k]*h[k] + sn[k]*h[k+1];
  h[k+1] = 0.0;

  return;
}

//--- Arnoldi function ---//      
void arnoldi(int k, std::complex<double>* Q, std::complex<double>* h, QRes::CorrectOp<std::complex<double>>* op) 
{
  int n = op->N_;
  op->apply(Q+k*n, Q+(k+1)*n);

  for (int i=0; i<=k; i++)
  {
    h[i] = complex_dot(n, Q+(k+1)*n, Q+i*n);
    complex_axpby(n, -h[i], Q+i*n, 1.0, Q+(k+1)*n);
  }

  h[k+1] = std::sqrt(complex_dot(n, Q+(k+1)*n, Q+(k+1)*n));

  for (int i=0; i<n; i++)
    Q[(k+1)*n+i] = Q[(k+1)*n+i] / h[k+1];

 return; 
}

//--- GMRES solver ---//
void gmres_solver(QRes::CorrectOp<std::complex<double>>* op, int n, 
	std::complex<double>* x, std::complex<double> const* b,
        double tol, int maxIter,
        double* resNorm, int* numIter,
        int verbose)
{
  std::complex<double>* r  = new std::complex<double>[n]; // residual vector
  std::complex<double>* sn = new std::complex<double>[maxIter]; // used in given rotation
  std::complex<double>* cs = new std::complex<double>[maxIter]; // used in given rotation
  std::complex<double>* e1 = new std::complex<double>[maxIter+1];
  std::complex<double>* beta=new std::complex<double>[maxIter+1];

  std::complex<double>* Q = new std::complex<double>[(maxIter+1) * n];                    // note that Q is stored by column major
  std::complex<double>* H = new std::complex<double>[((maxIter+1) * maxIter)/2+maxIter];  // note that H is stored by column major

  double r_norm;             // residual norm
  double b_norm;             // right hand side b norm

  // r=b-A*x
  op->apply(x,r);            // r = op * x
  complex_axpby(n, 1.0, b, -1.0, r); // r = b - r

  // compute the error
  r_norm = std::sqrt(complex_dot(n,r,r).real());
  b_norm = std::sqrt(complex_dot(n,b,b).real());
  double error = r_norm/b_norm;  // here I use the relative error instead of the residual norm

  // initialize the 1D vectors
  complex_init(maxIter, sn, 0.0); 
  complex_init(maxIter, cs, 0.0);
  complex_init(maxIter+1, e1, 0.0);
  complex_init(maxIter+1, beta, 0.0);
  e1[0]=1.0;
  beta[0]=r_norm;

  /* initialize Q and H;
   * Q and H have special data structures.
   * Q as a 1D double array stores a 2D matrix by column major
   * H is an upper Hessenburg matrix. To save the memory we only store the necessary elements, i.e.
   * __                 __
   * | H[0] H[2] H[5] ...|
   * | H[1] H[3] H[6] ...|
   * |      H[4] H[7] ...|
   * |           H[8] ...|
   * |                ...|
   * __                 __
   */

  complex_init((maxIter+1) * n, Q, 0.0);
  complex_init(((maxIter+1) * maxIter)/2+maxIter, H, 0.0);
  for (int i=0;i<n;i++) Q[i]=r[i]/r_norm;  

  // start GMRES iteration
  int iter = -1;
  while (true)
  {
    iter++;

    if (verbose)
    {
      std::cout << std::setw(4) << iter << "\t" << std::setw(8) << std::setprecision(4) << error << std::endl;
    }

    // check for convergence or failure
    if ( (error < tol) || (iter == maxIter) )
    {
      break;
    }

    arnoldi(iter, Q, H + iter*(iter+1)/2+iter, op);          // operation: Arnoldi process 

    given_rotation(iter, H + iter*(iter+1)/2+iter, cs, sn);  // operation: Given rotation

    beta[iter+1] = -sn[iter]*beta[iter];
    beta[iter] = cs[iter]*beta[iter];
    error = std::abs( beta[iter+1].real() ) / b_norm;

  } // end of while-loop

  // backward substitution
  std::complex<double>* y = new std::complex<double>[iter];
  complex_init(iter, y, 0.0);
  for (int i=0; i<iter; i++) 
  {
     for (int j=0; j<i; j++)
     {  
        beta[iter-1-i] -=  y[iter-1-j] * H[((iter-j+1) * (iter-j))/2 +iter-j-2-(i-j)]; 
     }
     y[iter-1-i] = beta[iter-1-i] / H[((iter-i+1) * (iter-i))/2+iter-i-2]; 
  }

  // x = x + Q*y
  for (int i=0; i<iter; i++) complex_axpby(n, y[i], Q+i*n, 1.0, x);

  // return number of iterations and achieved residual (or should I return error=norm_r/norm_b ?)
  *resNorm = beta[iter].real();
  *numIter = iter;

  // clean up
  delete [] r;
  delete [] sn;
  delete [] cs;
  delete [] e1;
  delete [] Q;
  delete [] H;
  delete [] y;
  delete [] beta;

  return;
}
