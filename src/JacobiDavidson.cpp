#include "mpi.h"
#include "header.hpp"
#include "tensor3b1d.hpp"
#include "operations.hpp"

std::unique_ptr<resultJD> JacobiDavidson(int nR,int nr,double LR,double Lr,std::map<std::string, std::string> jdopts)
{
  //--- some settings ---//
  using namespace std::complex_literals; 
  int N = (nr+1)*(nR+1);       //
  int proc_rank;               // processor rank
  int proc_numb;               // number of processors
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank); // get processor rank
  MPI_Comm_size(MPI_COMM_WORLD, &proc_numb); // get processor number

  int numeigs = stoi(jdopts["numeigs"]);        // number of eigenvalues desired
  int mindim = stoi(jdopts["mindim"]);          // minimum dimension of search space V
  int maxdim = stoi(jdopts["maxdim"]);          // maximum dimension of search space V
  int maxiter = stoi(jdopts["maxiter"]);        // maximum number of iterations 
  double tolerance = stod(jdopts["tolerance"]); // tolerance of residual norm
  // v0 ... (different from matlab)
  // lsolver (gmres)
  // precond (shift Hamiltonian)
  int verbose = stoi(jdopts["verbose"]);        // verbose controlling cout
  std::complex<double> sigma(stod(jdopts["target_real"]),stod(jdopts["target_imag"])); // target (shift) (complex)
  
  // there are three things stored in the result: eigenvalue, eigenvector and convergence history
  std::unique_ptr<resultJD> result_ptr(new resultJD);
  result_ptr->eigval = (std::complex<double>*)malloc(sizeof(std::complex<double>)*numeigs);
  result_ptr->eigvec = (std::complex<double>*)malloc(sizeof(std::complex<double>)*numeigs*(nR+1)*(nr+1));
  result_ptr->cvg_hist = (double*)malloc(sizeof(double)*maxiter);

  //--- build tensor operator ---//
  std::complex<double>* KR = (std::complex<double>*)malloc(sizeof(std::complex<double>)*(nR+1)*(nR+1));
  std::complex<double>* CR = (std::complex<double>*)malloc(sizeof(std::complex<double>)*(nR+1)*(nR+1));
  std::complex<double>* MR = (std::complex<double>*)malloc(sizeof(std::complex<double>)*(nR+1)*(nR+1));
  std::complex<double>* Kr = (std::complex<double>*)malloc(sizeof(std::complex<double>)*(nr+1)*(nr+1));
  std::complex<double>* Cr = (std::complex<double>*)malloc(sizeof(std::complex<double>)*(nr+1)*(nr+1));
  std::complex<double>* Mr = (std::complex<double>*)malloc(sizeof(std::complex<double>)*(nr+1)*(nr+1));
  std::complex<double>* Vp = (std::complex<double>*)malloc(sizeof(std::complex<double>)*(nR+1)*(nr+1));
  std::complex<double>* a0 = (std::complex<double>*)malloc(sizeof(std::complex<double>)*(nR+1)*(nr+1));

  buildKmatrix(nR, LR, KR);   // K matrix (nR coordinate)
  buildCmatrix(nR, CR);       // C matrix (nR coordinate)
  buildMmatrix(nR, MR);       // M matrix (nR coordinate)
  buildKmatrix(nr, Lr, Kr);   // K matrix (nr coordinate)
  buildCmatrix(nr, Cr);       // C matrix (nr coordinate)
  buildMmatrix(nr, Mr);       // M matrix (nr coordinate)

  buildGaussianPotential3b1d(nR, nr, LR, Lr, 1.0, 1.0, 1.0, Vp);
  complex_init((nR+1)*(nr+1), a0, 0.0+0.0i);

  QRes::Kron2D<std::complex<double>> Koperator(nR+1, KR, nr+1, Kr, Vp, 1.0, 1.0);
  QRes::Kron2D<std::complex<double>> Coperator(nR+1, CR, nr+1, Cr, a0, 1.0, 1.0);	   
  QRes::Kron2D<std::complex<double>> Moperator(nR+1, MR, nr+1, Mr, a0, 1.0, 1.0);
	   
  // search space V (stored by column major!)
  std::complex<double>* V = (std::complex<double>*)malloc(sizeof(std::complex<double>)*N*maxdim);
  complex_init(N, V, 1.0/std::sqrt(N)); // initialize and normalize the first column of search space V
  
  
  int iter = 0;     // iteration number
  int detected = 0; // number of eigenpairs found
  
  //--- start Jacobi-Davidson iteration ---//
  while(true)
  {
    /* Parallelization MPI+OpenMP 
     * Rayleigh Ritz projection
     * tips: tall skinny matrix ... */
    
     
    /* Linearization
     * solve the linear eigenvalue problem by LAPACK
     * */


    break;
  }

  for (int i=0; i< numeigs; i++)  result_ptr->eigval[i] = 1i;

// proc_rank=1;
  // result_ptr->eigvec = (double*)malloc(5 * (nR+1) * (nr+1) * sizeof(double));
 
  free(KR); free(CR); free(MR);
  free(Kr); free(Cr); free(Mr);
  free(Vp); free(a0);
  free(V);

  return result_ptr;
}
