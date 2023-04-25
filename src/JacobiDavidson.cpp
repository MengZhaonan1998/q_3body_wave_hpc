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
  buildCmatrix(nR, CR);       // C matrix (nR coordinate)  A problem about C matrix.. complex number? Check!
  buildMmatrix(nR, MR);       // M matrix (nR coordinate)
  buildKmatrix(nr, Lr, Kr);   // K matrix (nr coordinate)
  buildCmatrix(nr, Cr);       // C matrix (nr coordinate)
  buildMmatrix(nr, Mr);       // M matrix (nr coordinate)

  //complex_init(N, Vp, 0.0+0.0i);
  buildGaussianPotential3b1d(nR, nr, LR, Lr, 1.0, 1.0, 1.0, Vp);  // (already checked)
  complex_init(N, a0, 0.0);

  QRes::Kron2D<std::complex<double>> Koperator(nR+1, KR, nr+1, Kr, Vp, 1.0, 1.0);
  QRes::Kron2D<std::complex<double>> Coperator(nR+1, CR, nr+1, Cr, a0, 1.0, 1.0);	   
  QRes::Kron2D<std::complex<double>> Moperator(nR+1, MR, nr+1, Mr, a0, 1.0, 1.0);
	   
  // search space V (stored by column major!)
  std::complex<double>* V = (std::complex<double>*)malloc(sizeof(std::complex<double>)*N*maxdim);
  std::complex<double>* Kv = (std::complex<double>*)malloc(sizeof(std::complex<double>)*N);
  std::complex<double>* Cv = (std::complex<double>*)malloc(sizeof(std::complex<double>)*N);
  std::complex<double>* Mv = (std::complex<double>*)malloc(sizeof(std::complex<double>)*N);

  complex_init(N, V, 1.0/std::sqrt(N)); // initialize and normalize the first column of search space V
  
  int iter = 0;     // iteration number
  int detected = 0; // number of eigenpairs found
  int Vdim = 1;     // dimension of the search space V
  
  //--- start Jacobi-Davidson iteration ---//
  while(true)
  {
    /* Parallelization MPI+OpenMP 
     * Rayleigh Ritz projection
     * tips: tall skinny matrix ... 
     *      + 
     * Linearization
     * solve the linear eigenvalue problem by LAPACK*/
    Eigen::MatrixXcd Amat = Eigen::MatrixXcd(2*Vdim,2*Vdim); 
    Eigen::MatrixXcd Kmat = Eigen::MatrixXcd(Vdim,Vdim);
    Eigen::MatrixXcd Cmat = Eigen::MatrixXcd(Vdim,Vdim);
    Eigen::MatrixXcd Mmat = Eigen::MatrixXcd(Vdim,Vdim);

    for (int i=0; i<Vdim; i++)
       for (int j=0; j<Vdim; j++)
       {
          Koperator.apply(V+j*N, Kv);   // K*H -> K*h (single vector operator)
	  Coperator.apply(V+j*N, Cv);   // C*H -> C*h (single vector operator)
	  Moperator.apply(V+j*N, Mv);   // M*H -> M*h (single vector operator)

          Kmat(i,j) = complex_dot(N, V+i*N, Kv);  // entry of K 
          Cmat(i,j) = complex_dot(N, V+i*N, Cv);  // entry of C
          Mmat(i,j) = complex_dot(N, V+i*N, Mv);  // entry of M
       }

    //--- Assemble the standard eigenvalue problem (completely sequential)---//
    Mmat = Mmat.inverse();  // inverse of matrix M
    Cmat = Mmat * Cmat;
    Kmat = Mmat * Kmat;
    for (int i=0; i<Vdim; i++)
       for (int j=0; j<Vdim; j++)
       {
          Amat(i,j) = -Cmat(i,j);
          Amat(i,j+Vdim) = -Kmat(i,j);
          Amat(i+Vdim,j) = (i==j) ? 1.0:0.0;
          Amat(i+Vdim,j+Vdim) = 0.0;	  
       }
 
    /* Solve the projected standard eigenvalue problem 
     * <Eigen/Eigenvalues> */
    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> linearEigSolv;
    linearEigSolv.compute(Amat);
    /*std::cout << "The eigenvalues of Amat are" << std::endl <<  linearEigSolv.eigenvalues() << std::endl;
      std::cout << "The matrix of eigenvectors V is" << std::endl << linearEigSolv.eigenvectors() << std::endl;
      Eigen::VectorXcd v = Amat * linearEigSolv.eigenvectors().col(0) - linearEigSolv.eigenvalues()[0] * linearEigSolv.eigenvectors().col(0);
      std::cout << "A*v-lambda*v" << std::endl << v << std::endl;*/

 
    

     /*   for (int i=0; i<2*Vdim; i++){
       for (int j=0; j<2*Vdim; j++)
       {   std::cout << Amat(i,j) << "  ";}
       std::cout << "\n";
    }*/

    Vdim += 1;   // search space dimension +1

    break;
  }

  for (int i=0; i< numeigs; i++)  result_ptr->eigval[i] = 1i;

// proc_rank=1;
  // result_ptr->eigvec = (double*)malloc(5 * (nR+1) * (nr+1) * sizeof(double));
 
  free(KR); free(CR); free(MR);
  free(Kr); free(Cr); free(Mr);
  free(Vp); free(a0);
  free(V); free(Kv); free(Cv); free(Mv);

  return result_ptr;
}
