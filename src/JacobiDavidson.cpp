#include "mpi.h"
#include "header.hpp"
#include "tensor3b1d.hpp"
#include "operations.hpp"
#include "gmres_solver.hpp"

std::unique_ptr<resultJD> JacobiDavidson(int nR,int nr,double LR,double Lr,std::map<std::string, std::string> jdopts)
{
  //--- mpi ---//
  int proc_rank;                             // processor rank
  int proc_numb;                             // number of processors
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank); // get processor rank
  MPI_Comm_size(MPI_COMM_WORLD, &proc_numb); // get processor number

  //--- some settings for JacobiDavidson ---//
  using namespace std::complex_literals; 
  int N = (nr+1)*(nR+1);       // total problem size 
  int numeigs = stoi(jdopts["numeigs"]);        // number of eigenvalues desired
  int mindim = stoi(jdopts["mindim"]);          // minimum dimension of search space V
  int maxdim = stoi(jdopts["maxdim"]);          // maximum dimension of search space V
  int maxiter = stoi(jdopts["maxiter"]);        // maximum number of iterations 
  double tolerance = stod(jdopts["tolerance"]); // tolerance of residual norm
  int verbose = stoi(jdopts["verbose"]);        // verbose controlling cout
  std::complex<double> sigma(stod(jdopts["target_real"]),stod(jdopts["target_imag"])); // target (shift) (complex)
  
  //--- some settings for GMRES ---//
  double resNorm_gmres;
  double tol_gmres = 1e-4; 
  int numIter_gmres;
  int maxiter_gmres = 30; 
  int verbose_gmres = 0;

  //--- there are three things stored in the result: eigenvalue, eigenvector and convergence history ---//
  std::unique_ptr<resultJD> result_ptr(new resultJD);                                           // unique pointer used to store the result
  result_ptr->eigval = (std::complex<double>*)malloc(sizeof(std::complex<double>)*numeigs);     // eigenvalues detected
  result_ptr->eigvec = (std::complex<double>*)malloc(sizeof(std::complex<double>)*numeigs*N);   // corresponding eigenvectors
  result_ptr->cvg_hist = (double*)malloc(sizeof(double)*maxiter);                               // convergence history

  //--- memory allocation ---//
  std::complex<double>* KR = (std::complex<double>*)malloc(sizeof(std::complex<double>)*(nR+1)*(nR+1));
  std::complex<double>* CR = (std::complex<double>*)malloc(sizeof(std::complex<double>)*(nR+1)*(nR+1));
  std::complex<double>* MR = (std::complex<double>*)malloc(sizeof(std::complex<double>)*(nR+1)*(nR+1));
  std::complex<double>* Kr = (std::complex<double>*)malloc(sizeof(std::complex<double>)*(nr+1)*(nr+1));
  std::complex<double>* Cr = (std::complex<double>*)malloc(sizeof(std::complex<double>)*(nr+1)*(nr+1));
  std::complex<double>* Mr = (std::complex<double>*)malloc(sizeof(std::complex<double>)*(nr+1)*(nr+1));
  std::complex<double>* Vp = (std::complex<double>*)malloc(sizeof(std::complex<double>)*N);
  std::complex<double>* a0 = (std::complex<double>*)malloc(sizeof(std::complex<double>)*N);
  std::complex<double>* res= (std::complex<double>*)malloc(sizeof(std::complex<double>)*N);
  std::complex<double>* z = (std::complex<double>*)malloc(sizeof(std::complex<double>)*N);
  std::complex<double>* t = (std::complex<double>*)malloc(sizeof(std::complex<double>)*N);
  std::complex<double>* b = (std::complex<double>*)malloc(sizeof(std::complex<double>)*N); 
  double resNorm;

  //--- build tensor operator ---//
  buildKmatrix(nR, LR, KR);   // K matrix (nR coordinate)
  buildCmatrix(nR, CR);       // C matrix (nR coordinate)  A problem about C matrix.. complex number? Check!
  buildMmatrix(nR, MR);       // M matrix (nR coordinate)
  buildKmatrix(nr, Lr, Kr);   // K matrix (nr coordinate)
  buildCmatrix(nr, Cr);       // C matrix (nr coordinate)
  buildMmatrix(nr, Mr);       // M matrix (nr coordinate)
  buildGaussianPotential3b1d(nR, nr, LR, Lr, 1.0, 1.0, 1.0, Vp);  // (already checked)
  complex_init(N, a0, 0.0);
  QRes::Kron2D<std::complex<double>> Koperator(nR+1, KR, nr+1, Kr, Vp, 1.0, 1.0); // K tensor operator
  QRes::Kron2D<std::complex<double>> Coperator(nR+1, CR, nr+1, Cr, a0, 1.0, 1.0); // C tensor operator	   
  QRes::Kron2D<std::complex<double>> Moperator(nR+1, MR, nr+1, Mr, a0, 1.0, 1.0); // M tensor operator
	   
  //--- search space V (stored by column major!) ---//
  std::complex<double>* V = (std::complex<double>*)malloc(sizeof(std::complex<double>)*N*maxdim); // search space V (N*maxdim)
  std::complex<double>* v = (std::complex<double>*)malloc(sizeof(std::complex<double>)*N*mindim); // minor-search space v (N*mindim) used to store space vectors temporarily
  std::complex<double>* Kv = (std::complex<double>*)malloc(sizeof(std::complex<double>)*N);       // tensor_K*V[:,i] -> Kv
  std::complex<double>* Cv = (std::complex<double>*)malloc(sizeof(std::complex<double>)*N);       // tensor_C*V[:,i] -> Cv
  std::complex<double>* Mv = (std::complex<double>*)malloc(sizeof(std::complex<double>)*N);       // tensor_M*V[:,i] -> Mv
  std::complex<double>* vbest = (std::complex<double>*)malloc(sizeof(std::complex<double>)*N);    // best Ritz vector
  complex_init(N, V, 1.0/std::sqrt(N)); // initialize and normalize the first column of search space V
  
  int iter = 0;     // iteration number
  int detected = 0; // number of eigenpairs found
  int Vdim = 1;     // dimension of the search space V
  
  //--- start Jacobi-Davidson iteration ---//
  while(iter<maxiter && detected<numeigs)
  {
    /* Parallelization MPI+OpenMP 
     * Rayleigh Ritz projection
     * tips: tall skinny matrix ... 
     *      + 
     * Linearization
     * solve the linear eigenvalue problem by <eigen>*/
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

    /* Assemble the standard eigenvalue problem (completely sequential) 
     * -                 -
     * | -M^-1*C -M^-1*K |
     * |    I      O     |
     * -                 -
     * */
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

    /* Solve the projected standard eigenvalue problem <Eigen/Eigenvalues> */
    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> linearEigSolv;
    linearEigSolv.compute(Amat);
     
    /* different from matlab codes: here we don't compute the 
     * Ritz vectors v=V*c since we don't need all vectors. */
    
    /* sort the eigenvalues */
    Eigen::VectorXcd proj_eigval = linearEigSolv.eigenvalues();
    std::vector<int> idx_eigen(2*Vdim);
    std::iota(idx_eigen.begin(), idx_eigen.end(), 0);  
    bool sflag;
    int temp1;
    std::complex<double> temp2;

    do{
       sflag = false;
       for (int i=0; i<2*Vdim-1; i++)
       {
          if (std::abs(proj_eigval[i] - sigma) > std::abs(proj_eigval[i+1] - sigma))
	  {
	     temp1 = idx_eigen[i];	  
             temp2 = proj_eigval[i];
	     idx_eigen[i] = idx_eigen[i+1];
	     proj_eigval[i] = proj_eigval[i+1];
             idx_eigen[i+1] = temp1;
	     proj_eigval[i+1] = temp2;	     
	     sflag = true; 
	  }
       }
    }while(sflag);

    int idxBest = idx_eigen[detected];
    std::complex<double> thetaBest = proj_eigval[detected];
    std::complex<double> cbest[Vdim];
    for (int i=0; i<Vdim; i++) cbest[i] = linearEigSolv.eigenvectors().col(idxBest)[i+Vdim];

    /* most promising Ritz vector (projected back) */
    complex_init(N, vbest, 0.0);
    for (int i=0; i<Vdim; i++) complex_axpby(N, cbest[i], V+i*N, 1.0, vbest); 

    /* residual */
    Coperator.apply(vbest, Cv);
    Moperator.apply(vbest, Mv);
    Koperator.apply(vbest, res);
    complex_axpby(N, thetaBest, Cv, 1.0, res);
    complex_axpby(N, thetaBest*thetaBest, Mv, 1.0, res);
    resNorm = std::sqrt(complex_dot(N, res, res).real());
    if (verbose==1) std::cout << "iteration " << iter << ", residual norm: " << resNorm;
    result_ptr->cvg_hist[iter] = resNorm;

    /* if converge... */
    if (resNorm < tolerance)
    {
       result_ptr->eigval[detected] = thetaBest;
       vec_update(N, 1.0, vbest, result_ptr->eigvec+detected*N);
       vec_update(N, 1.0, vbest, V+detected*N);
       sigma = thetaBest;

       /* reduce the search space if necessary? */ 

       thetaBest = proj_eigval[detected+1];
       idxBest = idx_eigen[detected+1];
       complex_init(N, vbest, 0.0);
       for (int i=0; i<Vdim; i++) cbest[i] = linearEigSolv.eigenvectors().col(idxBest)[i+Vdim];
       for (int i=0; i<Vdim; i++) complex_axpby(N, cbest[i], V+i*N, 1.0, vbest); 
     
       Coperator.apply(vbest, Cv);
       Moperator.apply(vbest, Mv);
       Koperator.apply(vbest, res);
       complex_axpby(N, thetaBest, Cv, 1.0, res);
       complex_axpby(N, thetaBest*thetaBest, Mv, 1.0, res);

       detected += 1;
    }

    /* reduce the search space if necessary */
    if (Vdim == maxdim)
    {	    
       for (int j=0; j<mindim; j++)
       {
	  complex_init(N, v+j*N, 0.0);     
          for (int i=0; i<Vdim; i++) cbest[i] = linearEigSolv.eigenvectors().col(idx_eigen[j])[i+Vdim];
          for (int i=0; i<Vdim; i++) complex_axpby(N, cbest[i], V+i*N, 1.0, v+j*N);  
       }
       vec_update(N*mindim, 1.0, v, V);
       if (detected != 0) vec_update(N*detected, 1.0, result_ptr->eigvec, V);
       Vdim = mindim;
    }
     
    /* solve the (preconditioned) correction equation */
    Coperator.apply(vbest, z);
    complex_axpby(N, 2.0*thetaBest, Mv, 1.0, z); 
    QRes::CorrectOp<std::complex<double>> correctOp(N, Koperator, Coperator, Moperator, vbest, z, thetaBest); 
    
    double resNorm_gmres;
    int numIter_gmres;
    complex_init(N, t, 0.0);
    vec_update(N, -1.0, res, b); 
    gmres_solver(&correctOp, N, t, b, tol_gmres, maxiter_gmres, &resNorm_gmres, &numIter_gmres, verbose_gmres);
    std::cout << ", residual norm of gmres: "<< resNorm_gmres << std::endl;

    /* expand the search space V */
    
  //for (int i=0; i<N; i++) V[i+Vdim*N]=double(rand()%10)/10;
   
    

    Vdim += 1;   // search space dimension +1
    iter += 1;   // iteration +1

    break;
  }

  for (int i=0; i< numeigs; i++)  result_ptr->eigval[i] = 1i;

// proc_rank=1;
  // result_ptr->eigvec = (double*)malloc(5 * (nR+1) * (nr+1) * sizeof(double));
 
  free(KR); free(CR); free(MR);
  free(Kr); free(Cr); free(Mr);
  free(Vp); free(a0); free(res);
  free(V); free(Kv); free(Cv); free(Mv); free(vbest);
  free(z); free(t); free(b);
 
  return result_ptr;
}
