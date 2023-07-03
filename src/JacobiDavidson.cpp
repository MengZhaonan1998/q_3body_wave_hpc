#include "mpi.h"
#include "header.hpp"
#include "tensor3b1d.hpp"
#include "operations.hpp"
#include "gmres_solver.hpp"

std::unique_ptr<resultJD> JacobiDavidson(std::map<std::string, std::string> b3d1opts,
		                         std::map<std::string, std::string> jdopts,
					 std::map<std::string, std::string> gmresopts)
{
  //--- MPI Initialization ---//
  int rank;                               // processor rank
  int size;                               // number of processors
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // get processor rank
  MPI_Comm_size(MPI_COMM_WORLD, &size);   // get processor number

  //--- get settings for 3b1d problem ---//
  int nR = stoi(b3d1opts["nR"]);          // number of grid points on R
  int nr = stoi(b3d1opts["nr"]);          // number of grid points on r
  double LR = stod(b3d1opts["LR"]);       // R cutoff 
  double Lr = stod(b3d1opts["Lr"]);       // r cutoff

  //--- get settings for JacobiDavidson ---//
  int numeigs = stoi(jdopts["numeigs"]);        // number of eigenvalues desired
  int mindim  = stoi(jdopts["mindim"]);         // minimum dimension of search space V
  int maxdim  = stoi(jdopts["maxdim"]);         // maximum dimension of search space V
  int maxiter = stoi(jdopts["maxiter"]);        // maximum number of iterations 
  int verbose = stoi(jdopts["verbose"]);        // verbose controlling cout
  double tolerance = stod(jdopts["tolerance"]); // tolerance of residual norm
  std::complex<double> sigma(stod(jdopts["target_real"]),stod(jdopts["target_imag"])); // target (shift) (complex)
  
  //--- get settings for GMRES ---//
  int numIter_gmres;
  int maxiter_gmres= stoi(gmresopts["maxiter"]);   // maximum iterations of GMRES
  int verbose_gmres =stoi(gmresopts["verbose"]);   // output or not
  double tol_gmres = stod(gmresopts["tolerance"]); // tolerance of GMRES
  double resNorm_gmres;

  //--- domain decomposition ---// 
  int N = (nR+1)*(nr+1);             // total problem size 
  int loc_nR = domain_decomp(nR+1);  // partition nR dimension
  int loc_N = loc_nR * (nr+1);       // local problem size

  //--- there are three things stored in the result: eigenvalue, eigenvector and convergence history ---//
  std::unique_ptr<resultJD> result_ptr(new resultJD);                                               // unique pointer used to store the result
  result_ptr->eigval = (std::complex<double>*)malloc(sizeof(std::complex<double>)*numeigs);         // eigenvalues detected
  result_ptr->eigvec = (std::complex<double>*)malloc(sizeof(std::complex<double>)*numeigs*loc_N);   // corresponding eigenvectors
  result_ptr->cvg_hist = (double*)malloc(sizeof(double)*maxiter);                                   // convergence history

  //--- memory allocation ---//  
  std::complex<double>* KR = (std::complex<double>*)malloc(sizeof(std::complex<double>)*(nR+1)*(nR+1));
  std::complex<double>* CR = (std::complex<double>*)malloc(sizeof(std::complex<double>)*(nR+1)*(nR+1));
  std::complex<double>* Kr = (std::complex<double>*)malloc(sizeof(std::complex<double>)*(nr+1)*(nr+1));
  std::complex<double>* Cr = (std::complex<double>*)malloc(sizeof(std::complex<double>)*(nr+1)*(nr+1));
  std::complex<double>* Vp = (std::complex<double>*)malloc(sizeof(std::complex<double>)*(nR+1)*(nr+1));
  std::complex<double>* a0 = (std::complex<double>*)malloc(sizeof(std::complex<double>)*loc_N);
  std::complex<double>* res= (std::complex<double>*)malloc(sizeof(std::complex<double>)*loc_N);
  std::complex<double>* z = (std::complex<double>*)malloc(sizeof(std::complex<double>)*loc_N);
  std::complex<double>* t = (std::complex<double>*)malloc(sizeof(std::complex<double>)*loc_N);
  std::complex<double>* b = (std::complex<double>*)malloc(sizeof(std::complex<double>)*loc_N); 
  double resNorm;

  //--- build stiffness, damping and mass operator ---//
  buildKmatrix(nR, LR, KR);           // K matrix (nR coordinate) 
  buildKmatrix(nr, Lr, Kr);           // K matrix (nr coordinate)
  buildCmatrix_complex(nR, CR);       // C matrix (nR coordinate) 
  buildCmatrix_complex(nr, Cr);       // C matrix (nr coordinate)
 
  if (b3d1opts["pot"]=="G")
  {
   double V12 = stod(b3d1opts["V12"]);
   double V13 = stod(b3d1opts["V13"]);
   double V23 = stod(b3d1opts["V23"]);
   buildGaussianPotential3b1d(nR, nr, LR, Lr, V12, V13, V23, Vp);  // Gaussian potential (already checked)
  }
  else
  {
   std::cout << "only gaussian potential is supported so far ..." << std::endl;
  }

  init(loc_N, a0, 0.0);
  QRes::Kron2D<std::complex<double>> Koperator(nR+1, KR, nr+1, Kr, Vp, 1.0, 1.0); // K tensor operator
  QRes::Kron2D<std::complex<double>> Coperator(nR+1, CR, nr+1, Cr, a0, 1.0, 1.0); // C tensor operator	    
  QRes::DiagOp<std::complex<double>> Moperator(loc_N, -0.5);                      // M diagonal operator

  //--- search space V (stored by column major!) ---//
  std::complex<double>* V = (std::complex<double>*)malloc(sizeof(std::complex<double>)*loc_N*maxdim); // search space V (N*maxdim)
  std::complex<double>* v = (std::complex<double>*)malloc(sizeof(std::complex<double>)*loc_N*mindim); // minor-search space v (N*mindim) used to store space vectors temporarily
  std::complex<double>* Kv = (std::complex<double>*)malloc(sizeof(std::complex<double>)*loc_N);       // tensor_K*V[:,i] -> Kv
  std::complex<double>* Cv = (std::complex<double>*)malloc(sizeof(std::complex<double>)*loc_N);       // tensor_C*V[:,i] -> Cv
  std::complex<double>* Mv = (std::complex<double>*)malloc(sizeof(std::complex<double>)*loc_N);       // tensor_M*V[:,i] -> Mv
  std::complex<double>* vbest = (std::complex<double>*)malloc(sizeof(std::complex<double>)*loc_N);    // best Ritz vector
  init(loc_N, V, 1.0/std::sqrt(N));    // initialize and normalize the first column of search space V
  
  int iter = 0;                        // iteration number
  int detected = 0; 		       // number of eigenpairs found
  int Vdim = 1;                        // dimension of the search space V
  
  MPI_Barrier(MPI_COMM_WORLD);         // synchronization 
  if (rank==0) std::cout << "Container assembly is finished. Iteration starts." << std::endl;

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
          Koperator.apply(V+j*loc_N, Kv);   // K*H -> K*h (single vector operator)
	  Coperator.apply(V+j*loc_N, Cv);   // C*H -> C*h (single vector operator)
	  Moperator.apply(V+j*loc_N, Mv);   // M*H -> M*h (single vector operator)

          Kmat(i,j) = complex_dot(loc_N, V+i*loc_N, Kv);  // entry of K 
          Cmat(i,j) = complex_dot(loc_N, V+i*loc_N, Cv);  // entry of C
          Mmat(i,j) = complex_dot(loc_N, V+i*loc_N, Mv);  // entry of M
       }
    MPI_Barrier(MPI_COMM_WORLD);  // synchronization

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

    MPI_Barrier(MPI_COMM_WORLD);  // synchronization

    /* most promising Ritz vector (projected back) */
    init(loc_N, vbest, 0.0);
    for (int i=0; i<Vdim; i++) axpby(loc_N, cbest[i], V+i*loc_N, 1.0, vbest); 

    /* residual */
    Coperator.apply(vbest, Cv);
    Moperator.apply(vbest, Mv);
    Koperator.apply(vbest, res);
    axpby(loc_N, thetaBest, Cv, 1.0, res);
    axpby(loc_N, thetaBest*thetaBest, Mv, 1.0, res);
    resNorm = std::sqrt(complex_dot(loc_N, res, res).real());

    if (verbose==1) {if (rank==0) std::cout << "iteration " << iter << ", residual norm: " << resNorm;}
    result_ptr->cvg_hist[iter] = resNorm;

    /* if converge... */
    if (resNorm < tolerance)
    {
       result_ptr->eigval[detected] = thetaBest;
       vec_update(loc_N, 1.0, vbest, result_ptr->eigvec+detected*loc_N);
       vec_update(loc_N, 1.0, vbest, V+detected*loc_N);
       sigma = thetaBest;

       /* reduce the search space if necessary? */ 

       thetaBest = proj_eigval[detected+1];
       idxBest = idx_eigen[detected+1];
       init(loc_N, vbest, 0.0);
       for (int i=0; i<Vdim; i++) cbest[i] = linearEigSolv.eigenvectors().col(idxBest)[i+Vdim];
       for (int i=0; i<Vdim; i++) axpby(loc_N, cbest[i], V+i*loc_N, 1.0, vbest); 
     
       Coperator.apply(vbest, Cv);
       Moperator.apply(vbest, Mv);
       Koperator.apply(vbest, res);
       axpby(loc_N, thetaBest, Cv, 1.0, res);
       axpby(loc_N, thetaBest*thetaBest, Mv, 1.0, res);

       detected += 1;
    }

    /* reduce the search space if necessary TODO...*/
    if (Vdim == maxdim)
    {	    
       for (int j=0; j<mindim; j++)
       {
	  init(loc_N, v+j*loc_N, 0.0);     
          for (int i=0; i<Vdim; i++) cbest[i] = linearEigSolv.eigenvectors().col(idx_eigen[j])[i+Vdim];
          for (int i=0; i<Vdim; i++) axpby(loc_N, cbest[i], V+i*loc_N, 1.0, v+j*loc_N);  
       }
       vec_update(loc_N*mindim, 1.0, v, V);
       if (detected != 0) vec_update(loc_N*detected, 1.0, result_ptr->eigvec, V);
       Vdim = mindim;
    }

    MPI_Barrier(MPI_COMM_WORLD);  // synchronization

    /* solve the (preconditioned) correction equation */
    Coperator.apply(vbest, z);
    axpby(loc_N, 2.0*thetaBest, Mv, 1.0, z); 
    QRes::CorrectOp<std::complex<double>> correctOp(loc_N, Koperator, Coperator, Moperator, vbest, z, thetaBest);  // linear operator of the correction equation
     
    init(loc_N, t, 0.0);      // initial guess t0=zeros(N,1)
    vec_update(loc_N, -1.0, res, b);  // rhs b=-r
    gmres_solver(&correctOp, loc_N, t, b, tol_gmres, maxiter_gmres, &resNorm_gmres, &numIter_gmres, verbose_gmres);
    if (verbose==1){ if (rank==0) std::cout << ", residual norm of gmres: "<< resNorm_gmres << std::endl;}

    /* expand the search space V */
    vec_update(loc_N, 1.0, t, V+loc_N*Vdim);   // expand the search space V
    modifiedGS(V, loc_N, Vdim+1);          // modified Gram-Schmidt orthogonalization

    Vdim += 1;   // search space dimension +1
    iter += 1;   // iteration +1
  }

  free(KR); free(CR); 
  free(Kr); free(Cr); 
  free(Vp); free(a0); free(res);
  free(V); free(Kv); free(Cv); free(Mv); free(vbest);
  free(z); free(t); free(b);
 
  return result_ptr;
}
