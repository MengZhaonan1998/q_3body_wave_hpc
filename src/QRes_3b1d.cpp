#include "mpi.h"
#include "header.hpp"
#include "timer.hpp"
#include "tensor3b1d.hpp"

std::map<std::string, std::string> readJDopts()
{
  std::map<std::string, std::string> jdopts;

  std::string line;
  std::vector<std::string> v;
  std::ifstream fin;
  fin.open("jacobidavidsonOpts.txt");

  if (fin.is_open())
  {
    while (std::getline(fin, line))
    {
      std::stringstream ss(line);
      while (std::getline(ss, line, ':'))
      {
        v.push_back(line);
      }
    }
  }

  for (int i=0; i<v.size(); i++) 
    if (i%2==0) jdopts[v[i]] = v[i+1];
   
  return jdopts;
}	


int main(int argc, char* argv[])
{
  if (argc < 5){
    // number of grid points (+1): nR, nr
    // length of domain [-L,L]: LR, Lr
    fprintf(stderr, "Usage %s <nR> <nr> <LR> <Lr>",argv[0]);
    return 1;
  }

  std::cout << "Welcome to QRes_3b1d!\n" 
	    << "This program aims at computing resonances of a quantum three body problem in parallel.\n" << std::endl;
 
  int nR = atoi(argv[1]);  // x grid size
  int nr = atoi(argv[2]);  // y grid size
  if (nR <= 2 || nr <= 2) throw std::runtime_error("need at least two grid points in each dimension to implement boundary conditions");
  double LR = atoi(argv[3]);
  double Lr = atoi(argv[4]);

  int proc_rank;               // processor rank
  int proc_numb;               // number of processors
  MPI_Init(&argc, &argv);      // initialize mpi
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank); // get processor rank
  MPI_Comm_size(MPI_COMM_WORLD, &proc_numb); // get processor number

  //--- read options of Jacobi-Davidson algorithm ---//
  auto jdopts = readJDopts(); 
  std::cout << "Options of the JD algorithm are as follows:" << std::endl;
  for (const auto& [key, value] : jdopts)
	  std::cout<< key << "=" << value << std::endl;

  //--- Jacobi-Davidson eigensolver ---//
  auto result = JacobiDavidson(proc_numb, proc_rank, jdopts); 

  //--- display the result including eigenvalues... ---//
  std::cout << "Eigenvalues detected:" << std::endl;
  for (int i=0; i<stoi(jdopts["numeigs"]); i++) std::cout<<result->eigval[i]<<std::endl;
 
  /* 
  // build hamiltonian operator 1d
  double* D2_R = (double*)malloc((nR+1)*(nR+1) * sizeof(double));
  double* D2_r = (double*)malloc((nr+1)*(nr+1) * sizeof(double));
 // double* V = (double*)malloc((nR+1)*(nr+1) * sizeof(double));
  ChebyshevDiffMatrix(nR, LR, D2_R);
  ChebyshevDiffMatrix2(nr, Lr, D2_r);

 // double *V = new double[(nR+1)*(nr+1)]; 
//  V = buildGaussianPotential3b1d(nR, nr, LR, Lr, 1.0, 1.0, 1.0);  
  double* VGauss = (double*)malloc((nR+1)*(nr+1)*sizeof(double));
  VGauss = buildGaussianPotential3b1d(nR, nr, LR, Lr, 1.0, 1.0, 1.0); 
  free(VGauss);
  // -------------------------------------------------------------------------------------//
  // todo... need a function to build an Hamiltonian operator! or tensor product operator.
  // what's more, unify the memory allocation -> malloc instead of new
  // -------------------------------------------------------------------------------------//


  double D1[4]={1,2,3,4};  // Chebyshev diff mat x
  double D2[9]={1,1,1,3,4,5,7,7,8};  // Chebyshev diff mat y
  double V[6]={1,2,3,6,3,2};   // Potential matrix/or vector

  double v_in[6]={1,1,1,1,1,1};
  double v_out[6];
   
  QRes::Kron2D<double> Koperator(2, D1, 3, D2, V, 1.0, 1.0);
  Koperator.apply(v_in, v_out);

  std::cout << "finish!\n" << std::endl;
 
 // free(results->eigval);
 // free(results->eigvec);
 // free(results->cvg_hist);
 // free(results);
  free(D2_R);
  free(D2_r);
  */
  

  MPI_Finalize();
  return 0;
}
