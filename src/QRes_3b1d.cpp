#include "mpi.h"
#include "header.hpp"
#include "timer.hpp"
#include "tensor3b1d.hpp"

/* readopts()
 * used to read options for the JacobiDavidson algorithm and GMRES solver */
std::map<std::string, std::string> readopts(std::string filename)
{
  std::map<std::string, std::string> opts;

  std::string line;
  std::vector<std::string> v;
  std::ifstream fin;
  fin.open(filename);

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
    if (i%2==0) opts[v[i]] = v[i+1];
   
  return opts;
}	


int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);  // initialize mpi
  int proc_rank;           // processor rank
  int proc_numb;           // processor number
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank); // get processor rank
  MPI_Comm_size(MPI_COMM_WORLD, &proc_numb); // get processor number

  if (argc < 5){
    // number of grid points (+1): nR, nr
    // length of domain [-L,L]: LR, Lr
    if (proc_rank==0) fprintf(stderr, "Usage %s <nR> <nr> <LR> <Lr>",argv[0]);
    return 1;
  }

  if (proc_rank==0)
  std::cout << "Welcome to QRes_3b1d!\n" 
	    << "              -----------                      \n"
	    << "---------     \\        /              ------- \n"
	    << "\\      /-------\\      /---------------\\    / \n"             
            << " \\    /---------\\    /-----------------\\  /  \n"
	    << "  \\  /           \\  /                   \\/   \n"
	    << "   \\/             \\/                          \n"
	    << "This program aims at computing resonances of a quantum three body problem in parallel.\n" << std::endl;
 
  int nR = atoi(argv[1]);    // x grid size
  int nr = atoi(argv[2]);    // y grid size
  double LR = atoi(argv[3]); // x domain length
  double Lr = atoi(argv[4]); // y domain length
  if (nR <= 2 || nr <= 2) throw std::runtime_error("need at least two grid points in each dimension to implement boundary conditions");

  //--- read options of Jacobi-Davidson algorithm ---//
  auto jdopts = readopts("jacobidavidsonOpts.txt"); 
  std::cout << "Options of the Jacobi-Davidson algorithm are as follows:" << std::endl;
  for (const auto& [key, value] : jdopts)
	  std::cout<< key << "=" << value << std::endl;

  //--- read options of GMRES algorithm ---//
  auto gmresopts = readopts("gmresOpts.txt"); 
  std::cout << "Options of the GMRES algorithm are as follows:" << std::endl;
  for (const auto& [key, value] : gmresopts)
	  std::cout<< key << "=" << value << std::endl;

  //--- Jacobi-Davidson eigensolver ---//
  auto result = JacobiDavidson(nR, nr, LR, Lr, jdopts, gmresopts);    // quadratic jacobi-davidson algorithm
  
  MPI_Finalize();                          // finish mpi
  
  //--- display the result including eigenvalues... ---//
  std::cout << "Eigenvalues detected:" << std::endl;
  for (int i=0; i<stoi(jdopts["numeigs"]); i++) std::cout<<result->eigval[i]<<std::endl;
 
  return 0;
}
