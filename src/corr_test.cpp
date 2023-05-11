
#include "header.hpp"
#include "gtest_mpi.hpp"
#include "operations.hpp"
#include "tensor3b1d.hpp"
#include "gmres_solver.hpp"

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

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

  for (int i=0; i<size; i++)
  {if (rank==i){
     std::cout<<"------rank "<<rank << "--------"<<std::endl;
     for (int i=0;i<loc_len;i++) std::cout<<v_out[i]<<std::endl;
     std::cout<<"----------------------------------"<<std::endl;
  }}


  std::complex<double> result[N] = {6.6834, -0.0229, -0.0229, 6.6834, 0.1263, -6.7868, -6.7868, 0.1263,
                       	            0.1263, -6.7868, -6.7868, 0.1263, 6.6834, -0.0229, -0.0229, 6.6834};

  MPI_Finalize();
}

