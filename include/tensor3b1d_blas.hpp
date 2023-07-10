#pragma once
#include "header.hpp"
#include "operations.hpp"
#include "cblas.h"

namespace QResblas
{

  template<typename ST>
  class Kron2D{
    public:
      Kron2D(int m, ST* C_m, 
	     int n, ST* D_n,
	     ST* V, double a1, double a2):
             C_(C_m), D_(D_n), V_(V), a1_(a1), a2_(a2), n_(n), m_(m)
      {
        if (std::is_same<ST, float>::value)
	   mpi_type = MPI_FLOAT;
	else if (std::is_same<ST, double>::value)
	   mpi_type = MPI_DOUBLE;	
	else if (std::is_same<ST, std::complex<float>>::value)
	   mpi_type = MPI_C_COMPLEX;
        else if (std::is_same<ST, std::complex<double>>::value)
	   mpi_type = MPI_C_DOUBLE_COMPLEX;	
      
        loc_m = domain_decomp(m_);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//	MPI_Type_vector(loc_m, 1, n_, mpi_type, &column_type);
//	MPI_Type_commit(&column_type);
	
      };

      // reshape operator
      // y = Op*x = reshape(a1*C*X+a2*X*D', n*m,1)
      void apply(const ST* v_in, ST* v_out) // to do...
      {

	int i,j,k;     
	ST ele;

	// V*w 
        #pragma omp parallel for
	for (i=0; i<loc_m*n_; i++) v_out[i] = V_[i] * v_in[i]; 	

	/*
	// a2*C2*W
	for (i=0; i<m_; i++) 
          for (j=0; j<n_; j++) 
            for (k=0; k<n_; k++) 
	      v_out[i*n_+j] += a2_ * D_[j*n_+k] * v_in[i*n_+k];
	      // v_out[i*n_+j] += a2_ * dot(n_, D_+j*n_+k, v_in+i*n_+k);
	
	// a1*W*C1^T
	for (i=0; i<m_; i++)
          for (j=0; j<n_; j++)
	    for (k=0; k<m_; k++)
	      v_out[i*n_+j] += a1_ * v_in[k*n_+j] * C_[i*m_+k];    
              // axpby(n_, a1_ * C_[i*m_+k], v_in+k*n_, 1.0, v_out+i*n_); 
	*/
   
        // non-blocking allgather operation 
        ST* vcol = new ST[m_*n_];
        init(m_, vcol, 0.0);
	MPI_Request request;
        MPI_Iallgather(v_in, loc_m*n_, mpi_type, vcol, loc_m*n_, mpi_type, MPI_COMM_WORLD, &request);

    	cblas_zgemm(101, 111, 112, loc_m, n_, n_, a2_, v_in, loc_m, D_, n_, 1.0, v_out, loc_m);

	MPI_Wait(&request, MPI_STATUS_IGNORE);

	cblas_zgemm(101, 111, 111, loc_m, n_, m_, a1_, C_+(rank*loc_m)*m_, loc_m, v_in, m_, 1.0, v_out, loc_m);

	delete [] vcol;
        return;
      }

      // approximate inverse operation, y = Op\b
      // The operation is intended as a preconditioner and may not give the exact solution.
      void invapply(const ST* b, ST* v_out)
      {
        int i, j, k;
      }

    private:
      int n_, m_, loc_m, rank;
      ST a1_, a2_;
      ST* C_;
      ST* D_;
      ST* V_;
      MPI_Datatype mpi_type;
      //MPI_Datatype column_type;
  };

}


