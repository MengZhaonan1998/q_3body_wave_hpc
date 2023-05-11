#pragma once
#include "header.hpp"
#include "operations.hpp"

namespace QRes
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
	//for (i=0; i<m_*n_; i++) v_out[i] = V_[i] * v_in[i]; 	
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

        #pragma omp declare reduction(+: ST: omp_out += omp_in) initializer (omp_priv = omp_orig)	
        #pragma omp parallel for reduction(+:ele)
        for (i=0; i<loc_m; i++)
          for (j=0; j<n_; j++)
	    for (k=0; k<n_; k++)
	    {
	       ele =  a2_ * v_in[i*n_+k] * D_[j*n_+k];
               v_out[i*n_+j] += ele;
	    }

	MPI_Wait(&request, MPI_STATUS_IGNORE);
        //for (k=0; k<m_*n_; k++) std::cout << "vcol["<< k<<"]="<<vcol[k]<<std::endl;

        #pragma omp parallel for reduction(+:ele)
	for (i=0; i<loc_m; i++)
          for (j=0; j<n_; j++)
	    for (k=0; k<m_; k++)
	    {
              ele = a1_ * vcol[k*n_+j] * C_[(rank*loc_m+i)*m_+k]; 
	      v_out[i*n_+j] += ele;    
	    }

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

  template<typename ST>
  class CorrectOp{
    public:
      CorrectOp(int N, 
		QRes::Kron2D<ST> Koperator, 
		QRes::Kron2D<ST> Coperator,
		QRes::Kron2D<ST> Moperator,
		ST* u, ST* w, ST theta):
	  N_(N), Kop_(Koperator), Cop_(Coperator), Mop_(Moperator),
	  u_(u), w_(w), theta_(theta){};
      int N_; 
      void apply(const ST* v_in, ST* v_out)
      {
        int rank, size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        std::complex<double>* vtemp1 = (std::complex<double>*)malloc(sizeof(std::complex<double>)*N_);
	std::complex<double>* vtemp2 = (std::complex<double>*)malloc(sizeof(std::complex<double>)*N_);
	std::complex<double> dux,duy,duw;
	dux = complex_dot(N_, u_, v_in);            // u'*v_in

	vec_update(N_, 1.0, v_in, vtemp1);          // vtemp = v_in
	axpby(N_, -dux, u_, 1.0, vtemp1);   // vtemp = v_out - dux*u
      
	// v_temp2 = tensorapply(K,vtemp1) + 
	//   theta*tensorapply(C,vtemp1) + 
	//   theta*theta*tensorapply(M,vtemp1) 
	Kop_.apply(vtemp1, v_out);
        vec_update(N_, 1.0, v_out, vtemp2);
	
	Cop_.apply(vtemp1, v_out);    // Problem!
	axpby(N_, theta_, v_out, 1.0, vtemp2);

	Mop_.apply(vtemp1, v_out);
	axpby(N_, theta_*theta_, v_out, 1.0, vtemp2);

	duy = complex_dot(N_, u_, vtemp2);        // duy = u'*v_out
	duw = complex_dot(N_, u_, w_);            // duw = u'*w
        axpby(N_, -duy/duw, w_, 1.0, vtemp2);     // vtemp2 = vtemp2 - (duy/duw)*w

        vec_update(N_, 1.0, vtemp2, v_out);
    
	/*
        for (int i=0; i<size; i++)
        {if (rank==i){
           std::cout<<"------rank "<<rank << "--------"<<std::endl;
           for (int i=0;i<N_;i++)std::cout<<v_out[i]<<std::endl;
           std::cout<<"----------------------------------"<<std::endl;
        }}*/

	free(vtemp1); free(vtemp2);
	return;
      }

    private:
      QRes::Kron2D<ST> Kop_;
      QRes::Kron2D<ST> Cop_;
      QRes::Kron2D<ST> Mop_;
      ST* u_;
      ST* w_;
      ST theta_;
  };

}


