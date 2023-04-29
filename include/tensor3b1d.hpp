#pragma once
#include "operations.hpp"

namespace QRes
{

  template<typename ST>
  class Kron2D{
    public:
      Kron2D(int m, ST* C_m, 
	     int n, ST* D_n,
	     ST* V, double a1, double a2):
             C_(C_m), D_(D_n), V_(V), a1_(a1), a2_(a2), n_(n), m_(m){};

      // reshape operator
      // y = Op*x = reshape(a1*C*X+a2*X*D', n*m,1)
      void apply(const ST* v_in, ST* v_out) // to do...
      {

	int i,j,k; 
	// V*w 
	for (i=0; i<m_*n_; i++) v_out[i] = V_[i] * v_in[i]; 
	
	// a2*C2*W
	for (i=0; i<m_; i++) 
          for (j=0; j<n_; j++) 
            for (k=0; k<n_; k++) 
	      v_out[i*n_+j] += a2_ * D_[j*n_+k] * v_in[i*n_+k];
       
        // a1*W*C1^T
	for (i=0; i<m_; i++)
          for (j=0; j<n_; j++)
	    for (k=0; k<m_; k++)
	      v_out[i*n_+j] += a1_ * v_in[k*n_+j] * C_[i*m_+k];   

        return;
      }

    private:
      int n_, m_;
      ST a1_, a2_;
      ST* C_;
      ST* D_;
      ST* V_;
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
        std::complex<double>* vtemp1 = (std::complex<double>*)malloc(sizeof(std::complex<double>)*N_);
	std::complex<double>* vtemp2 = (std::complex<double>*)malloc(sizeof(std::complex<double>)*N_);
	std::complex<double> dux,duy,duw;

	dux = complex_dot(N_, u_, v_in);            // u'*v_in
	vec_update(N_, 1.0, v_in, vtemp1);          // vtemp = v_in
	complex_axpby(N_, -dux, u_, 1.0, vtemp1);   // vtemp = v_out - dux*u
        
	/* v_out = tensorapply(K,vtemp1) + 
	 *   theta*tensorapply(C,vtemp1) + 
	 *   theta*theta*tensorapply(M,vtemp1) */
	Kop_.apply(vtemp1, v_out);                  
	Cop_.apply(vtemp1, vtemp2); 
	complex_axpby(N_, theta_, vtemp2, 1.0, v_out);
	Mop_.apply(vtemp1, vtemp2);
	complex_axpby(N_, theta_*theta_, vtemp2, 1.0, v_out);
	
	duy = complex_dot(N_, u_, v_out);            // duy = u'*v_out
	duw = complex_dot(N_, u_, w_);               // duw = u'*w
        complex_axpby(N_, -duy/duw, w_, 1.0, v_out); // v_out = v_out - (duy/duw)*w

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
