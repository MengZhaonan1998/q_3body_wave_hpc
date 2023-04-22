#pragma once

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

}
