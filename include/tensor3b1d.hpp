#pragma once

namespace QRes
{

  template<typename ST>
  class Kron2D{
    public:
      Kron2D(int m, ST* C_n, 
	     int n, ST* D_m,
	     ST* V, ST a1, ST a2):
             C_(C_n), D_(D_m), V_(V), a1_(a1), a2_(a2), n_(n), m_(m){};

      void apply() // to do...
      { std::cout << "tensor apply operator succeed!\n" << std::endl;
	      return;}

    private:
      int n_, m_;
      ST a1_, a2_;
      ST* C_;
      ST* D_;
      ST* V_;
  };

}
