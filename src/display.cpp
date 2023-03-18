#include <iostream>
#include <string>
#include "header.hpp"

void MatrixShow(int x, int y, int n, double *Matrix )
{
  try
  {
  if (n == x*y)
  {
    std::cout<<"Matrix="<< std::endl;
    for (int i=0; i<x; i++)
      {
      for (int j=0; j<y; j++)
        std::cout << Matrix[i*x+j]<< "   ";
      std::cout << "\n";
      } 
  }
  else
    throw x*y;
  }
  catch (int l)
  {
    std::cout<< "dimension of matrix doesn't match x,y\n";
    std::cout<< "x*y = " << l << std::endl;
  }
  return;
}

