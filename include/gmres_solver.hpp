#pragma once

#include "tensor3b1d.hpp"
#include "operations.hpp"
#include "header.hpp"

// run GMRES iterations to solve the linear system
// op*x=b, where op is the 7-point stencil representation of a linear
// operator. The function returns if the 2-norm of the residual reaches
// tol, or the number of iterations reaches maxIter. The residual norm
// is returned in *resNorm, the number of iterations in *numIter.
// arnoldi function (gmres)                                       

void arnoldi(int k, std::complex<double>* Q, std::complex<double>* h, QRes::CorrectOp<std::complex<double>>* op); 

void given_rotation(int k, std::complex<double>* h, std::complex<double>* cs, std::complex<double>* sn);

void gmres_solver( QRes::CorrectOp<std::complex<double>>* op, int n, std::complex<double>* x, std::complex<double> const* b,
                   double tol, int maxIter, double* resNorm, int* numIter, int verbose=1);


