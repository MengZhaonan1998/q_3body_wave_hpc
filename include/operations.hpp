#pragma once
#include <sstream>
#include "header.hpp"
#include <stdexcept>

//////////////////////////////////
// Vector operations            //
//////////////////////////////////

// initialize a vector with a constant value, x[i] = value for 0<=i<n
void init(int n, double* x, double value);

// initialize a complex vector with a complex number, x[i] = value for 0<=i<n
void complex_init(int n, std::complex<double>* x, std::complex<double> value);

// scalar product: return sum_i x[i]*y[i] for 0<=i<n
std::complex<double> dot(int n, std::complex<double> const* x, std::complex<double> const* y);

// complex scalar product: return  sum_i conj(x[i])*y[i] for 0<=i<n
std::complex<double> complex_dot(int n, std::complex<double> const* x, std::complex<double> const* y);

// vector update: compute y[i] = a*x[i] + b*y[i] for 0<=i<n
void axpby(int n, double a, double const* x, double b, double* y);

// complex vector update: compute y[i] = a*x[i] + b*y[i] for 0<=i<n
void complex_axpby(int n, std::complex<double> a, std::complex<double> const* x, std::complex<double> b, std::complex<double>* y);

// complex vector update: compute y[i] = a*x[i] for 0<=i<n
void vec_update(int n, std::complex<double> a, std::complex<double> const* x, std::complex<double>* y);


