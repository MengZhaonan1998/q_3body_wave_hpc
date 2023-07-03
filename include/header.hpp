#pragma once 
#include <cstdlib>
#include <stdexcept>
#include <memory>
#include <string>
#include <map>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <complex>
#include <cmath>
#include <omp.h>
#include <Eigen/Eigenvalues>
#include <numeric>
#include <mpi.h>
//#include <mkl.h> 


typedef struct r_resultJD {
  std::complex<double> *eigval;
  std::complex<double> *eigvec;
  double *cvg_hist;
} resultJD;

std::map<std::string, std::string> readopts();
void ChebyshevDiffMatrix(int n, double L, double *ChebD1);
void ChebyshevDiffMatrix2(int n, double L, double *ChebD2);
void buildGaussianPotential3b1d(int nR, int nr, double LR, double Lr, double V12, double V13, double V23, std::complex<double>* VGauss); 
void buildKmatrix(int n, double L, std::complex<double> *K);
void buildCmatrix(int n, std::complex<double>* C);
void buildMmatrix(int n, std::complex<double>* M);
void buildCmatrix_complex(int n, std::complex<double>* C);

std::unique_ptr<resultJD> JacobiDavidson(std::map<std::string,std::string> b3d1opts, std::map<std::string,std::string> jdopts, std::map<std::string, std::string> gmresopts);
void modifiedGS(std::complex<double>* V, int m, int n);


//resultJD quadJacobiDavidson(int i);




/*
void MatrixShow(int x, int y, int n, double *Matrix );
void ChebyshevDiffMatrix(int n, double L, double *ChebD1);
void ChebyshevDiffMatrix2(int n,double L, double *ChebD2);


typedef struct s_SparseMatrix {
	int *columnInd;		//index of (zero)+non-zero elements due to tensorproduct (first row, second row,...)
	double *value;	//values of (zero)+non-zero elements due to tensorproduct (first row, second row,...)
	int nz;			//number of (zero)+non zero elements per row (has to be the same for every row)
	int dim;		//dimension of the matrix
} SparseMatrix;


SparseMatrix *buildGaussianPotential2B1D(int n, double L, double v0);

// assemble the operator
double *buildDenseHamiltonian2B1D(int n, double L, SparseMatrix *Vsp);


// delete
void DeleteSparseMatrix(SparseMatrix *Sp);

*/
