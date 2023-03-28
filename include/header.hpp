
typedef struct r_resultJD {
  double *eigval;
  double *eigvec;
  double *cvg_hist;
} resultJD;

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