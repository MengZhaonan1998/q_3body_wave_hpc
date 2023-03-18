void MatrixShow(int x, int y, int n, double *Matrix );
void ChebyshevDiffMatrix(int n, double *ChebD1);
void ChebyshevDiffMatrix2(int n,double *ChebD2);

typedef struct s_SparseMatrix {
	int *columnInd;		//index of (zero)+non-zero elements due to tensorproduct (first row, second row,...)
	__float128 *value;	//values of (zero)+non-zero elements due to tensorproduct (first row, second row,...)
	int nz;			//number of (zero)+non zero elements per row (has to be the same for every row)
	int dim;		//dimension of the matrix
} SparseMatrix;




/*typedef struct ChebyshevDiffMatrix
{
  int n; // number of grid points;
  inline double entry(int i, int j) const
  {
    return i+j;
  }
	
} ChebyshevDiffMatirx;*/
