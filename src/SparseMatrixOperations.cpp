#include "header.hpp"

void DeleteSparseMatrix(SparseMatrix *Sp){
        delete [] Sp->value;
	delete [] Sp->columnInd;
	delete [] Sp;
//	free((Sp->columnInd));
//	free(Sp);
}

