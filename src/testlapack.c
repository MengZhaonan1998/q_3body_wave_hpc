/* Calling DGELS using row-major order */
#include <stdio.h>

#include "lapacke.h"
#include "cblas.h"

#include <complex.h>

int main()
{
        const int dim=2;
        double a[4]={1.0,1.0,1.0,1.0},b[4]={2.0,2.0,2.0,2.0},c[4];
        int m=dim,n=dim,k=dim,lda=dim,ldb=dim,ldc=dim;
        double al=1.0,be=0.0;
        cblas_dgemm(101,111,111,m,n,k,al,a,lda,b,ldb,be,c,ldc);
        printf("the matrix c is:%f,%f\n%f,%f\n",c[0],c[1],c[2],c[3]);
        return 0;
}



//lapack_int test_ztrsyl();
//lapack_int test_dhseqr();

/*lapack_int test_dhseqr()
{
   lapack_int n, ilo, ihi, ldh, ldz;
   
}*/
/*
lapack_int test_ztrsyl()
{

   lapack_int info, isgn, m, n, lda, ldb, ldc;
    
   double a[4] = {1,1,2,3};
   double b[4] = {-10,-3,1,6};
   double c[4] = {1,4,-2,0};

   m=2; // rows of A
   n=2; // rows of B
   isgn=1; // sign
   
   lda=2; 
   ldb=2;
   ldc=2;

   double scale=1.0;
   double* pscale = &scale;

   info = LAPACKE_dtrsyl(LAPACK_ROW_MAJOR,'N','N',isgn, m, n, a, lda, b, ldb, c, ldc, pscale);
   printf("info=%d \n",info);

   for(int i=0;i<m;i++){
	for (int j=0;j<n;j++)
     	    printf("%f " , c[i*n+j]);		
   	printf("\n");}

   double ele;
   for(int i=0;i<m;i++)
   {	   
      for(int j=0;j<n;j++)
      {
	 ele=0.0;
         for(int k=0; k<m; k++)
	 {
	    ele += a[i*m+k] * c[k*n+j];	 
	 }
	 for(int k=0; k<n; k++)
	 {
	    ele += c[i*m+k] * b[k*n+j];
	 }

	 printf("%f ,", ele);
      }
      printf("\n");
   }
   
   return info;
}


int main (int argc, const char * argv[])
{
   // first try of LAPACK	
   double a[5][3] = {1,1,1,2,3,4,3,5,2,4,2,5,5,4,3};
   double b[5][2] = {-10,-3,12,14,14,12,16,16,18,16};
   lapack_int info,m,n,lda,ldb,nrhs;
   int i,j,k;

   m = 5;
   n = 3;
   nrhs = 2;
   lda = 3;
   ldb = 2;

   info = LAPACKE_dgels(LAPACK_ROW_MAJOR,'N',m,n,nrhs,*a,lda,*b,ldb);

   for(i=0;i<n;i++)
   {
      for(j=0;j<nrhs;j++)
      {
         printf("%lf ",b[i][j]);
      }
      printf("\n");
   }

   test_ztrsyl();
   
   // second try of LAPACK
    lapack_int LAPACKE_dgges( int matrix_order, char jobvsl, char jobvsr, char sort, 
    *                          LAPACK_D_SELECT3 select, lapack_int n, double* a, lapack_int lda, 
    *                          double* b, lapack_int ldb, lapack_int* sdim, double* alphar, 
    *                          double* alphai, double* beta, double* vsl, lapack_int ldvsl, 
    *                          double* vsr, lapack_int ldvsr );
    
   
   LAPACK_D_SELECT3 select;

   double A[4][4] = {1.0, 4.1, 0.9, 1.4,
	             6.8,-0.4,-3.8, 6.5,
		     4.8, 3.0, 7.2, 2.1,
		    -4.4, 1.5, 2.0, 6.0};

   double B[4][4] = {1.0, 0.0, 0.0, 0.0,
	             0.0, 1.0, 0.0, 0.0,
		     0.0, 0.0, 1.0, 0.0,
		     0.0, 0.0, 0.0, 1.0};
   double alphar[4];
   double alphai[4];
   double beta[4];
   double vsl[4][4];
   double vsr[4][4];

   info = LAPACKE_dgges(LAPACK_ROW_MAJOR, 'V', 'V', 'N', 
                        select, 4, *A, 4,
			*B, 4, 0, alphar,
			alphai, beta, *vsl, 4, 
			*vsr, 4);
   for(i=0;i<4;i++)
   {
      for(j=0;j<4;j++)
      {
	 printf("%lf ", A[i][j]); 
      }
      printf("\n");
   }

   return 0;
}

*/

