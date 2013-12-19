#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <mpi.h>
int n, p;
int main(int argc, char **argv) {
 int myn, myrank;
 double *a, *b, *c, *allB, start, sum, *allC, sumdiag,global_sum=0.0;
 int i, j, k;
 n = atoi(argv[1]);
 MPI_Init(&argc, &argv);
 MPI_Comm_size(MPI_COMM_WORLD,&p);
 MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
 myn = n/p;
 a = malloc(myn*n*sizeof(double));
 b = malloc(myn*n*sizeof(double));
 c = malloc(myn*n*sizeof(double));
 allB = malloc(n*n*sizeof(double));
 for(i=0; i<myn*n; i++) {
 a[i] = 1.;
 b[i] = 2.;
 }
 MPI_Barrier(MPI_COMM_WORLD);
 if(myrank==0)
 start = MPI_Wtime();
 for(i=0; i<p; i++)
 MPI_Gather(b, myn*n, MPI_DOUBLE, allB, myn*n, MPI_DOUBLE,
 i, MPI_COMM_WORLD);
 for(i=0; i<myn; i++)
 for(j=0; j<n; j++) {
 sum = 0.;
 for(k=0; k<n; k++)
 sum += a[i*n+k]*allB[k*n+j];
 c[i*n+j] = sum;
 }
 free(allB);
 
 for (i = 0,sumdiag=0.; i < myn; i++){
            sumdiag += c[i * n  + i];// wasn't needed + myrank * p
    }
    printf("The trace of %i matrix is %f\n",myrank, sumdiag);//for debugging 
 
 MPI_Barrier(MPI_COMM_WORLD);
 if(myrank==0)
 printf("It took %f seconds to multiply 2 %dx%d matrices.\n",
 MPI_Wtime()-start, n, n);
 
 MPI_Reduce(&sumdiag, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);
 
 if(myrank==0) {
    printf("The trace of the resulting matrix is %f\n", global_sum);
 }
 
 /*
 if(myrank==0)
 allC = malloc(n*n*sizeof(double));
 MPI_Gather(c, myn*n, MPI_DOUBLE, allC, myn*n, MPI_DOUBLE,
0, MPI_COMM_WORLD);
 if(myrank==0) {
 for(i=0, sumdiag=0.; i<n; i++)
 sumdiag += allC[i*n+i];
 printf("The trace of the resulting matrix is %f\n", sumdiag);
 }
 if(myrank==0)
 free(allC);
 */
 MPI_Finalize(); free(a);
 free(b);
 free(c);
}