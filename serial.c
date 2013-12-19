#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <sys/time.h>
//#include <mpi.h>
int n, p=1;
int main(int argc, char **argv) {
 int myn, myrank=0;
 double *a, *b, *c, *allB, start, sum, *allC, sumdiag,global_sum=0.0;
 int i, j, k;
 n = atoi(argv[1]);
 struct timeval tv1, tv2;
 struct timezone tz;
 
 //MPI_Init(&argc, &argv);
 //MPI_Comm_size(MPI_COMM_WORLD,&p);
 //MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
 myn = n/p;
 a = malloc(myn*n*sizeof(double));
 b = malloc(myn*n*sizeof(double));
 c = malloc(myn*n*sizeof(double));
 allB = malloc(n*n*sizeof(double));
 
 for(i=0; i<myn*n; i++) {
 a[i] = 1.;
 b[i] = 2.;
 }
 
 gettimeofday(&tv1, &tz);
 
 for(i=0; i<p; i++)
 //MPI_Gather(b, myn*n, MPI_DOUBLE, allB, myn*n, MPI_DOUBLE,i, MPI_COMM_WORLD);
 for(i=0; i<myn; i++)
 for(j=0; j<n; j++) {
 sum = 0.;
 for(k=0; k<n; k++)
 sum += a[i*n+k]*b[k*n+j];
 c[i*n+j] = sum;
 }
 
 for(i=0, sumdiag=0.; i<n; i++){
 sumdiag += c[i*n+i];
 }
 
 gettimeofday(&tv2, &tz);
 double elapsed = (double) (tv2.tv_sec-tv1.tv_sec) + (double) (tv2.tv_usec-tv1.tv_usec) * 1.e-6;
 
 printf("It took %lf seconds to multiply 2 %dx%d matrices.\n",
 elapsed, n, n);
 
 if(myrank==0) {
    printf("The trace of the resulting matrix is %f\n", sumdiag);
 }
 
 free(a);
 free(b);
 free(c);
}
