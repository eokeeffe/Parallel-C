/*
ATLAS.c
question1-c.c     brett becker    12/09/2005
performs a 1000x1000 matrix matrix multiplication using the ATLAS library when compiled with the following line:

gcc -o ATLAS ATLAS.c -I/home/bbecker/local/ATLAS/include/ -L/home/bbecker/local/ATLAS/lib/Linux_P4SSE2/ -lcblas -latlas -lm -O3
*/

#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <math.h>

void initMat( int M, int N, double mat[], double val )
{
    int     i, j, k;
    for (i= 0; i< M; i++){
        for (j= 0; j< N; j++){
            mat[i*N+j] = val;
        }
    }
}

int main(int argc,char **argv)
{
    double *A, *B, *C;
    int N = 100,numreps = 1;
    
    sscanf(argv[1],"%d",&N);
    sscanf(argv[2],"%d",&numreps);
    
    //printf ("Please enter matrix dimension n : ");scanf("%d", &N);
    int i;

    //allocate and initialize arrays
    A = malloc (N*N*sizeof(double));
    B = malloc (N*N*sizeof(double));
    C = malloc (N*N*sizeof(double));
    if (!A  ||  !B ||   !C)
    {
        printf( "Out of memory, reduce N value.\n");
        exit(-1);
    }
    initMat(N, N, A, rand());
    initMat(N, N, B, rand());

    //multiply
    printf("Multiply matrices %d times...\n", numreps);
    for (i=0; i<numreps; i++)
    {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, B, N, 0.0, C, N);
    }
    printf("Done ...\n");

    free(A);
    free(B);
    free(C);

    return 0;
}