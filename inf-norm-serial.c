/*
    Row wise slicing
    Matrix Multiplication & Inifinite Normal
    
    How to compile: gcc -o program program.c
*/
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#define DONT_INCREMENT 0
#define INCREMENT 1
double   **A,**B,**C, NormRow = 0 ;
int      N;

/* Thread callback function */
void Mulitply_AND_INF_Norm()
{
    int i,j,k,current_row,column;
    double sum,mynorm=0.0,myRowSum=0.0;
    
    // matrix mult over the strip of rows for this thread
    for(i=0; i < N ;i++)
    {
        for(j=0;j < N;j++)
        {
            sum = 0.0;
            for(k=0;k < N;k++)
            {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    
    for (current_row = 0; current_row < N; current_row++)
    {
        myRowSum = 0.0;
        for(column = 0 ;column < N; column++)
        {
            myRowSum += C[current_row][column];
        }
        
        if(mynorm < myRowSum )
        {
            mynorm = myRowSum;
        }
    }
    
    if (NormRow < mynorm)
    {
        NormRow = mynorm;
    }
}

void printMatrix(double **mat) 
{
    int i,j;
    fprintf(stdout,"\n%d*%d\n",N,N);
    for(i=0; i < N; i++)
    {
        for(j=0; j < N; j++)
        {
            fprintf(stdout,"%.lf ", mat[i][j]);
        }
        fprintf(stdout,"\n");
    }
}

void initMat(int M, int N, double **mat)
{//put basic incremented values into the matrix , val=1
    int i, j;
    double val = 1.0;
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
        {
            mat[i][j] = val;
        }
    }
}

void cleanMat( int M, int N, double **mat)
{//set the array values to 0
    int i, j, k;
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
        {
            mat[i][j] = 0;
        }
    }
}

double** allocate2dArray(int rows,int columns)
{//return a 2d double array
    int i;
    double **ptr = malloc(rows * sizeof(double *));
    for(i = 0; i < rows; i++)
    {
        ptr[i] = malloc(columns * sizeof(double));
    }
    return ptr;
}

void deallocate2dArray(int rows,int columns,double **ptr)
{//clean up the allocations to the given pointer
    int i;
    for(i = 0; i < rows; i++)
    {
        free(ptr[i]);
    }
    free(ptr);
}

/* Main function starts*/
main(int argc, char *argv[])
{
    /* variable declaration */;
    double memoryused=0.0;
    struct timeval tv1, tv2;
    struct timezone tz;

    /*Argument validation*/
    if( argc < 2 )
    {
        fprintf(stderr,"\t\t Not enough Arguments\n ");
        fprintf(stderr,"\t\t usage : %s <N>\n",argv[0]);
        exit(-1);
    }
    else
    {
        sscanf(argv[1],"%d",&N);
    }

    fprintf(stdout,"\n\t\t Input Parameters:");
    fprintf(stdout,"\n\t\t Size of Matrix      :  %d",N);
    fprintf(stdout,"\n");

    /*  Allocate the memory for the matrices*/
  
    A = allocate2dArray(N,N);
    B = allocate2dArray(N,N);
    C = allocate2dArray(N,N);
    
    if(A == NULL||B == NULL||C == NULL)
    {
        fprintf(stderr,"\n Not sufficient memory to accomodate the size of %d * %d Matrix",N,N);
        exit(0);
    }
    
    // simply profiling approximate memory size
    memoryused = 3 * (N * N * sizeof(double));

    initMat(N,N,A);//set all of A to 1
    initMat(N,N,B);//set all of B to 1
    cleanMat(N,N,C);//set all of  C to 0
    
    /*Taking start time*/
    gettimeofday(&tv1, &tz);
    
    Mulitply_AND_INF_Norm();
    
    gettimeofday(&tv2, &tz);
    double elapsed = (double) (tv2.tv_sec-tv1.tv_sec) + (double) (tv2.tv_usec-tv1.tv_usec) * 1.e-6;
    
    fprintf(stdout,"\n\t\t Multiplication and Infinity Norm of a Square Matrix has completed ");

    //printMatrix(A);
    //printMatrix(B);
    //printMatrix(C);
    
    deallocate2dArray(N,N,A);
    deallocate2dArray(N,N,B);
    deallocate2dArray(N,N,C);
    
    fprintf(stdout,"\n");
    fprintf(stdout,"\n\t\t INF-Norm Answer       :  %lf",NormRow);
    fprintf(stdout,"\n\t\t Memory Used           :  %lf MB",(memoryused/(1024*1024)));
    fprintf(stdout,"\n\t\t Time in  Seconds (T)  :  %lfs",elapsed);
    fprintf(stdout,"\n\t\t..........................................................................\n");
}