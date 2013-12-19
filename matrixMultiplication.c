#include<stdio.h>
#include<sys/time.h>
#include<stdlib.h>
//simple min macro
#define min(a, b) (((a) < (b)) ? (a) : (b))

void multiplyMatrixNonBlockedIJK(double **A,double **B,double **C,int N)
{//ijk non blocking matrix multiplication
    
    int i,j,k;
    double sum;
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
}

void multiplyMatrixBlockedIJK(double **A,double **B,double **C,int N,int block_size)
{//ijk blocked matrix multiplication
    int i,j,k,jblock,kblock;
    double sum;
    
    for(jblock=0;jblock<N;jblock+=block_size)
    {
        for(i=0;i<N;i++)
        {
            for(j=jblock;j< min(jblock + block_size,N); j++)
            {
                C[i][j] = 0.0;
            }
        }
        
        for(kblock=0;kblock<N;kblock+=block_size)
        {
            for(i=0;i<N;i++)
            {
                for(j=jblock;j< min(jblock+block_size,N);j++)
                {
                    sum = 0.0;
                    for(k=kblock;k< min(kblock+block_size,N);k++)
                    {
                        sum += A[i][k] * B[k][j];
                    }
                    C[i][j] += sum;
                }
            }
        }
    }
}

void multiplyMatrixBlockedKIJ(double **A,double **B,double **C,int N,int block_size)
{//blocked kij algorithm

    int i,j,k,jblock,kblock,iblock;
    double temp=0.0;
    
    for(kblock=0;kblock<N;kblock+=block_size)
    {
        for(iblock=0;iblock<N;iblock+=block_size)
        {
            for(jblock=0;jblock<N;jblock+=block_size)
            {
                for(k=kblock;k< min(kblock+block_size,N);k++)
                {
                    for(i=0;i<N;i++)
                    {
                        temp = A[i][k];
                        for(j=jblock;j< min(jblock+block_size,N);j++)
                        {
                            C[i][j] += temp * B[k][j];
                        }
                    }
                }
            }
        }
    }
}

void initMat( int M, int N, double **mat)
{//put basic incremented values into the matrix , val=1
    int i, j;
    double val = 1.0;
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
        {
            mat[i][j] = val++;
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

void deallocate2dArray(double **ptr,int rows,int columns)
{//clean up the allocations to the given pointer
    int i;
    for(i = 0; i < rows; i++) 
    {
        free(ptr[i]);
    }
    free(ptr);
}

void print2dMatrix(double **ptr,int rows,int columns,char name)
{
    printf("\n%c",name);
    int i,j;
    for(i=0;i<rows;i++)
    {
        for(j=0;j<columns;j++)
        {
            printf("%.1lf ",ptr[i][j]);
        }
    }
    printf("\n");
}

int main(int argc,char **argv)
{
    double **A, **B, **C;
    int N = 1,block_size=1 ,i ,j ,matrix=0;
    struct timeval tv1, tv2;
    struct timezone tz;
    
    sscanf(argv[1],"%d",&N);
    sscanf(argv[2],"%d",&block_size);
    sscanf(argv[2],"%d",&matrix);
    
    //allocate and initialize arrays
    A = allocate2dArray(N,N);
    B = allocate2dArray(N,N);
    C = allocate2dArray(N,N);
    
    if (!A||!B||!C)
    {
        printf( "Out of memory, reduce N value.\n");
        exit(-1);
    }
    initMat(N,N,A);
    initMat(N,N,B);
    cleanMat(N,N,C);

    switch(matrix)
    {
        case 1:
        {
            gettimeofday(&tv1, &tz);
            multiplyMatrixBlockedIJK(A,B,C,N,block_size);
            gettimeofday(&tv2, &tz);
            break;
        }
        case 2:
        {
            gettimeofday(&tv1, &tz);
            multiplyMatrixBlockedKIJ(A,B,C,N,block_size);
            gettimeofday(&tv2, &tz);
            break;
        }
        default:
        {
            gettimeofday(&tv1, &tz);
            multiplyMatrixNonBlockedIJK(A,B,C,N);
            gettimeofday(&tv2, &tz);
            break;
        } 
    }
    
    double elapsed = (double) (tv2.tv_sec-tv1.tv_sec) + (double) (tv2.tv_usec-tv1.tv_usec) * 1.e-6;
    
    printf("%lf",elapsed);
    //printf("\nDone...\n");
    //clean up the allocation
    deallocate2dArray(A,N,N);
    deallocate2dArray(B,N,N);
    deallocate2dArray(C,N,N);

    return 0;
}