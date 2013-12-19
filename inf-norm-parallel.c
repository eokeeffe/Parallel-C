/*
    Row wise slicing
    Matrix Multiplication & Inifinite Normal
    
    opennMP function get_num_procs() used for MAXTHREAD variable
    
    N.B All N-Dimensional arrays in C are 1-Dimensional ,
    C arrays are contiguous arrays
    
    How to compile: gcc -o program program.c -lpthread -fopenmp
*/
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <pthread.h>
#include <omp.h>

/* pthread mutex and condition variable declaration */
pthread_mutex_t  mutex_norm = PTHREAD_MUTEX_INITIALIZER;

double **A,**B,**C, NormRow = 0;
int number_of_rows,numberOfThreads,N,MAXTHREADS;

/* Thread callback function */
void  * Mulitply_AND_INF_Norm(int thread_id)
{
    int   current_row = 0,column,i,j,k;
    double   mynorm=0,myRowSum,sum;
    
    int start = ((thread_id - 1) * number_of_rows);
    int end = ((thread_id * number_of_rows) - 1);

    // matrix mult over the strip of rows for this thread
    for (i = start; i <= end; i++)
    {
        for (j = 0; j < N; j++)
        {
            sum = 00;
            for (k = 0; k < N; k++)
            {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    
    //Lock the sum of the matrix , prevents other threads from changing the matrix
    pthread_mutex_lock(&mutex_norm);
    for (current_row = start; current_row <= end; current_row++)
    {
        myRowSum = 0;
        for(column = 0 ;column < N; column++)
        {
            myRowSum += C[current_row][column];
        }

        if(mynorm < myRowSum )
        {
            mynorm = myRowSum;
        }
    }
    pthread_mutex_unlock(&mutex_norm);
    
    //lock the max so it is only value is that of this thread
    pthread_mutex_lock(&mutex_norm);
    //fprintf(stdout,"%d < %d\n",NormRow,mynorm);
    if (NormRow < mynorm)
    {
        NormRow = mynorm;
    }
    pthread_mutex_unlock(&mutex_norm);

    pthread_exit(NULL);
}

void printMatrix(double **mat) 
{
    int i,j;
    fprintf(stdout,"%d*%d\n",N,N);
    for(i=0; i < N; i++)
    {
        for(j=0; j < N; j++)
        {
            fprintf(stdout,"%lf ", mat[i][j]);
        }
        fprintf(stdout,"\n");
    }
}

void initMat(char inc,int M, int N, double **mat)
{//put basic incremented values into the matrix , val=1
    int i, j;
    double val = 10;
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
        {
            if(inc){val++;}
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
    /* variable declaration */
    int            i, j,p,q,
                   return_value_counter,counter,thread_count;
    double         time_start, time_end,memoryused=00;
    struct timeval tv1, tv2;
    struct timezone tz;

    pthread_t      *threads;
    pthread_attr_t ptr;

    /*Argument validation*/
    if( argc < 3 )
    {
        fprintf(stderr,"\t\t Not enough Arguments\n ");
        fprintf(stderr,"\t\t usage : %s <N> <numberOfThreads>\n",argv[0]);
        exit(-1);
    }
    else
    {
        sscanf(argv[1],"%d",&N);
        sscanf(argv[2],"%d",&numberOfThreads);
        MAXTHREADS = omp_get_num_procs() * 8;
    }

    fprintf(stdout,"\n\t\t Input Parameters:");
    fprintf(stdout,"\n\t\t Size of Matrix      :  %d",N);
    fprintf(stdout,"\n\t\t Max #Threads        :  %d",MAXTHREADS);
    fprintf(stdout,"\n\t\t #Threads To be Used :  %d",numberOfThreads);
    fprintf(stdout,"\n");

    // check to make sure the number of threads is suitable for the size of N
    if (N % numberOfThreads != 0)
    {
        fprintf(stderr,"\n Number of numberOfThreads must evenly divide N\n");
        exit(0);
    }
    if(numberOfThreads > N)
    {
        fprintf(stderr,"\nNumber of numberOfThreads should be <= %d",N);
        exit(0);
    }

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

    initMat(0,N,N,A);//set all of A to 1
    initMat(0,N,N,B);//set all of B to 1
    cleanMat(N,N,C);//set all of  C to 0
    
    /* InfinityNorm Of Resulting Matrix */
    /* Row Wise Partitioning */
    number_of_rows = N / numberOfThreads;

    threads = (pthread_t *) malloc(sizeof(pthread_t) * numberOfThreads);

    /*Taking start time*/
    gettimeofday(&tv1, &tz);
    /*
        Create a thread that will have an incremented ID
        Using this ID , the Mulitply_AND_INF_Norm function
        will run on a specific 
    */
    pthread_attr_init(&ptr);
    for (thread_count = 0; thread_count < numberOfThreads; thread_count++)
    {
        return_value_counter = pthread_create(&threads[thread_count], &ptr, (void *(*) (void *)) Mulitply_AND_INF_Norm, (void *) (thread_count+1));
        if(return_value_counter)
    	{
            fprintf(stderr,"\n ERROR : Return code from pthread_create() is %d ",return_value_counter);
            exit(-1);
    	}
    }
    
    /*Joining Threads */
    for (thread_count = 0; thread_count < numberOfThreads; thread_count++)
    {
        return_value_counter = pthread_join(threads[thread_count], NULL);
        if(return_value_counter)
    	{
            fprintf(stderr,"\n ERROR : Return code from pthread_join() is %d ",return_value_counter);
            exit(-1);
    	}
    }
    
    gettimeofday(&tv2, &tz);
    double elapsed = (double) (tv2.tv_sec-tv1.tv_sec) + (double) (tv2.tv_usec-tv1.tv_usec) * 1e-6;
    
    fprintf(stdout,"\n\t\t Row Wise partitioning - Infinity Norm of a Square MatrixDone ");

    pthread_attr_destroy(&ptr);
    
    //printMatrix(A);
    //printMatrix(B);
    //printMatrix(C);
    
    deallocate2dArray(N,N,A);
    deallocate2dArray(N,N,B);
    deallocate2dArray(N,N,C);
    free(threads);
    
    fprintf(stdout,"\n");
    fprintf(stdout,"\n\t\t INF-Norm Answer       :  %lf",NormRow);
    fprintf(stdout,"\n\t\t Memory Used           :  %lf MB",(memoryused/(1024*1024)));
    fprintf(stdout,"\n\t\t Time in  Seconds (T)  :  %lfs",elapsed);
    fprintf(stdout,"\n\t\t\n");
}