#include "gemm.h"
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#define reg32 1    //set this value as 1 if using double buffer scheme
#define doublebuffer 0    //set this value as 1 if using double buffer scheme
#define nodoublebuffer 0    //set this value as 1 if using double buffer scheme
#define unroll24 0  //set this value as 1 if using unroll24


void gemm_bin(int M, int N, int K, float ALPHA, 
        char  *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            char A_PART = A[i*lda+k];
            if(A_PART){
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] += B[k*ldb+j];
                }
            } else {
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] -= B[k*ldb+j];
                }
            }
        }
    }
}

float *random_matrix(int rows, int cols)
{
    int i;
    float *m = calloc(rows*cols, sizeof(float));
    for(i = 0; i < rows*cols; ++i){
        m[i] = (float)rand()/RAND_MAX;
    }
    return m;
}

void time_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<10; ++i){
        gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf ms\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}


void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}


/***********************3. loop interchange with manual vectorization with ALPHA!=1 double buffer ****************/
/* Manual vectorization with loop interchange + loop unrolling with unroll degree 24*/

void gemm_nn_unroll16(int ii, int jj, int kk, float *A, float *B, float *C, float ALPHA, int M, int N, int K,  int lda,int ldb,int ldc)
{
int i1=ii, j1=jj, k1=kk;
  int i=0,j=0,k=0;
  long gvl;
  for ( j = 0; j < N; ) {
      gvl = __builtin_epi_vsetvl(N-j, __epi_e32, __epi_m1);
     for (i = 0; i < M-15; i += 16) {
//	__builtin_prefetch(&C[(i+i1)*ldc+(j+j1)], 0, 3);
     //           __builtin_prefetch(B, 0, 2);
       //         __builtin_prefetch(A, 0, 2);
        __epi_2xf32 vc, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vc10, vc11, vc12, vc13, vc14, vc15,vc16,vc17,vc18,vc19,vc20,vc21,vc22,vc23;
        vc= __builtin_epi_vload_2xf32(&C[(i+i1)*ldc+(j+j1)], gvl);
        vc1= __builtin_epi_vload_2xf32(&C[(i+i1+1)*ldc+(j+j1)], gvl);
        vc2= __builtin_epi_vload_2xf32(&C[(i+i1+2)*ldc+(j+j1)], gvl);
        vc3= __builtin_epi_vload_2xf32(&C[(i+i1+3)*ldc+(j+j1)], gvl);
        vc4= __builtin_epi_vload_2xf32(&C[(i+i1+4)*ldc+(j+j1)], gvl);
        vc5= __builtin_epi_vload_2xf32(&C[(i+i1+5)*ldc+(j+j1)], gvl);
        vc6= __builtin_epi_vload_2xf32(&C[(i+i1+6)*ldc+(j+j1)], gvl);
        vc7= __builtin_epi_vload_2xf32(&C[(i+i1+7)*ldc+(j+j1)], gvl);
        vc8= __builtin_epi_vload_2xf32(&C[(i+i1+8)*ldc+(j+j1)], gvl);
        vc9= __builtin_epi_vload_2xf32(&C[(i+i1+9)*ldc+(j+j1)], gvl);
        vc10= __builtin_epi_vload_2xf32(&C[(i+i1+10)*ldc+(j+j1)], gvl);
        vc11= __builtin_epi_vload_2xf32(&C[(i+i1+11)*ldc+(j+j1)], gvl);
        vc12= __builtin_epi_vload_2xf32(&C[(i+i1+12)*ldc+(j+j1)], gvl);
        vc13= __builtin_epi_vload_2xf32(&C[(i+i1+13)*ldc+(j+j1)], gvl);
        vc14= __builtin_epi_vload_2xf32(&C[(i+i1+14)*ldc+(j+j1)], gvl);
        vc15= __builtin_epi_vload_2xf32(&C[(i+i1+15)*ldc+(j+j1)], gvl);
        //
//	__builtin_prefetch(B, 0, 3);
  //              __builtin_prefetch(A, 0, 3);
        for ( k = 0; k < K; k ++) {
                //__epi_2xf32 vb = __builtin_epi_vload_2xf32(&B[k*ldb+j], gvl);
		__epi_2xf32 vb = __builtin_epi_vload_2xf32( &B[((k+(K*(j/ldb)))*ldb)+0], gvl);
               // register float alpha =  A[i+lda*k];
               __epi_2xf32 vaalpha = __builtin_epi_vfmv_v_f_2xf32(A[i+lda*k], gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb, gvl); // sum += ALPHA*A*B
                //register float alpha1 =  A[(i+1)+lda*k];
               __epi_2xf32 vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)+lda*k], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl); // sum += ALPHA*A*B
                //register float alpha2 =  A[(i+2)+lda*k];
               __epi_2xf32 vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)+lda*k], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl); // sum += ALPHA*A*B
                //register float alpha3 =  A[(i+3)+lda*k];
               __epi_2xf32 vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)+lda*k], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl); // sum += ALPHA*A*B
               // register float alpha4 =  A[(i+4)+lda*k];
               __epi_2xf32 vaalpha4 = __builtin_epi_vfmv_v_f_2xf32(A[(i+4)+lda*k], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb, gvl); // sum += ALPHA*A*B
               // register float alpha5 =  A[(i+5)+lda*k];
               __epi_2xf32 vaalpha5 = __builtin_epi_vfmv_v_f_2xf32(A[(i+5)+lda*k], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb, gvl); // sum += ALPHA*A*B
               // register float alpha6 =  A[(i+6)+lda*k];
               __epi_2xf32 vaalpha6 = __builtin_epi_vfmv_v_f_2xf32(A[(i+6)+lda*k], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb, gvl); // sum += ALPHA*A*B
               // register float alpha7 =  A[(i+7)+lda*k];
               __epi_2xf32 vaalpha7 = __builtin_epi_vfmv_v_f_2xf32(A[(i+7)+lda*k], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb, gvl); // sum += ALPHA*A*B
               // register float alpha8 =  A[(i+8)+lda*k];
               __epi_2xf32 vaalpha8 = __builtin_epi_vfmv_v_f_2xf32(A[(i+8)+lda*k], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb, gvl); // sum += ALPHA*A*B
               // register float alpha9 =  A[(i+9)+lda*k];
               __epi_2xf32 vaalpha9= __builtin_epi_vfmv_v_f_2xf32(A[(i+9)+lda*k], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb, gvl); // sum += ALPHA*A*B
               // register float alpha10 =  A[(i+10)+lda*k];
               __epi_2xf32 vaalpha10 = __builtin_epi_vfmv_v_f_2xf32(A[(i+10)+lda*k], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb, gvl); // sum += ALPHA*A*B
               // register float alpha11 =  A[(i+11)+lda*k];
               __epi_2xf32 vaalpha11 = __builtin_epi_vfmv_v_f_2xf32(A[(i+11)+lda*k], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb, gvl); // sum += ALPHA*A*B
               // register float alpha12 =  A[(i+12)+lda*k];
               __epi_2xf32 vaalpha12 = __builtin_epi_vfmv_v_f_2xf32(A[(i+12)+lda*k], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb, gvl); // sum += ALPHA*A*B
               // register float alpha13 =  A[(i+13)+lda*k];
               __epi_2xf32 vaalpha13 = __builtin_epi_vfmv_v_f_2xf32(A[(i+13)+lda*k], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb, gvl); // sum += ALPHA*A*B
               // register float alpha14 =  A[(i+14)+lda*k];
               __epi_2xf32 vaalpha14 = __builtin_epi_vfmv_v_f_2xf32(A[(i+14)+lda*k], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb, gvl); // sum += ALPHA*A*B
               // register float alpha15 = A[(i+15)+lda*k];
		__epi_2xf32 vaalpha15 = __builtin_epi_vfmv_v_f_2xf32(A[(i+15)+lda*k], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb, gvl); // sum += ALPHA*A*B
                  //-----
        }
                __builtin_epi_vstore_2xf32(&C[(i+i1)*ldc+(j+j1)], vc, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+i1+1)*ldc+(j+j1)], vc1, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+i1+2)*ldc+(j+j1)], vc2, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+i1+3)*ldc+(j+j1)], vc3, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+i1+4)*ldc+(j+j1)], vc4, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+i1+5)*ldc+(j+j1)], vc5, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+i1+6)*ldc+(j+j1)], vc6, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+i1+7)*ldc+(j+j1)], vc7, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+i1+8)*ldc+(j+j1)], vc8, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+i1+9)*ldc+(j+j1)], vc9, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+i1+10)*ldc+(j+j1)], vc10, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+i1+11)*ldc+(j+j1)], vc11, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+i1+12)*ldc+(j+j1)], vc12, gvl);
  		 __builtin_epi_vstore_2xf32(&C[(i+i1+13)*ldc+(j+j1)], vc13, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+i1+14)*ldc+(j+j1)], vc14, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+i1+15)*ldc+(j+j1)], vc15, gvl);
                //

        }
    j += gvl;
     }

  int i_left=i;
  //itr=0;
  for (int j = 0; j < N; ) {
     __epi_2xf32  vaalpha, vaalpha1, vaalpha2, vaalpha3, vc, vc1, vc2, vc3,vb;
     float alpha1, alpha2, alpha3, alpha;

     unsigned long int gvl = __builtin_epi_vsetvl(N-j, __epi_e32, __epi_m1);
     for (i=i_left; i < M; i += 4) {    // change according to unroll degree
        vc= __builtin_epi_vload_2xf32(&C[(i+i1)*ldc+(j+j1)], gvl);
       if (i+1 < M) { vc1= __builtin_epi_vload_2xf32(&C[(i+i1+1)*ldc+(j+j1)], gvl);}
       if (i+2 < M) { vc2= __builtin_epi_vload_2xf32(&C[(i+i1+2)*ldc+(j+j1)], gvl);}
       if (i+3 < M) {vc3= __builtin_epi_vload_2xf32(&C[(i+i1+3)*ldc+(j+j1)], gvl);}
  for (int k = 0; k < K; k ++) {
                alpha =  A[i+lda*k];
                if (i+1 < M) {alpha1 = A[(i+1)+lda*k]; }
                if (i+2 < M) { alpha2 =  A[(i+2)+lda*k];}
                if (i+3 < M) { alpha3 =  A[(i+3)+lda*k];}
                vaalpha = __builtin_epi_vfmv_v_f_2xf32(alpha, gvl); // ALPHA*A
               if (i+1 < M) { vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(alpha1, gvl);} // ALPHA*A
               if (i+2 < M) { vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(alpha2, gvl);} // ALPHA*A
               if (i+3 < M) { vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(alpha3, gvl);} // ALPHA*A
                //vb = __builtin_epi_vload_2xf32(&B[k*ldb+j], gvl);
                vb = __builtin_epi_vload_2xf32(&B[((k+(K*(j/ldb)))*ldb)+0], gvl);
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb, gvl); // sum += ALPHA*A*B
                  if (i+1 < M) {vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl);} // sum += ALPHA*A*B
                  if (i+2 < M) {vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl);} // sum += ALPHA*A*B
                  if (i+3 < M) {vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl);}// sum += ALPHA*A*B
        }
          __builtin_epi_vstore_2xf32(&C[(i+i1)*ldc+(j+j1)], vc, gvl);
          if (i+1 < M)      {__builtin_epi_vstore_2xf32(&C[(i+i1+1)*ldc+(j+j1)], vc1, gvl);}
          if (i+2 < M)      {__builtin_epi_vstore_2xf32(&C[(i+i1+2)*ldc+(j+j1)], vc2, gvl);}
          if (i+3 < M)      {__builtin_epi_vstore_2xf32(&C[(i+i1+3)*ldc+(j+j1)], vc3, gvl);}
     }
     j += gvl;
  }
}


/***********************3. loop interchange with manual vectorization with ALPHA==1 double buffer ****************/
/* Manual vectorization with loop interchange + loop unrolling with unroll degree 24*/
void gemm_nn_noalpha_unroll163loops(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
	printf("3-loops");
  int i=0,j=0,k=0;
  long gvl;
  if(M>15){
  for ( j = 0; j < N; ) {
      gvl = __builtin_epi_vsetvl(N-j, __epi_e32, __epi_m1);
     //for (i = 0; i < M-23; i += 24) {
     for (i = 0; i < M-15; i += 16) {
        __epi_2xf32 vc, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vc10, vc11, vc12, vc13, vc14, vc15,vc16,vc17,vc18,vc19,vc20,vc21,vc22,vc23;
        vc= __builtin_epi_vload_2xf32(&C[i*ldc+j], gvl);
        vc1= __builtin_epi_vload_2xf32(&C[(i+1)*ldc+j], gvl);
        vc2= __builtin_epi_vload_2xf32(&C[(i+2)*ldc+j], gvl);
        vc3= __builtin_epi_vload_2xf32(&C[(i+3)*ldc+j], gvl);
        vc4= __builtin_epi_vload_2xf32(&C[(i+4)*ldc+j], gvl);
        vc5= __builtin_epi_vload_2xf32(&C[(i+5)*ldc+j], gvl);
        vc6= __builtin_epi_vload_2xf32(&C[(i+6)*ldc+j], gvl);
        vc7= __builtin_epi_vload_2xf32(&C[(i+7)*ldc+j], gvl);
        vc8= __builtin_epi_vload_2xf32(&C[(i+8)*ldc+j], gvl);
        vc9= __builtin_epi_vload_2xf32(&C[(i+9)*ldc+j], gvl);
        vc10= __builtin_epi_vload_2xf32(&C[(i+10)*ldc+j], gvl);
        vc11= __builtin_epi_vload_2xf32(&C[(i+11)*ldc+j], gvl);
        vc12= __builtin_epi_vload_2xf32(&C[(i+12)*ldc+j], gvl);
        vc13= __builtin_epi_vload_2xf32(&C[(i+13)*ldc+j], gvl);
        vc14= __builtin_epi_vload_2xf32(&C[(i+14)*ldc+j], gvl);
        vc15= __builtin_epi_vload_2xf32(&C[(i+15)*ldc+j], gvl);
        //
	
        for ( k = 0; k < K; k ++) {
                __epi_2xf32 vb = __builtin_epi_vload_2xf32(&B[k*ldb+j], gvl);

               __epi_2xf32 vaalpha = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+k], gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+k], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)*lda+k], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+k], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha4 = __builtin_epi_vfmv_v_f_2xf32(A[(i+4)*lda+k], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha5 = __builtin_epi_vfmv_v_f_2xf32(A[(i+5)*lda+k], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha6 = __builtin_epi_vfmv_v_f_2xf32(A[(i+6)*lda+k], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha7 = __builtin_epi_vfmv_v_f_2xf32(A[(i+7)*lda+k], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha8 = __builtin_epi_vfmv_v_f_2xf32(A[(i+8)*lda+k], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha9= __builtin_epi_vfmv_v_f_2xf32(A[(i+9)*lda+k], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha10 = __builtin_epi_vfmv_v_f_2xf32(A[(i+10)*lda+k], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha11 = __builtin_epi_vfmv_v_f_2xf32(A[(i+11)*lda+k], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha12 = __builtin_epi_vfmv_v_f_2xf32(A[(i+12)*lda+k], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha13 = __builtin_epi_vfmv_v_f_2xf32(A[(i+13)*lda+k], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha14 = __builtin_epi_vfmv_v_f_2xf32(A[(i+14)*lda+k], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb, gvl); // sum += ALPHA*A*B
		__epi_2xf32 vaalpha15 = __builtin_epi_vfmv_v_f_2xf32(A[(i+15)*lda+k], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb, gvl); // sum += ALPHA*A*B
                  //-----
        
	}
                __builtin_epi_vstore_2xf32(&C[i*ldc+j], vc, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+1)*ldc+j], vc1, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+2)*ldc+j], vc2, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+3)*ldc+j], vc3, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+4)*ldc+j], vc4, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+5)*ldc+j], vc5, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+6)*ldc+j], vc6, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+7)*ldc+j], vc7, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+8)*ldc+j], vc8, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+9)*ldc+j], vc9, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+10)*ldc+j], vc10, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+11)*ldc+j], vc11, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+12)*ldc+j], vc12, gvl);
  		 __builtin_epi_vstore_2xf32(&C[(i+13)*ldc+j], vc13, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+14)*ldc+j], vc14, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+15)*ldc+j], vc15, gvl);
                //
        }
    j += gvl;
     }}

  int i_left=i;
  for (int j = 0; j < N; ) {
     __epi_2xf32  vaalpha, vaalpha1, vaalpha2, vaalpha3, vc, vc1, vc2, vc3,vb;
     float alpha1, alpha2, alpha3, alpha;

     unsigned long int gvl = __builtin_epi_vsetvl(N-j, __epi_e32, __epi_m1);
     for (i=i_left; i < M; i += 4) {    // change according to unroll degree
        vc= __builtin_epi_vload_2xf32(&C[i*ldc+j], gvl);
       if (i+1 < M) { vc1= __builtin_epi_vload_2xf32(&C[(i+1)*ldc+j], gvl);}
       if (i+2 < M) { vc2= __builtin_epi_vload_2xf32(&C[(i+2)*ldc+j], gvl);}
       if (i+3 < M) {vc3= __builtin_epi_vload_2xf32(&C[(i+3)*ldc+j], gvl);}
  for (int k = 0; k < K; k ++) {
                vaalpha = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+k], gvl); // ALPHA*A
               if (i+1 < M) { vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+k], gvl);} // ALPHA*A
               if (i+2 < M) { vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)*lda+k], gvl);} // ALPHA*A
               if (i+3 < M) { vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+k], gvl);} // ALPHA*A
                vb = __builtin_epi_vload_2xf32(&B[k*ldb+j], gvl);
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb, gvl); // sum += ALPHA*A*B
                  if (i+1 < M) {vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl);} // sum += ALPHA*A*B
                  if (i+2 < M) {vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl);} // sum += ALPHA*A*B
                  if (i+3 < M) {vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl);}// sum += ALPHA*A*B
        }
          __builtin_epi_vstore_2xf32(&C[i*ldc+j], vc, gvl);
          if (i+1 < M)      {__builtin_epi_vstore_2xf32(&C[(i+1)*ldc+j], vc1, gvl);}
          if (i+2 < M)      {__builtin_epi_vstore_2xf32(&C[(i+2)*ldc+j], vc2, gvl);}
          if (i+3 < M)      {__builtin_epi_vstore_2xf32(&C[(i+3)*ldc+j], vc3, gvl);}
     }
     j += gvl;
  }
}

/***********************3. loop interchange with manual vectorization with ALPHA==1 double buffer ****************/
/* Manual vectorization with loop interchange + loop unrolling with unroll degree 24*/
/*void gemm_nn_noalpha_unroll24(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
  int i=0,j=0,k=0;
  long gvl;
  if(M>23){
  for ( j = 0; j < N; ) {
      gvl = __builtin_epi_vsetvl(N-j, __epi_e32, __epi_m1);
     for (i = 0; i < M-23; i += 24) {
        __epi_2xf32 vc, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vc10, vc11, vc12, vc13, vc14, vc15,vc16,vc17,vc18,vc19,vc20,vc21,vc22,vc23;
        vc= __builtin_epi_vload_2xf32(&C[i*ldc+j], gvl);
        vc1= __builtin_epi_vload_2xf32(&C[(i+1)*ldc+j], gvl);
        vc2= __builtin_epi_vload_2xf32(&C[(i+2)*ldc+j], gvl);
        vc3= __builtin_epi_vload_2xf32(&C[(i+3)*ldc+j], gvl);
        vc4= __builtin_epi_vload_2xf32(&C[(i+4)*ldc+j], gvl);
        vc5= __builtin_epi_vload_2xf32(&C[(i+5)*ldc+j], gvl);
        vc6= __builtin_epi_vload_2xf32(&C[(i+6)*ldc+j], gvl);
        vc7= __builtin_epi_vload_2xf32(&C[(i+7)*ldc+j], gvl);
        vc8= __builtin_epi_vload_2xf32(&C[(i+8)*ldc+j], gvl);
        vc9= __builtin_epi_vload_2xf32(&C[(i+9)*ldc+j], gvl);
        vc10= __builtin_epi_vload_2xf32(&C[(i+10)*ldc+j], gvl);
        vc11= __builtin_epi_vload_2xf32(&C[(i+11)*ldc+j], gvl);
        vc12= __builtin_epi_vload_2xf32(&C[(i+12)*ldc+j], gvl);
        vc13= __builtin_epi_vload_2xf32(&C[(i+13)*ldc+j], gvl);
        vc14= __builtin_epi_vload_2xf32(&C[(i+14)*ldc+j], gvl);
        vc15= __builtin_epi_vload_2xf32(&C[(i+15)*ldc+j], gvl);
        //
       vc16= __builtin_epi_vload_2xf32(&C[(i+16)*ldc+j], gvl);
        vc17= __builtin_epi_vload_2xf32(&C[(i+17)*ldc+j], gvl);
        vc18= __builtin_epi_vload_2xf32(&C[(i+18)*ldc+j], gvl);
        vc19= __builtin_epi_vload_2xf32(&C[(i+19)*ldc+j], gvl);
        vc20= __builtin_epi_vload_2xf32(&C[(i+20)*ldc+j], gvl);
        vc21= __builtin_epi_vload_2xf32(&C[(i+21)*ldc+j], gvl);
        vc22= __builtin_epi_vload_2xf32(&C[(i+22)*ldc+j], gvl);
        vc23= __builtin_epi_vload_2xf32(&C[(i+23)*ldc+j], gvl);
	
        for ( k = 0; k < K; k ++) {
                __epi_2xf32 vb = __builtin_epi_vload_2xf32(&B[k*ldb+j], gvl);

               __epi_2xf32 vaalpha = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+k], gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+k], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)*lda+k], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+k], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha4 = __builtin_epi_vfmv_v_f_2xf32(A[(i+4)*lda+k], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha5 = __builtin_epi_vfmv_v_f_2xf32(A[(i+5)*lda+k], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha6 = __builtin_epi_vfmv_v_f_2xf32(A[(i+6)*lda+k], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha7 = __builtin_epi_vfmv_v_f_2xf32(A[(i+7)*lda+k], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha8 = __builtin_epi_vfmv_v_f_2xf32(A[(i+8)*lda+k], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha9= __builtin_epi_vfmv_v_f_2xf32(A[(i+9)*lda+k], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha10 = __builtin_epi_vfmv_v_f_2xf32(A[(i+10)*lda+k], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha11 = __builtin_epi_vfmv_v_f_2xf32(A[(i+11)*lda+k], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha12 = __builtin_epi_vfmv_v_f_2xf32(A[(i+12)*lda+k], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha13 = __builtin_epi_vfmv_v_f_2xf32(A[(i+13)*lda+k], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb, gvl); // sum += ALPHA*A*B
               __epi_2xf32 vaalpha14 = __builtin_epi_vfmv_v_f_2xf32(A[(i+14)*lda+k], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb, gvl); // sum += ALPHA*A*B
__epi_2xf32 vaalpha15 = __builtin_epi_vfmv_v_f_2xf32(A[(i+15)*lda+k], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb, gvl); // sum += ALPHA*A*B
                  //-----
               __epi_2xf32 vaalpha16 = __builtin_epi_vfmv_v_f_2xf32(A[(i+16)*lda+k], gvl); // ALPHA*A
                  vc16 = __builtin_epi_vfmacc_2xf32(vc16, vaalpha16, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha17 = __builtin_epi_vfmv_v_f_2xf32(A[(i+17)*lda+k], gvl); // ALPHA*A
                  vc17 = __builtin_epi_vfmacc_2xf32(vc17, vaalpha17, vb, gvl); // sum += ALPHA*A*B


               __epi_2xf32 vaalpha18 = __builtin_epi_vfmv_v_f_2xf32(A[(i+18)*lda+k], gvl); // ALPHA*A
                  vc18 = __builtin_epi_vfmacc_2xf32(vc18, vaalpha18, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha19 = __builtin_epi_vfmv_v_f_2xf32(A[(i+19)*lda+k], gvl); // ALPHA*A
                  vc19 = __builtin_epi_vfmacc_2xf32(vc19, vaalpha19, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha20 = __builtin_epi_vfmv_v_f_2xf32(A[(i+20)*lda+k], gvl); // ALPHA*A
                  vc20 = __builtin_epi_vfmacc_2xf32(vc20, vaalpha20, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha21 = __builtin_epi_vfmv_v_f_2xf32(A[(i+21)*lda+k], gvl); // ALPHA*A
                  vc21 = __builtin_epi_vfmacc_2xf32(vc21, vaalpha21, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha22 = __builtin_epi_vfmv_v_f_2xf32(A[(i+22)*lda+k], gvl); // ALPHA*A
                  vc22 = __builtin_epi_vfmacc_2xf32(vc22, vaalpha22, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha23 = __builtin_epi_vfmv_v_f_2xf32( A[(i+23)*lda+k], gvl); // ALPHA*A
                  vc23 = __builtin_epi_vfmacc_2xf32(vc23, vaalpha23, vb, gvl); // sum += ALPHA*A*B
        
	}
                __builtin_epi_vstore_2xf32(&C[i*ldc+j], vc, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+1)*ldc+j], vc1, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+2)*ldc+j], vc2, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+3)*ldc+j], vc3, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+4)*ldc+j], vc4, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+5)*ldc+j], vc5, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+6)*ldc+j], vc6, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+7)*ldc+j], vc7, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+8)*ldc+j], vc8, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+9)*ldc+j], vc9, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+10)*ldc+j], vc10, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+11)*ldc+j], vc11, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+12)*ldc+j], vc12, gvl);
  		 __builtin_epi_vstore_2xf32(&C[(i+13)*ldc+j], vc13, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+14)*ldc+j], vc14, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+15)*ldc+j], vc15, gvl);
                //
                __builtin_epi_vstore_2xf32(&C[(i+16)*ldc+j], vc16, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+17)*ldc+j], vc17, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+18)*ldc+j], vc18, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+19)*ldc+j], vc19, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+20)*ldc+j], vc20, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+21)*ldc+j], vc21, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+22)*ldc+j], vc22, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+23)*ldc+j], vc23, gvl);
	

        }
    j += gvl;
     }}

  int i_left=i;
  for (int j = 0; j < N; ) {
     __epi_2xf32  vaalpha, vaalpha1, vaalpha2, vaalpha3, vc, vc1, vc2, vc3,vb;
     float alpha1, alpha2, alpha3, alpha;

     unsigned long int gvl = __builtin_epi_vsetvl(N-j, __epi_e32, __epi_m1);
     for (i=i_left; i < M; i += 4) {    // change according to unroll degree
        vc= __builtin_epi_vload_2xf32(&C[i*ldc+j], gvl);
       if (i+1 < M) { vc1= __builtin_epi_vload_2xf32(&C[(i+1)*ldc+j], gvl);}
       if (i+2 < M) { vc2= __builtin_epi_vload_2xf32(&C[(i+2)*ldc+j], gvl);}
       if (i+3 < M) {vc3= __builtin_epi_vload_2xf32(&C[(i+3)*ldc+j], gvl);}
  for (int k = 0; k < K; k ++) {
                vaalpha = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+k], gvl); // ALPHA*A
               if (i+1 < M) { vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+k], gvl);} // ALPHA*A
               if (i+2 < M) { vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)*lda+k], gvl);} // ALPHA*A
               if (i+3 < M) { vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+k], gvl);} // ALPHA*A
                vb = __builtin_epi_vload_2xf32(&B[k*ldb+j], gvl);
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb, gvl); // sum += ALPHA*A*B
                  if (i+1 < M) {vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl);} // sum += ALPHA*A*B
                  if (i+2 < M) {vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl);} // sum += ALPHA*A*B
                  if (i+3 < M) {vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl);}// sum += ALPHA*A*B
        }
          __builtin_epi_vstore_2xf32(&C[i*ldc+j], vc, gvl);
          if (i+1 < M)      {__builtin_epi_vstore_2xf32(&C[(i+1)*ldc+j], vc1, gvl);}
          if (i+2 < M)      {__builtin_epi_vstore_2xf32(&C[(i+2)*ldc+j], vc2, gvl);}
          if (i+3 < M)      {__builtin_epi_vstore_2xf32(&C[(i+3)*ldc+j], vc3, gvl);}
     }
     j += gvl;
  }
}*/


/***** Manual vectorization with 16 unroll + 8 unroll + double buffer = 32 vector register usage*///
/***********************3. loop interchange with manual vectorization with ALPHA=1 double buffer ****************/
/* Manual vectorization with loop interchange + loop unrolling*/
void gemm_nn_noalpha_doublebuffwith32reg(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
  int i=0,j=0,k=0;
  long gvl;
  if(M>15){
  for ( j = 0; j < N; ) {
      gvl = __builtin_epi_vsetvl(N-j, __epi_e32, __epi_m1);
     for (i = 0; i < M-15; i += 16) {
        __epi_2xf32 vb,vb1,vb2,vb3,vb4,vb5,vb6,vb7,vb8, vb9, vb10, vb11, vb12, vb13, vb14, vb15,vc, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vc10, vc11, vc12, vc13, vc14, vc15,vc16,vc17,vc18,vc19,vc20,vc21,vc22,vc23;

        vc= __builtin_epi_vload_2xf32(&C[i*ldc+j], gvl);
        vc1= __builtin_epi_vload_2xf32(&C[(i+1)*ldc+j], gvl);
        vc2= __builtin_epi_vload_2xf32(&C[(i+2)*ldc+j], gvl);
        vc3= __builtin_epi_vload_2xf32(&C[(i+3)*ldc+j], gvl);
        vc4= __builtin_epi_vload_2xf32(&C[(i+4)*ldc+j], gvl);
        vc5= __builtin_epi_vload_2xf32(&C[(i+5)*ldc+j], gvl);
        vc6= __builtin_epi_vload_2xf32(&C[(i+6)*ldc+j], gvl);
        vc7= __builtin_epi_vload_2xf32(&C[(i+7)*ldc+j], gvl);
        vc8= __builtin_epi_vload_2xf32(&C[(i+8)*ldc+j], gvl);
        vc9= __builtin_epi_vload_2xf32(&C[(i+9)*ldc+j], gvl);
        vc10= __builtin_epi_vload_2xf32(&C[(i+10)*ldc+j], gvl);
        vc11= __builtin_epi_vload_2xf32(&C[(i+11)*ldc+j], gvl);
        vc12= __builtin_epi_vload_2xf32(&C[(i+12)*ldc+j], gvl);
        vc13= __builtin_epi_vload_2xf32(&C[(i+13)*ldc+j], gvl);
        vc14= __builtin_epi_vload_2xf32(&C[(i+14)*ldc+j], gvl);
        vc15= __builtin_epi_vload_2xf32(&C[(i+15)*ldc+j], gvl);
	// double buffer scheme implementation -start
        int flag=0;
	for ( k = 0; k < K-7; k +=8) {
		 if (flag==0){
                vb = __builtin_epi_vload_2xf32(&B[k*ldb+j], gvl);
                vb1 = __builtin_epi_vload_2xf32(&B[(k+1)*ldb+j], gvl);
                vb2 = __builtin_epi_vload_2xf32(&B[(k+2)*ldb+j], gvl);
                vb3 = __builtin_epi_vload_2xf32(&B[(k+3)*ldb+j], gvl);
                 vb4 = __builtin_epi_vload_2xf32(&B[(k+4)*ldb+j], gvl);
                 vb5 = __builtin_epi_vload_2xf32(&B[(k+5)*ldb+j], gvl);
                 vb6 = __builtin_epi_vload_2xf32(&B[(k+6)*ldb+j], gvl);
                 vb7 = __builtin_epi_vload_2xf32(&B[(k+7)*ldb+j], gvl);
		  vb8 = __builtin_epi_vload_2xf32(&B[(k+8)*ldb+j], gvl);
                vb9 = __builtin_epi_vload_2xf32(&B[(k+9)*ldb+j], gvl);
                vb10 = __builtin_epi_vload_2xf32(&B[(k+10)*ldb+j], gvl);
                vb11 = __builtin_epi_vload_2xf32(&B[(k+11)*ldb+j], gvl);
                 vb12 = __builtin_epi_vload_2xf32(&B[(k+12)*ldb+j], gvl);
                 vb13 = __builtin_epi_vload_2xf32(&B[(k+13)*ldb+j], gvl);
                 vb14 = __builtin_epi_vload_2xf32(&B[(k+14)*ldb+j], gvl);
                 vb15 = __builtin_epi_vload_2xf32(&B[(k+15)*ldb+j], gvl);
                }
                else
                {
			if(flag & 1)
			{
			   if(k<K-8)
			   {
                 		vb = __builtin_epi_vload_2xf32(&B[(k+8)*ldb+j], gvl);
                 		vb1 = __builtin_epi_vload_2xf32(&B[(k+9)*ldb+j], gvl);
                 		vb2 = __builtin_epi_vload_2xf32(&B[(k+10)*ldb+j], gvl);
                 		vb3 = __builtin_epi_vload_2xf32(&B[(k+11)*ldb+j], gvl);
				vb4 = __builtin_epi_vload_2xf32(&B[(k+12)*ldb+j], gvl);
                 		vb5 = __builtin_epi_vload_2xf32(&B[(k+13)*ldb+j], gvl);
                 		vb6 = __builtin_epi_vload_2xf32(&B[(k+14)*ldb+j], gvl);
                 		vb7 = __builtin_epi_vload_2xf32(&B[(k+15)*ldb+j], gvl);

                           }
                        }
                        else
			{
			    if(k<K-8)
                           {
                                vb8 = __builtin_epi_vload_2xf32(&B[(k+8)*ldb+j], gvl);
                                vb9 = __builtin_epi_vload_2xf32(&B[(k+9)*ldb+j], gvl);
                                vb10 = __builtin_epi_vload_2xf32(&B[(k+10)*ldb+j], gvl);
                                vb11 = __builtin_epi_vload_2xf32(&B[(k+11)*ldb+j], gvl);
			       vb12 = __builtin_epi_vload_2xf32(&B[(k+12)*ldb+j], gvl);
                                vb13 = __builtin_epi_vload_2xf32(&B[(k+13)*ldb+j], gvl);
                                vb14 = __builtin_epi_vload_2xf32(&B[(k+14)*ldb+j], gvl);
                                vb15 = __builtin_epi_vload_2xf32(&B[(k+15)*ldb+j], gvl);
                           }
                	}
		}

		//double buffer scheme implementation - end
		if(flag & 1)
		{

               __epi_2xf32 vaalpha = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+k], gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb8, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha0 = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+(k+1)], gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha0, vb9, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+k], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb8, gvl); // sum += ALPHA*A*B

		__epi_2xf32 vaalpha01 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+(k+1)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb9, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)*lda+k], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb8, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha02 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)*lda+(k+1)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb9, gvl); // sum += ALPHA*A*B


               __epi_2xf32 vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+k], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb8, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha03 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+(k+1)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb9, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha4 = __builtin_epi_vfmv_v_f_2xf32(A[(i+4)*lda+k], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb8, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha04 = __builtin_epi_vfmv_v_f_2xf32(A[(i+4)*lda+(k+1)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb9, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha5 = __builtin_epi_vfmv_v_f_2xf32(A[(i+5)*lda+k], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb8, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha05 = __builtin_epi_vfmv_v_f_2xf32( A[(i+5)*lda+(k+1)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb9, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha6 = __builtin_epi_vfmv_v_f_2xf32(A[(i+6)*lda+k], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb8, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha06 = __builtin_epi_vfmv_v_f_2xf32(A[(i+6)*lda+(k+1)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb9, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha7 = __builtin_epi_vfmv_v_f_2xf32(A[(i+7)*lda+k], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb8, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha07 = __builtin_epi_vfmv_v_f_2xf32(A[(i+7)*lda+(k+1)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb9, gvl); // sum += ALPHA*A*B


               __epi_2xf32 vaalpha8 = __builtin_epi_vfmv_v_f_2xf32(A[(i+8)*lda+k], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb8, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha08 = __builtin_epi_vfmv_v_f_2xf32(A[(i+8)*lda+(k+1)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb9, gvl); // sum += ALPHA*A*B


               __epi_2xf32 vaalpha9= __builtin_epi_vfmv_v_f_2xf32(A[(i+9)*lda+k], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb8, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha09= __builtin_epi_vfmv_v_f_2xf32(A[(i+9)*lda+(k+1)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb9, gvl); // sum += ALPHA*A*B


               __epi_2xf32 vaalpha10 = __builtin_epi_vfmv_v_f_2xf32(A[(i+10)*lda+k], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb8, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha010 = __builtin_epi_vfmv_v_f_2xf32(A[(i+10)*lda+(k+1)], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb9, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha11 = __builtin_epi_vfmv_v_f_2xf32(A[(i+11)*lda+k], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb8, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha011 = __builtin_epi_vfmv_v_f_2xf32(A[(i+11)*lda+(k+1)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb9, gvl); // sum += ALPHA*A*B


               __epi_2xf32 vaalpha12 = __builtin_epi_vfmv_v_f_2xf32(A[(i+12)*lda+k], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb8, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha012 = __builtin_epi_vfmv_v_f_2xf32(A[(i+12)*lda+(k+1)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb9, gvl); // sum += ALPHA*A*B


               __epi_2xf32 vaalpha13 = __builtin_epi_vfmv_v_f_2xf32(A[(i+13)*lda+k], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb8, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha013 = __builtin_epi_vfmv_v_f_2xf32(A[(i+13)*lda+(k+1)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb9, gvl); // sum += ALPHA*A*B


               __epi_2xf32 vaalpha14 = __builtin_epi_vfmv_v_f_2xf32(A[(i+14)*lda+k], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb8, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha014 = __builtin_epi_vfmv_v_f_2xf32(A[(i+14)*lda+(k+1)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb9, gvl); // sum += ALPHA*A*B


               __epi_2xf32 vaalpha15 = __builtin_epi_vfmv_v_f_2xf32(A[(i+15)*lda+k], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb8, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha015 = __builtin_epi_vfmv_v_f_2xf32(A[(i+15)*lda+(k+1)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb9, gvl); // sum += ALPHA*A*B
		  //-----


		/* unroll 4*/

                vaalpha = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+(k+2)], gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb10, gvl); // sum += ALPHA*A*B

                vaalpha0 = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+(k+3)], gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha0, vb11, gvl); // sum += ALPHA*A*B

               vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+(k+2)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb10, gvl); // sum += ALPHA*A*B

               vaalpha01 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+(k+3)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb11, gvl); // sum += ALPHA*A*B

               vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)*lda+(k+2)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb10, gvl); // sum += ALPHA*A*B

                vaalpha02 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)*lda+(k+3)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb11, gvl); // sum += ALPHA*A*B


                vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+(k+2)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb10, gvl); // sum += ALPHA*A*B

                vaalpha03 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+(k+3)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb11, gvl); // sum += ALPHA*A*B

               vaalpha4 = __builtin_epi_vfmv_v_f_2xf32( A[(i+4)*lda+(k+2)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb10, gvl); // sum += ALPHA*A*B

               vaalpha04 = __builtin_epi_vfmv_v_f_2xf32(A[(i+4)*lda+(k+3)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb11, gvl); // sum += ALPHA*A*B

                vaalpha5 = __builtin_epi_vfmv_v_f_2xf32(A[(i+5)*lda+(k+2)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb10, gvl); // sum += ALPHA*A*B

		vaalpha05 = __builtin_epi_vfmv_v_f_2xf32(A[(i+5)*lda+(k+3)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb11, gvl); // sum += ALPHA*A*B

                vaalpha6 = __builtin_epi_vfmv_v_f_2xf32(A[(i+6)*lda+(k+2)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb10, gvl); // sum += ALPHA*A*B

		vaalpha06 = __builtin_epi_vfmv_v_f_2xf32(A[(i+6)*lda+(k+3)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb11, gvl); // sum += ALPHA*A*B

                 vaalpha7 = __builtin_epi_vfmv_v_f_2xf32(A[(i+7)*lda+(k+2)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb10, gvl); // sum += ALPHA*A*B

                  vaalpha07 = __builtin_epi_vfmv_v_f_2xf32(A[(i+7)*lda+(k+3)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb11, gvl); // sum += ALPHA*A*B


                  vaalpha8 = __builtin_epi_vfmv_v_f_2xf32(A[(i+8)*lda+(k+2)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb10, gvl); // sum += ALPHA*A*B

                  vaalpha08 = __builtin_epi_vfmv_v_f_2xf32(A[(i+8)*lda+(k+3)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb11, gvl); // sum += ALPHA*A*B


                   vaalpha9= __builtin_epi_vfmv_v_f_2xf32(A[(i+9)*lda+(k+2)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb10, gvl); // sum += ALPHA*A*B

               	 vaalpha09= __builtin_epi_vfmv_v_f_2xf32(A[(i+9)*lda+(k+3)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb11, gvl); // sum += ALPHA*A*B


                 vaalpha10 = __builtin_epi_vfmv_v_f_2xf32(A[(i+10)*lda+(k+2)], gvl); // ALPHA*A
                   vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb10, gvl); // sum += ALPHA*A*B

                 vaalpha010 = __builtin_epi_vfmv_v_f_2xf32(A[(i+10)*lda+(k+3)], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb11, gvl); // sum += ALPHA*A*B

                vaalpha11 = __builtin_epi_vfmv_v_f_2xf32(A[(i+11)*lda+(k+2)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb10, gvl); // sum += ALPHA*A*B

                 vaalpha011 = __builtin_epi_vfmv_v_f_2xf32(A[(i+11)*lda+(k+3)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb11, gvl); // sum += ALPHA*A*B


                vaalpha12 = __builtin_epi_vfmv_v_f_2xf32(A[(i+12)*lda+(k+2)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb10, gvl); // sum += ALPHA*A*B

                vaalpha012 = __builtin_epi_vfmv_v_f_2xf32(A[(i+12)*lda+(k+3)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb11, gvl); // sum += ALPHA*A*B


                 vaalpha13 = __builtin_epi_vfmv_v_f_2xf32(A[(i+13)*lda+(k+2)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb10, gvl); // sum += ALPHA*A*B

		vaalpha013 = __builtin_epi_vfmv_v_f_2xf32( A[(i+13)*lda+(k+3)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb11, gvl); // sum += ALPHA*A*B


                   vaalpha14 = __builtin_epi_vfmv_v_f_2xf32(A[(i+14)*lda+(k+2)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb10, gvl); // sum += ALPHA*A*B

                 vaalpha014 = __builtin_epi_vfmv_v_f_2xf32(A[(i+14)*lda+(k+3)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb11, gvl); // sum += ALPHA*A*B


                   vaalpha15 = __builtin_epi_vfmv_v_f_2xf32(A[(i+15)*lda+(k+2)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb10, gvl); // sum += ALPHA*A*B

                   vaalpha015 = __builtin_epi_vfmv_v_f_2xf32(A[(i+15)*lda+(k+3)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb11, gvl); // sum += ALPHA*A*B

		/////
		/* unroll 6*/
		  vaalpha = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+(k+4)], gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb12, gvl); // sum += ALPHA*A*B

                vaalpha0 = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+(k+5)], gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha0, vb13, gvl); // sum += ALPHA*A*B

		  vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+(k+4)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb12, gvl); // sum += ALPHA*A*B

	 	 vaalpha01 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+(k+5)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb13, gvl); // sum += ALPHA*A*B

                vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)*lda+(k+4)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb12, gvl); // sum += ALPHA*A*B

                vaalpha02 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)*lda+(k+5)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb13, gvl); // sum += ALPHA*A*B


                vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+(k+4)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb12, gvl); // sum += ALPHA*A*B

                vaalpha03 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+(k+5)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb13, gvl); // sum += ALPHA*A*B

                vaalpha4 = __builtin_epi_vfmv_v_f_2xf32(A[(i+4)*lda+(k+4)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb12, gvl); // sum += ALPHA*A*B

                vaalpha04 = __builtin_epi_vfmv_v_f_2xf32(A[(i+4)*lda+(k+5)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb13, gvl); // sum += ALPHA*A*B

                vaalpha5 = __builtin_epi_vfmv_v_f_2xf32(A[(i+5)*lda+(k+4)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb12, gvl); // sum += ALPHA*A*B

               vaalpha05 = __builtin_epi_vfmv_v_f_2xf32( A[(i+5)*lda+(k+5)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb13, gvl); // sum += ALPHA*A*B

                vaalpha6 = __builtin_epi_vfmv_v_f_2xf32(A[(i+6)*lda+(k+4)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb12, gvl); // sum += ALPHA*A*B

                vaalpha06 = __builtin_epi_vfmv_v_f_2xf32(A[(i+6)*lda+(k+5)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb13, gvl); // sum += ALPHA*A*B

                vaalpha7 = __builtin_epi_vfmv_v_f_2xf32(A[(i+7)*lda+(k+4)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb12, gvl); // sum += ALPHA*A*B

                vaalpha07 = __builtin_epi_vfmv_v_f_2xf32(A[(i+7)*lda+(k+5)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb13, gvl); // sum += ALPHA*A*B


                vaalpha8 = __builtin_epi_vfmv_v_f_2xf32(A[(i+8)*lda+(k+4)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb12, gvl); // sum += ALPHA*A*B

                vaalpha08 = __builtin_epi_vfmv_v_f_2xf32(A[(i+8)*lda+(k+5)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb13, gvl); // sum += ALPHA*A*B


                vaalpha9= __builtin_epi_vfmv_v_f_2xf32(A[(i+9)*lda+(k+4)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb12, gvl); // sum += ALPHA*A*B

                vaalpha09= __builtin_epi_vfmv_v_f_2xf32(A[(i+9)*lda+(k+5)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb13, gvl); // sum += ALPHA*A*B


                vaalpha10 = __builtin_epi_vfmv_v_f_2xf32(A[(i+10)*lda+(k+4)], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb12, gvl); // sum += ALPHA*A*B

                vaalpha010 = __builtin_epi_vfmv_v_f_2xf32(A[(i+10)*lda+(k+5)], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb13, gvl); // sum += ALPHA*A*B

                vaalpha11 = __builtin_epi_vfmv_v_f_2xf32(A[(i+11)*lda+(k+4)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb12, gvl); // sum += ALPHA*A*B

                vaalpha011 = __builtin_epi_vfmv_v_f_2xf32(A[(i+11)*lda+(k+5)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb13, gvl); // sum += ALPHA*A*B


                vaalpha12 = __builtin_epi_vfmv_v_f_2xf32(A[(i+12)*lda+(k+4)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb12, gvl); // sum += ALPHA*A*B

                vaalpha012 = __builtin_epi_vfmv_v_f_2xf32(A[(i+12)*lda+(k+5)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb13, gvl); // sum += ALPHA*A*B


                vaalpha13 = __builtin_epi_vfmv_v_f_2xf32(A[(i+13)*lda+(k+4)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb12, gvl); // sum += ALPHA*A*B

                vaalpha013 = __builtin_epi_vfmv_v_f_2xf32(A[(i+13)*lda+(k+5)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb13, gvl); // sum += ALPHA*A*B


                vaalpha14 = __builtin_epi_vfmv_v_f_2xf32(A[(i+14)*lda+(k+4)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb12, gvl); // sum += ALPHA*A*B

                vaalpha014 = __builtin_epi_vfmv_v_f_2xf32(A[(i+14)*lda+(k+5)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb13, gvl); // sum += ALPHA*A*B


                vaalpha15 = __builtin_epi_vfmv_v_f_2xf32(A[(i+15)*lda+(k+4)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb12, gvl); // sum += ALPHA*A*B

                vaalpha015 = __builtin_epi_vfmv_v_f_2xf32(A[(i+15)*lda+(k+5)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb13, gvl); // sum += ALPHA*A*B
		  //-----
		/* unroll 8*///

                vaalpha = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+(k+6)], gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb14, gvl); // sum += ALPHA*A*B

                vaalpha0 = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+(k+7)], gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha0, vb15, gvl); // sum += ALPHA*A*B

               vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+(k+6)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb14, gvl); // sum += ALPHA*A*B

               vaalpha01 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+(k+7)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb15, gvl); // sum += ALPHA*A*B

               vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)*lda+(k+6)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb14, gvl); // sum += ALPHA*A*B

                vaalpha02 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)*lda+(k+7)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb15, gvl); // sum += ALPHA*A*B


                vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+(k+6)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb14, gvl); // sum += ALPHA*A*B

                vaalpha03 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+(k+7)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb15, gvl); // sum += ALPHA*A*B

               vaalpha4 = __builtin_epi_vfmv_v_f_2xf32( A[(i+4)*lda+(k+6)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb14, gvl); // sum += ALPHA*A*B

               vaalpha04 = __builtin_epi_vfmv_v_f_2xf32(A[(i+4)*lda+(k+7)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb15, gvl); // sum += ALPHA*A*B

                vaalpha5 = __builtin_epi_vfmv_v_f_2xf32(A[(i+5)*lda+(k+6)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb14, gvl); // sum += ALPHA*A*B

		vaalpha05 = __builtin_epi_vfmv_v_f_2xf32(A[(i+5)*lda+(k+7)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb15, gvl); // sum += ALPHA*A*B

                vaalpha6 = __builtin_epi_vfmv_v_f_2xf32(A[(i+6)*lda+(k+6)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb14, gvl); // sum += ALPHA*A*B

		vaalpha06 = __builtin_epi_vfmv_v_f_2xf32(A[(i+6)*lda+(k+7)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb15, gvl); // sum += ALPHA*A*B

                 vaalpha7 = __builtin_epi_vfmv_v_f_2xf32(A[(i+7)*lda+(k+6)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb14, gvl); // sum += ALPHA*A*B

                  vaalpha07 = __builtin_epi_vfmv_v_f_2xf32(A[(i+7)*lda+(k+7)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb15, gvl); // sum += ALPHA*A*B


                  vaalpha8 = __builtin_epi_vfmv_v_f_2xf32(A[(i+8)*lda+(k+6)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb14, gvl); // sum += ALPHA*A*B

                  vaalpha08 = __builtin_epi_vfmv_v_f_2xf32(A[(i+8)*lda+(k+7)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb15, gvl); // sum += ALPHA*A*B


                   vaalpha9= __builtin_epi_vfmv_v_f_2xf32(A[(i+9)*lda+(k+6)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb14, gvl); // sum += ALPHA*A*B

               	 vaalpha09= __builtin_epi_vfmv_v_f_2xf32(A[(i+9)*lda+(k+7)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb15, gvl); // sum += ALPHA*A*B


                 vaalpha10 = __builtin_epi_vfmv_v_f_2xf32(A[(i+10)*lda+(k+6)], gvl); // ALPHA*A
                   vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb14, gvl); // sum += ALPHA*A*B

                 vaalpha010 = __builtin_epi_vfmv_v_f_2xf32(A[(i+10)*lda+(k+7)], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb15, gvl); // sum += ALPHA*A*B

                vaalpha11 = __builtin_epi_vfmv_v_f_2xf32(A[(i+11)*lda+(k+6)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb14, gvl); // sum += ALPHA*A*B

                 vaalpha011 = __builtin_epi_vfmv_v_f_2xf32(A[(i+11)*lda+(k+7)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb15, gvl); // sum += ALPHA*A*B


                vaalpha12 = __builtin_epi_vfmv_v_f_2xf32(A[(i+12)*lda+(k+6)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb14, gvl); // sum += ALPHA*A*B

                vaalpha012 = __builtin_epi_vfmv_v_f_2xf32(A[(i+12)*lda+(k+7)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb15, gvl); // sum += ALPHA*A*B


                 vaalpha13 = __builtin_epi_vfmv_v_f_2xf32(A[(i+13)*lda+(k+6)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb14, gvl); // sum += ALPHA*A*B

		vaalpha013 = __builtin_epi_vfmv_v_f_2xf32( A[(i+13)*lda+(k+7)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb15, gvl); // sum += ALPHA*A*B


                   vaalpha14 = __builtin_epi_vfmv_v_f_2xf32(A[(i+14)*lda+(k+6)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb14, gvl); // sum += ALPHA*A*B

                 vaalpha014 = __builtin_epi_vfmv_v_f_2xf32(A[(i+14)*lda+(k+7)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb15, gvl); // sum += ALPHA*A*B


                   vaalpha15 = __builtin_epi_vfmv_v_f_2xf32(A[(i+15)*lda+(k+6)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb14, gvl); // sum += ALPHA*A*B

                   vaalpha015 = __builtin_epi_vfmv_v_f_2xf32(A[(i+15)*lda+(k+7)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb15, gvl); // sum += ALPHA*A*B

	}
	else
	{

               __epi_2xf32 vaalpha = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+k], gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha0 = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+(k+1)], gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha0, vb1, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+k], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl); // sum += ALPHA*A*B

		__epi_2xf32 vaalpha01 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+(k+1)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb1, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)*lda+k], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha02 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)*lda+(k+1)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb1, gvl); // sum += ALPHA*A*B


               __epi_2xf32 vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+k], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha03 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+(k+1)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb1, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha4 = __builtin_epi_vfmv_v_f_2xf32(A[(i+4)*lda+k], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha04 = __builtin_epi_vfmv_v_f_2xf32(A[(i+4)*lda+(k+1)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb1, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha5 = __builtin_epi_vfmv_v_f_2xf32(A[(i+5)*lda+k], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha05 = __builtin_epi_vfmv_v_f_2xf32( A[(i+5)*lda+(k+1)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb1, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha6 = __builtin_epi_vfmv_v_f_2xf32(A[(i+6)*lda+k], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha06 = __builtin_epi_vfmv_v_f_2xf32(A[(i+6)*lda+(k+1)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb1, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha7 = __builtin_epi_vfmv_v_f_2xf32(A[(i+7)*lda+k], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha07 = __builtin_epi_vfmv_v_f_2xf32(A[(i+7)*lda+(k+1)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb1, gvl); // sum += ALPHA*A*B


               __epi_2xf32 vaalpha8 = __builtin_epi_vfmv_v_f_2xf32(A[(i+8)*lda+k], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha08 = __builtin_epi_vfmv_v_f_2xf32(A[(i+8)*lda+(k+1)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb1, gvl); // sum += ALPHA*A*B


               __epi_2xf32 vaalpha9= __builtin_epi_vfmv_v_f_2xf32(A[(i+9)*lda+k], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha09= __builtin_epi_vfmv_v_f_2xf32(A[(i+9)*lda+(k+1)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb1, gvl); // sum += ALPHA*A*B


               __epi_2xf32 vaalpha10 = __builtin_epi_vfmv_v_f_2xf32(A[(i+10)*lda+k], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha010 = __builtin_epi_vfmv_v_f_2xf32(A[(i+10)*lda+(k+1)], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb1, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha11 = __builtin_epi_vfmv_v_f_2xf32(A[(i+11)*lda+k], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha011 = __builtin_epi_vfmv_v_f_2xf32(A[(i+11)*lda+(k+1)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb1, gvl); // sum += ALPHA*A*B


               __epi_2xf32 vaalpha12 = __builtin_epi_vfmv_v_f_2xf32(A[(i+12)*lda+k], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha012 = __builtin_epi_vfmv_v_f_2xf32(A[(i+12)*lda+(k+1)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb1, gvl); // sum += ALPHA*A*B


               __epi_2xf32 vaalpha13 = __builtin_epi_vfmv_v_f_2xf32(A[(i+13)*lda+k], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha013 = __builtin_epi_vfmv_v_f_2xf32(A[(i+13)*lda+(k+1)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb1, gvl); // sum += ALPHA*A*B


               __epi_2xf32 vaalpha14 = __builtin_epi_vfmv_v_f_2xf32(A[(i+14)*lda+k], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha014 = __builtin_epi_vfmv_v_f_2xf32(A[(i+14)*lda+(k+1)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb1, gvl); // sum += ALPHA*A*B


               __epi_2xf32 vaalpha15 = __builtin_epi_vfmv_v_f_2xf32(A[(i+15)*lda+k], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha015 = __builtin_epi_vfmv_v_f_2xf32(A[(i+15)*lda+(k+1)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb1, gvl); // sum += ALPHA*A*B
		  //-----


		/* unroll 4*/

                vaalpha = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+(k+2)], gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb2, gvl); // sum += ALPHA*A*B

                vaalpha0 = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+(k+3)], gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha0, vb3, gvl); // sum += ALPHA*A*B

               vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+(k+2)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb2, gvl); // sum += ALPHA*A*B

               vaalpha01 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+(k+3)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb3, gvl); // sum += ALPHA*A*B

               vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)*lda+(k+2)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb2, gvl); // sum += ALPHA*A*B

                vaalpha02 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)*lda+(k+3)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb3, gvl); // sum += ALPHA*A*B


                vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+(k+2)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb2, gvl); // sum += ALPHA*A*B

                vaalpha03 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+(k+3)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb3, gvl); // sum += ALPHA*A*B

               vaalpha4 = __builtin_epi_vfmv_v_f_2xf32( A[(i+4)*lda+(k+2)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb2, gvl); // sum += ALPHA*A*B

               vaalpha04 = __builtin_epi_vfmv_v_f_2xf32(A[(i+4)*lda+(k+3)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb3, gvl); // sum += ALPHA*A*B

                vaalpha5 = __builtin_epi_vfmv_v_f_2xf32(A[(i+5)*lda+(k+2)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb2, gvl); // sum += ALPHA*A*B

		vaalpha05 = __builtin_epi_vfmv_v_f_2xf32(A[(i+5)*lda+(k+3)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb3, gvl); // sum += ALPHA*A*B

                vaalpha6 = __builtin_epi_vfmv_v_f_2xf32(A[(i+6)*lda+(k+2)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb2, gvl); // sum += ALPHA*A*B

		vaalpha06 = __builtin_epi_vfmv_v_f_2xf32(A[(i+6)*lda+(k+3)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb3, gvl); // sum += ALPHA*A*B

                 vaalpha7 = __builtin_epi_vfmv_v_f_2xf32(A[(i+7)*lda+(k+2)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb2, gvl); // sum += ALPHA*A*B

                  vaalpha07 = __builtin_epi_vfmv_v_f_2xf32(A[(i+7)*lda+(k+3)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb3, gvl); // sum += ALPHA*A*B


                  vaalpha8 = __builtin_epi_vfmv_v_f_2xf32(A[(i+8)*lda+(k+2)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb2, gvl); // sum += ALPHA*A*B

                  vaalpha08 = __builtin_epi_vfmv_v_f_2xf32(A[(i+8)*lda+(k+3)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb3, gvl); // sum += ALPHA*A*B


                   vaalpha9= __builtin_epi_vfmv_v_f_2xf32(A[(i+9)*lda+(k+2)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb2, gvl); // sum += ALPHA*A*B

               	 vaalpha09= __builtin_epi_vfmv_v_f_2xf32(A[(i+9)*lda+(k+3)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb3, gvl); // sum += ALPHA*A*B


                 vaalpha10 = __builtin_epi_vfmv_v_f_2xf32(A[(i+10)*lda+(k+2)], gvl); // ALPHA*A
                   vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb2, gvl); // sum += ALPHA*A*B

                 vaalpha010 = __builtin_epi_vfmv_v_f_2xf32(A[(i+10)*lda+(k+3)], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb3, gvl); // sum += ALPHA*A*B

                vaalpha11 = __builtin_epi_vfmv_v_f_2xf32(A[(i+11)*lda+(k+2)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb2, gvl); // sum += ALPHA*A*B

                 vaalpha011 = __builtin_epi_vfmv_v_f_2xf32(A[(i+11)*lda+(k+3)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb3, gvl); // sum += ALPHA*A*B


                vaalpha12 = __builtin_epi_vfmv_v_f_2xf32(A[(i+12)*lda+(k+2)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb2, gvl); // sum += ALPHA*A*B

                vaalpha012 = __builtin_epi_vfmv_v_f_2xf32(A[(i+12)*lda+(k+3)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb3, gvl); // sum += ALPHA*A*B


                 vaalpha13 = __builtin_epi_vfmv_v_f_2xf32(A[(i+13)*lda+(k+2)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb2, gvl); // sum += ALPHA*A*B

		vaalpha013 = __builtin_epi_vfmv_v_f_2xf32( A[(i+13)*lda+(k+3)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb3, gvl); // sum += ALPHA*A*B


                   vaalpha14 = __builtin_epi_vfmv_v_f_2xf32(A[(i+14)*lda+(k+2)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb2, gvl); // sum += ALPHA*A*B

                 vaalpha014 = __builtin_epi_vfmv_v_f_2xf32(A[(i+14)*lda+(k+3)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb3, gvl); // sum += ALPHA*A*B


                   vaalpha15 = __builtin_epi_vfmv_v_f_2xf32(A[(i+15)*lda+(k+2)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb2, gvl); // sum += ALPHA*A*B

                   vaalpha015 = __builtin_epi_vfmv_v_f_2xf32(A[(i+15)*lda+(k+3)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb3, gvl); // sum += ALPHA*A*B

		/*******/
		// unroll 6
		 vaalpha = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+(k+4)], gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb4, gvl); // sum += ALPHA*A*B

                vaalpha0 = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+(k+5)], gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha0, vb5, gvl); // sum += ALPHA*A*B

                vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+(k+4)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb4, gvl); // sum += ALPHA*A*B

		 vaalpha01 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+(k+5)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb5, gvl); // sum += ALPHA*A*B

                 vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)*lda+(k+4)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb4, gvl); // sum += ALPHA*A*B

                vaalpha02 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)*lda+(k+5)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb5, gvl); // sum += ALPHA*A*B


                vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+(k+4)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb4, gvl); // sum += ALPHA*A*B

                vaalpha03 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+(k+5)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb5, gvl); // sum += ALPHA*A*B

                vaalpha4 = __builtin_epi_vfmv_v_f_2xf32(A[(i+4)*lda+(k+4)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb4, gvl); // sum += ALPHA*A*B

                vaalpha04 = __builtin_epi_vfmv_v_f_2xf32(A[(i+4)*lda+(k+5)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb5, gvl); // sum += ALPHA*A*B

                vaalpha5 = __builtin_epi_vfmv_v_f_2xf32(A[(i+5)*lda+(k+4)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb4, gvl); // sum += ALPHA*A*B

                vaalpha05 = __builtin_epi_vfmv_v_f_2xf32( A[(i+5)*lda+(k+5)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb5, gvl); // sum += ALPHA*A*B

                vaalpha6 = __builtin_epi_vfmv_v_f_2xf32(A[(i+6)*lda+(k+4)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb4, gvl); // sum += ALPHA*A*B

                vaalpha06 = __builtin_epi_vfmv_v_f_2xf32(A[(i+6)*lda+(k+5)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb5, gvl); // sum += ALPHA*A*B

                vaalpha7 = __builtin_epi_vfmv_v_f_2xf32(A[(i+7)*lda+(k+4)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb4, gvl); // sum += ALPHA*A*B

                vaalpha07 = __builtin_epi_vfmv_v_f_2xf32(A[(i+7)*lda+(k+5)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb5, gvl); // sum += ALPHA*A*B


                vaalpha8 = __builtin_epi_vfmv_v_f_2xf32(A[(i+8)*lda+(k+4)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb4, gvl); // sum += ALPHA*A*B

                vaalpha08 = __builtin_epi_vfmv_v_f_2xf32(A[(i+8)*lda+(k+5)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb5, gvl); // sum += ALPHA*A*B


                vaalpha9= __builtin_epi_vfmv_v_f_2xf32(A[(i+9)*lda+(k+4)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb4, gvl); // sum += ALPHA*A*B

                vaalpha09= __builtin_epi_vfmv_v_f_2xf32(A[(i+9)*lda+(k+5)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb5, gvl); // sum += ALPHA*A*B


                vaalpha10 = __builtin_epi_vfmv_v_f_2xf32(A[(i+10)*lda+(k+4)], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb4, gvl); // sum += ALPHA*A*B

                vaalpha010 = __builtin_epi_vfmv_v_f_2xf32(A[(i+10)*lda+(k+5)], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb5, gvl); // sum += ALPHA*A*B

                vaalpha11 = __builtin_epi_vfmv_v_f_2xf32(A[(i+11)*lda+(k+4)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb4, gvl); // sum += ALPHA*A*B

                vaalpha011 = __builtin_epi_vfmv_v_f_2xf32(A[(i+11)*lda+(k+5)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb5, gvl); // sum += ALPHA*A*B


                vaalpha12 = __builtin_epi_vfmv_v_f_2xf32(A[(i+12)*lda+(k+4)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb4, gvl); // sum += ALPHA*A*B

                vaalpha012 = __builtin_epi_vfmv_v_f_2xf32(A[(i+12)*lda+(k+5)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb5, gvl); // sum += ALPHA*A*B


                vaalpha13 = __builtin_epi_vfmv_v_f_2xf32(A[(i+13)*lda+(k+4)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb4, gvl); // sum += ALPHA*A*B

                vaalpha013 = __builtin_epi_vfmv_v_f_2xf32(A[(i+13)*lda+(k+5)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb5, gvl); // sum += ALPHA*A*B


                vaalpha14 = __builtin_epi_vfmv_v_f_2xf32(A[(i+14)*lda+(k+4)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb4, gvl); // sum += ALPHA*A*B

               vaalpha014 = __builtin_epi_vfmv_v_f_2xf32(A[(i+14)*lda+(k+5)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb5, gvl); // sum += ALPHA*A*B


               vaalpha15 = __builtin_epi_vfmv_v_f_2xf32(A[(i+15)*lda+(k+4)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb4, gvl); // sum += ALPHA*A*B

                vaalpha015 = __builtin_epi_vfmv_v_f_2xf32(A[(i+15)*lda+(k+5)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb5, gvl); // sum += ALPHA*A*B
		  //-----


		/* unroll 8*/

                vaalpha = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+(k+6)], gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb6, gvl); // sum += ALPHA*A*B

                vaalpha0 = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+(k+7)], gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha0, vb7, gvl); // sum += ALPHA*A*B

               vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+(k+6)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb6, gvl); // sum += ALPHA*A*B

               vaalpha01 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+(k+7)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb7, gvl); // sum += ALPHA*A*B

               vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)*lda+(k+6)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb6, gvl); // sum += ALPHA*A*B

                vaalpha02 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)*lda+(k+7)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb7, gvl); // sum += ALPHA*A*B


                vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+(k+6)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb6, gvl); // sum += ALPHA*A*B

                vaalpha03 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+(k+7)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb7, gvl); // sum += ALPHA*A*B

               vaalpha4 = __builtin_epi_vfmv_v_f_2xf32( A[(i+4)*lda+(k+6)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb6, gvl); // sum += ALPHA*A*B

               vaalpha04 = __builtin_epi_vfmv_v_f_2xf32(A[(i+4)*lda+(k+7)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb7, gvl); // sum += ALPHA*A*B

                vaalpha5 = __builtin_epi_vfmv_v_f_2xf32(A[(i+5)*lda+(k+6)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb6, gvl); // sum += ALPHA*A*B

		vaalpha05 = __builtin_epi_vfmv_v_f_2xf32(A[(i+5)*lda+(k+7)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb7, gvl); // sum += ALPHA*A*B

                vaalpha6 = __builtin_epi_vfmv_v_f_2xf32(A[(i+6)*lda+(k+6)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb6, gvl); // sum += ALPHA*A*B

		vaalpha06 = __builtin_epi_vfmv_v_f_2xf32(A[(i+6)*lda+(k+7)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb7, gvl); // sum += ALPHA*A*B

                 vaalpha7 = __builtin_epi_vfmv_v_f_2xf32(A[(i+7)*lda+(k+6)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb6, gvl); // sum += ALPHA*A*B

                  vaalpha07 = __builtin_epi_vfmv_v_f_2xf32(A[(i+7)*lda+(k+7)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb7, gvl); // sum += ALPHA*A*B


                  vaalpha8 = __builtin_epi_vfmv_v_f_2xf32(A[(i+8)*lda+(k+6)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb6, gvl); // sum += ALPHA*A*B

                  vaalpha08 = __builtin_epi_vfmv_v_f_2xf32(A[(i+8)*lda+(k+7)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb7, gvl); // sum += ALPHA*A*B


                   vaalpha9= __builtin_epi_vfmv_v_f_2xf32(A[(i+9)*lda+(k+6)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb6, gvl); // sum += ALPHA*A*B

               	 vaalpha09= __builtin_epi_vfmv_v_f_2xf32(A[(i+9)*lda+(k+7)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb7, gvl); // sum += ALPHA*A*B


                 vaalpha10 = __builtin_epi_vfmv_v_f_2xf32(A[(i+10)*lda+(k+6)], gvl); // ALPHA*A
                   vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb6, gvl); // sum += ALPHA*A*B

                 vaalpha010 = __builtin_epi_vfmv_v_f_2xf32(A[(i+10)*lda+(k+7)], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb7, gvl); // sum += ALPHA*A*B

                vaalpha11 = __builtin_epi_vfmv_v_f_2xf32(A[(i+11)*lda+(k+6)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb6, gvl); // sum += ALPHA*A*B

                 vaalpha011 = __builtin_epi_vfmv_v_f_2xf32(A[(i+11)*lda+(k+7)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb7, gvl); // sum += ALPHA*A*B


                vaalpha12 = __builtin_epi_vfmv_v_f_2xf32(A[(i+12)*lda+(k+6)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb6, gvl); // sum += ALPHA*A*B

                vaalpha012 = __builtin_epi_vfmv_v_f_2xf32(A[(i+12)*lda+(k+7)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb7, gvl); // sum += ALPHA*A*B


                 vaalpha13 = __builtin_epi_vfmv_v_f_2xf32(A[(i+13)*lda+(k+6)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb6, gvl); // sum += ALPHA*A*B

		vaalpha013 = __builtin_epi_vfmv_v_f_2xf32( A[(i+13)*lda+(k+7)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb7, gvl); // sum += ALPHA*A*B


                   vaalpha14 = __builtin_epi_vfmv_v_f_2xf32(A[(i+14)*lda+(k+6)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb6, gvl); // sum += ALPHA*A*B

                 vaalpha014 = __builtin_epi_vfmv_v_f_2xf32(A[(i+14)*lda+(k+7)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb7, gvl); // sum += ALPHA*A*B


                   vaalpha15 = __builtin_epi_vfmv_v_f_2xf32(A[(i+15)*lda+(k+6)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb6, gvl); // sum += ALPHA*A*B

                   vaalpha015 = __builtin_epi_vfmv_v_f_2xf32(A[(i+15)*lda+(k+7)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb7, gvl); // sum += ALPHA*A*B

	}
	flag++;
	}
       for ( int k1 = k; k1 < K; k1 += 1) {
		__epi_2xf32 vb = __builtin_epi_vload_2xf32(&B[k1*ldb+j], gvl);

                register float alpha = ALPHA * A[i*lda+k1];
               __epi_2xf32 vaalpha = __builtin_epi_vfmv_v_f_2xf32(alpha, gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb, gvl); // sum += ALPHA*A*B
                register float alpha1 = ALPHA * A[(i+1)*lda+k1];
               __epi_2xf32 vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(alpha1, gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl); // sum += ALPHA*A*B
                register float alpha2 = ALPHA * A[(i+2)*lda+k1];
               __epi_2xf32 vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(alpha2, gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl); // sum += ALPHA*A*B
                register float alpha3 = ALPHA * A[(i+3)*lda+k1];
               __epi_2xf32 vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(alpha3, gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl); // sum += ALPHA*A*B
                register float alpha4 = ALPHA * A[(i+4)*lda+k1];
               __epi_2xf32 vaalpha4 = __builtin_epi_vfmv_v_f_2xf32(alpha4, gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb, gvl); // sum += ALPHA*A*B
                register float alpha5 = ALPHA * A[(i+5)*lda+k1];
               __epi_2xf32 vaalpha5 = __builtin_epi_vfmv_v_f_2xf32(alpha5, gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb, gvl); // sum += ALPHA*A*B
                register float alpha6 = ALPHA * A[(i+6)*lda+k1];
               __epi_2xf32 vaalpha6 = __builtin_epi_vfmv_v_f_2xf32(alpha6, gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb, gvl); // sum += ALPHA*A*B
                register float alpha7 = ALPHA * A[(i+7)*lda+k1];
               __epi_2xf32 vaalpha7 = __builtin_epi_vfmv_v_f_2xf32(alpha7, gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb, gvl); // sum += ALPHA*A*B
                register float alpha8 = ALPHA * A[(i+8)*lda+k1];
               __epi_2xf32 vaalpha8 = __builtin_epi_vfmv_v_f_2xf32(alpha8, gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb, gvl); // sum += ALPHA*A*B
                register float alpha9 = ALPHA * A[(i+9)*lda+k1];
               __epi_2xf32 vaalpha9= __builtin_epi_vfmv_v_f_2xf32(alpha9, gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb, gvl); // sum += ALPHA*A*B
                register float alpha10 = ALPHA * A[(i+10)*lda+k1];
               __epi_2xf32 vaalpha10 = __builtin_epi_vfmv_v_f_2xf32(alpha10, gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb, gvl); // sum += ALPHA*A*B
		register float alpha11 = ALPHA * A[(i+11)*lda+k1];
               __epi_2xf32 vaalpha11 = __builtin_epi_vfmv_v_f_2xf32(alpha11, gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb, gvl); // sum += ALPHA*A*B
                register float alpha12 = ALPHA * A[(i+12)*lda+k1];
               __epi_2xf32 vaalpha12 = __builtin_epi_vfmv_v_f_2xf32(alpha12, gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb, gvl); // sum += ALPHA*A*B
                register float alpha13 = ALPHA * A[(i+13)*lda+k1];
               __epi_2xf32 vaalpha13 = __builtin_epi_vfmv_v_f_2xf32(alpha13, gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb, gvl); // sum += ALPHA*A*B
                register float alpha14 = ALPHA * A[(i+14)*lda+k1];
               __epi_2xf32 vaalpha14 = __builtin_epi_vfmv_v_f_2xf32(alpha14, gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb, gvl); // sum += ALPHA*A*B
                register float alpha15 = ALPHA * A[(i+15)*lda+k1];
               __epi_2xf32 vaalpha15 = __builtin_epi_vfmv_v_f_2xf32(alpha15, gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb, gvl); // sum += ALPHA*A*B

	}
                __builtin_epi_vstore_2xf32(&C[i*ldc+j], vc, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+1)*ldc+j], vc1, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+2)*ldc+j], vc2, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+3)*ldc+j], vc3, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+4)*ldc+j], vc4, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+5)*ldc+j], vc5, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+6)*ldc+j], vc6, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+7)*ldc+j], vc7, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+8)*ldc+j], vc8, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+9)*ldc+j], vc9, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+10)*ldc+j], vc10, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+11)*ldc+j], vc11, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+12)*ldc+j], vc12, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+13)*ldc+j], vc13, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+14)*ldc+j], vc14, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+15)*ldc+j], vc15, gvl);
		//
        }
    j += gvl;
     }}

  int i_left=i;
  for (int j = 0; j < N; ) {
     __epi_2xf32  vaalpha, vaalpha1, vaalpha2, vaalpha3, vc, vc1, vc2, vc3,vb;
     float alpha1, alpha2, alpha3, alpha;

     unsigned long int gvl = __builtin_epi_vsetvl(N-j, __epi_e32, __epi_m1);
     for (i=i_left; i < M; i += 4) {    // change according to unroll degree
        vc= __builtin_epi_vload_2xf32(&C[i*ldc+j], gvl);
       if (i+1 < M) { vc1= __builtin_epi_vload_2xf32(&C[(i+1)*ldc+j], gvl);}
       if (i+2 < M) { vc2= __builtin_epi_vload_2xf32(&C[(i+2)*ldc+j], gvl);}
       if (i+3 < M) {vc3= __builtin_epi_vload_2xf32(&C[(i+3)*ldc+j], gvl);}

        for (int k = 0; k < K; k ++) {
                vaalpha = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+k], gvl); // ALPHA*A
               if (i+1 < M) { vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+k], gvl);} // ALPHA*A
               if (i+2 < M) { vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)*lda+k], gvl);} // ALPHA*A
               if (i+3 < M) { vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+k], gvl);} // ALPHA*A
                vb = __builtin_epi_vload_2xf32(&B[k*ldb+j], gvl);
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb, gvl); // sum += ALPHA*A*B
                  if (i+1 < M) {vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl);} // sum += ALPHA*A*B
                  if (i+2 < M) {vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl);} // sum += ALPHA*A*B
                  if (i+3 < M) {vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl);}// sum += ALPHA*A*B
        }
          __builtin_epi_vstore_2xf32(&C[i*ldc+j], vc, gvl);
          if (i+1 < M)      {__builtin_epi_vstore_2xf32(&C[(i+1)*ldc+j], vc1, gvl);}
          if (i+2 < M)      {__builtin_epi_vstore_2xf32(&C[(i+2)*ldc+j], vc2, gvl);}
          if (i+3 < M)      {__builtin_epi_vstore_2xf32(&C[(i+3)*ldc+j], vc3, gvl);}
     }
     j += gvl;
  }
}

/***** new kernel end*////



/***********************3. loop interchange with manual vectorization with ALPHA=1 double buffer ****************/
/* Manual vectorization with loop interchange + loop unrolling*/
void gemm_nn_noalpha_doublebuff(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
  int i=0,j=0,k=0;
  long gvl;
  if(M>15){
  for ( j = 0; j < N; ) {
      gvl = __builtin_epi_vsetvl(N-j, __epi_e32, __epi_m1); 
     for (i = 0; i < M-15; i += 16) {                        
        __epi_2xf32 vb,vb1,vb2,vb3,vb4,vb5,vb6,vb7,vc, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vc10, vc11, vc12, vc13, vc14, vc15,vc16,vc17,vc18,vc19,vc20,vc21,vc22,vc23;
        
        vc= __builtin_epi_vload_2xf32(&C[i*ldc+j], gvl);
        vc1= __builtin_epi_vload_2xf32(&C[(i+1)*ldc+j], gvl);
        vc2= __builtin_epi_vload_2xf32(&C[(i+2)*ldc+j], gvl);
        vc3= __builtin_epi_vload_2xf32(&C[(i+3)*ldc+j], gvl);
        vc4= __builtin_epi_vload_2xf32(&C[(i+4)*ldc+j], gvl);
        vc5= __builtin_epi_vload_2xf32(&C[(i+5)*ldc+j], gvl);
        vc6= __builtin_epi_vload_2xf32(&C[(i+6)*ldc+j], gvl);
        vc7= __builtin_epi_vload_2xf32(&C[(i+7)*ldc+j], gvl);
        vc8= __builtin_epi_vload_2xf32(&C[(i+8)*ldc+j], gvl);
        vc9= __builtin_epi_vload_2xf32(&C[(i+9)*ldc+j], gvl);
        vc10= __builtin_epi_vload_2xf32(&C[(i+10)*ldc+j], gvl);
        vc11= __builtin_epi_vload_2xf32(&C[(i+11)*ldc+j], gvl);
        vc12= __builtin_epi_vload_2xf32(&C[(i+12)*ldc+j], gvl);
        vc13= __builtin_epi_vload_2xf32(&C[(i+13)*ldc+j], gvl);
        vc14= __builtin_epi_vload_2xf32(&C[(i+14)*ldc+j], gvl);
        vc15= __builtin_epi_vload_2xf32(&C[(i+15)*ldc+j], gvl);
	// double buffer scheme implementation -start 
        int flag=0;
	for ( k = 0; k < K-3; k +=4) {
		 if (flag==0){
                vb = __builtin_epi_vload_2xf32(&B[k*ldb+j], gvl);
                vb1 = __builtin_epi_vload_2xf32(&B[(k+1)*ldb+j], gvl);
                vb2 = __builtin_epi_vload_2xf32(&B[(k+2)*ldb+j], gvl);
                vb3 = __builtin_epi_vload_2xf32(&B[(k+3)*ldb+j], gvl);
                 vb4 = __builtin_epi_vload_2xf32(&B[(k+4)*ldb+j], gvl);
                 vb5 = __builtin_epi_vload_2xf32(&B[(k+5)*ldb+j], gvl);
                 vb6 = __builtin_epi_vload_2xf32(&B[(k+6)*ldb+j], gvl);
                 vb7 = __builtin_epi_vload_2xf32(&B[(k+7)*ldb+j], gvl);
                }
                else
                {
			if(flag & 1) 
			{
			   if(k<K-4)
			   {
                 		vb = __builtin_epi_vload_2xf32(&B[(k+4)*ldb+j], gvl);
                 		vb1 = __builtin_epi_vload_2xf32(&B[(k+5)*ldb+j], gvl);
                 		vb2 = __builtin_epi_vload_2xf32(&B[(k+6)*ldb+j], gvl);
                 		vb3 = __builtin_epi_vload_2xf32(&B[(k+7)*ldb+j], gvl);
                           }
                        }
                        else
			{
			    if(k<K-4)
                           {
                                vb4 = __builtin_epi_vload_2xf32(&B[(k+4)*ldb+j], gvl);
                                vb5 = __builtin_epi_vload_2xf32(&B[(k+5)*ldb+j], gvl);
                                vb6 = __builtin_epi_vload_2xf32(&B[(k+6)*ldb+j], gvl);
                                vb7 = __builtin_epi_vload_2xf32(&B[(k+7)*ldb+j], gvl);
                           }
                	}
		}
		
		//double buffer scheme implementation - end
		if(flag & 1) 
		{

               __epi_2xf32 vaalpha = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+k], gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb4, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha0 = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+(k+1)], gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha0, vb5, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+k], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb4, gvl); // sum += ALPHA*A*B
               
		__epi_2xf32 vaalpha01 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+(k+1)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb5, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)*lda+k], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb4, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha02 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)*lda+(k+1)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb5, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+k], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb4, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha03 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+(k+1)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb5, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha4 = __builtin_epi_vfmv_v_f_2xf32(A[(i+4)*lda+k], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb4, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha04 = __builtin_epi_vfmv_v_f_2xf32(A[(i+4)*lda+(k+1)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb5, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha5 = __builtin_epi_vfmv_v_f_2xf32(A[(i+5)*lda+k], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb4, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha05 = __builtin_epi_vfmv_v_f_2xf32( A[(i+5)*lda+(k+1)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb5, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha6 = __builtin_epi_vfmv_v_f_2xf32(A[(i+6)*lda+k], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb4, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha06 = __builtin_epi_vfmv_v_f_2xf32(A[(i+6)*lda+(k+1)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb5, gvl); // sum += ALPHA*A*B
               
               __epi_2xf32 vaalpha7 = __builtin_epi_vfmv_v_f_2xf32(A[(i+7)*lda+k], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb4, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha07 = __builtin_epi_vfmv_v_f_2xf32(A[(i+7)*lda+(k+1)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb5, gvl); // sum += ALPHA*A*B
               

               __epi_2xf32 vaalpha8 = __builtin_epi_vfmv_v_f_2xf32(A[(i+8)*lda+k], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb4, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha08 = __builtin_epi_vfmv_v_f_2xf32(A[(i+8)*lda+(k+1)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb5, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha9= __builtin_epi_vfmv_v_f_2xf32(A[(i+9)*lda+k], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb4, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha09= __builtin_epi_vfmv_v_f_2xf32(A[(i+9)*lda+(k+1)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb5, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha10 = __builtin_epi_vfmv_v_f_2xf32(A[(i+10)*lda+k], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb4, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha010 = __builtin_epi_vfmv_v_f_2xf32(A[(i+10)*lda+(k+1)], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb5, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha11 = __builtin_epi_vfmv_v_f_2xf32(A[(i+11)*lda+k], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb4, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha011 = __builtin_epi_vfmv_v_f_2xf32(A[(i+11)*lda+(k+1)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb5, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha12 = __builtin_epi_vfmv_v_f_2xf32(A[(i+12)*lda+k], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb4, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha012 = __builtin_epi_vfmv_v_f_2xf32(A[(i+12)*lda+(k+1)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb5, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha13 = __builtin_epi_vfmv_v_f_2xf32(A[(i+13)*lda+k], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb4, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha013 = __builtin_epi_vfmv_v_f_2xf32(A[(i+13)*lda+(k+1)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb5, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha14 = __builtin_epi_vfmv_v_f_2xf32(A[(i+14)*lda+k], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb4, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha014 = __builtin_epi_vfmv_v_f_2xf32(A[(i+14)*lda+(k+1)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb5, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha15 = __builtin_epi_vfmv_v_f_2xf32(A[(i+15)*lda+k], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb4, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha015 = __builtin_epi_vfmv_v_f_2xf32(A[(i+15)*lda+(k+1)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb5, gvl); // sum += ALPHA*A*B
		  //-----


		/* unroll 4*/

                vaalpha = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+(k+2)], gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb6, gvl); // sum += ALPHA*A*B
                
                vaalpha0 = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+(k+3)], gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha0, vb7, gvl); // sum += ALPHA*A*B

               vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+(k+2)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb6, gvl); // sum += ALPHA*A*B
               
               vaalpha01 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+(k+3)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb7, gvl); // sum += ALPHA*A*B
                
               vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)*lda+(k+2)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb6, gvl); // sum += ALPHA*A*B
	
                vaalpha02 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)*lda+(k+3)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb7, gvl); // sum += ALPHA*A*B
                

                vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+(k+2)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb6, gvl); // sum += ALPHA*A*B
		
                vaalpha03 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+(k+3)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb7, gvl); // sum += ALPHA*A*B
                
               vaalpha4 = __builtin_epi_vfmv_v_f_2xf32( A[(i+4)*lda+(k+2)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb6, gvl); // sum += ALPHA*A*B
	
               vaalpha04 = __builtin_epi_vfmv_v_f_2xf32(A[(i+4)*lda+(k+3)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb7, gvl); // sum += ALPHA*A*B
                
                vaalpha5 = __builtin_epi_vfmv_v_f_2xf32(A[(i+5)*lda+(k+2)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb6, gvl); // sum += ALPHA*A*B
               
		vaalpha05 = __builtin_epi_vfmv_v_f_2xf32(A[(i+5)*lda+(k+3)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb7, gvl); // sum += ALPHA*A*B
                
                vaalpha6 = __builtin_epi_vfmv_v_f_2xf32(A[(i+6)*lda+(k+2)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb6, gvl); // sum += ALPHA*A*B
                 
		vaalpha06 = __builtin_epi_vfmv_v_f_2xf32(A[(i+6)*lda+(k+3)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb7, gvl); // sum += ALPHA*A*B
               
                 vaalpha7 = __builtin_epi_vfmv_v_f_2xf32(A[(i+7)*lda+(k+2)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb6, gvl); // sum += ALPHA*A*B
		 
                  vaalpha07 = __builtin_epi_vfmv_v_f_2xf32(A[(i+7)*lda+(k+3)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb7, gvl); // sum += ALPHA*A*B
               

                  vaalpha8 = __builtin_epi_vfmv_v_f_2xf32(A[(i+8)*lda+(k+2)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb6, gvl); // sum += ALPHA*A*B
		  
                  vaalpha08 = __builtin_epi_vfmv_v_f_2xf32(A[(i+8)*lda+(k+3)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb7, gvl); // sum += ALPHA*A*B
                

                   vaalpha9= __builtin_epi_vfmv_v_f_2xf32(A[(i+9)*lda+(k+2)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb6, gvl); // sum += ALPHA*A*B
		  
               	 vaalpha09= __builtin_epi_vfmv_v_f_2xf32(A[(i+9)*lda+(k+3)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb7, gvl); // sum += ALPHA*A*B
                

                 vaalpha10 = __builtin_epi_vfmv_v_f_2xf32(A[(i+10)*lda+(k+2)], gvl); // ALPHA*A
                   vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb6, gvl); // sum += ALPHA*A*B
		 
                 vaalpha010 = __builtin_epi_vfmv_v_f_2xf32(A[(i+10)*lda+(k+3)], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb7, gvl); // sum += ALPHA*A*B
                
                vaalpha11 = __builtin_epi_vfmv_v_f_2xf32(A[(i+11)*lda+(k+2)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb6, gvl); // sum += ALPHA*A*B
		 
                 vaalpha011 = __builtin_epi_vfmv_v_f_2xf32(A[(i+11)*lda+(k+3)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb7, gvl); // sum += ALPHA*A*B
                

                vaalpha12 = __builtin_epi_vfmv_v_f_2xf32(A[(i+12)*lda+(k+2)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb6, gvl); // sum += ALPHA*A*B
		
                vaalpha012 = __builtin_epi_vfmv_v_f_2xf32(A[(i+12)*lda+(k+3)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb7, gvl); // sum += ALPHA*A*B
                

                 vaalpha13 = __builtin_epi_vfmv_v_f_2xf32(A[(i+13)*lda+(k+2)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb6, gvl); // sum += ALPHA*A*B
                 
		vaalpha013 = __builtin_epi_vfmv_v_f_2xf32( A[(i+13)*lda+(k+3)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb7, gvl); // sum += ALPHA*A*B
                

                   vaalpha14 = __builtin_epi_vfmv_v_f_2xf32(A[(i+14)*lda+(k+2)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb6, gvl); // sum += ALPHA*A*B
		  
                 vaalpha014 = __builtin_epi_vfmv_v_f_2xf32(A[(i+14)*lda+(k+3)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb7, gvl); // sum += ALPHA*A*B
                

                   vaalpha15 = __builtin_epi_vfmv_v_f_2xf32(A[(i+15)*lda+(k+2)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb6, gvl); // sum += ALPHA*A*B
		   
                   vaalpha015 = __builtin_epi_vfmv_v_f_2xf32(A[(i+15)*lda+(k+3)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb7, gvl); // sum += ALPHA*A*B
	
	}
	else
	{

               __epi_2xf32 vaalpha = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+k], gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha0 = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+(k+1)], gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha0, vb1, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+k], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl); // sum += ALPHA*A*B
               
		__epi_2xf32 vaalpha01 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+(k+1)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb1, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)*lda+k], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha02 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)*lda+(k+1)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb1, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+k], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha03 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+(k+1)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb1, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha4 = __builtin_epi_vfmv_v_f_2xf32(A[(i+4)*lda+k], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha04 = __builtin_epi_vfmv_v_f_2xf32(A[(i+4)*lda+(k+1)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb1, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha5 = __builtin_epi_vfmv_v_f_2xf32(A[(i+5)*lda+k], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha05 = __builtin_epi_vfmv_v_f_2xf32( A[(i+5)*lda+(k+1)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb1, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha6 = __builtin_epi_vfmv_v_f_2xf32(A[(i+6)*lda+k], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha06 = __builtin_epi_vfmv_v_f_2xf32(A[(i+6)*lda+(k+1)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb1, gvl); // sum += ALPHA*A*B
               
               __epi_2xf32 vaalpha7 = __builtin_epi_vfmv_v_f_2xf32(A[(i+7)*lda+k], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha07 = __builtin_epi_vfmv_v_f_2xf32(A[(i+7)*lda+(k+1)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb1, gvl); // sum += ALPHA*A*B
               

               __epi_2xf32 vaalpha8 = __builtin_epi_vfmv_v_f_2xf32(A[(i+8)*lda+k], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha08 = __builtin_epi_vfmv_v_f_2xf32(A[(i+8)*lda+(k+1)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb1, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha9= __builtin_epi_vfmv_v_f_2xf32(A[(i+9)*lda+k], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha09= __builtin_epi_vfmv_v_f_2xf32(A[(i+9)*lda+(k+1)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb1, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha10 = __builtin_epi_vfmv_v_f_2xf32(A[(i+10)*lda+k], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha010 = __builtin_epi_vfmv_v_f_2xf32(A[(i+10)*lda+(k+1)], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb1, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha11 = __builtin_epi_vfmv_v_f_2xf32(A[(i+11)*lda+k], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha011 = __builtin_epi_vfmv_v_f_2xf32(A[(i+11)*lda+(k+1)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb1, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha12 = __builtin_epi_vfmv_v_f_2xf32(A[(i+12)*lda+k], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha012 = __builtin_epi_vfmv_v_f_2xf32(A[(i+12)*lda+(k+1)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb1, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha13 = __builtin_epi_vfmv_v_f_2xf32(A[(i+13)*lda+k], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha013 = __builtin_epi_vfmv_v_f_2xf32(A[(i+13)*lda+(k+1)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb1, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha14 = __builtin_epi_vfmv_v_f_2xf32(A[(i+14)*lda+k], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha014 = __builtin_epi_vfmv_v_f_2xf32(A[(i+14)*lda+(k+1)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb1, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha15 = __builtin_epi_vfmv_v_f_2xf32(A[(i+15)*lda+k], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha015 = __builtin_epi_vfmv_v_f_2xf32(A[(i+15)*lda+(k+1)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb1, gvl); // sum += ALPHA*A*B
		  //-----


		/* unroll 4*/

                vaalpha = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+(k+2)], gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb2, gvl); // sum += ALPHA*A*B
                
                vaalpha0 = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+(k+3)], gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha0, vb3, gvl); // sum += ALPHA*A*B

               vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+(k+2)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb2, gvl); // sum += ALPHA*A*B
               
               vaalpha01 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+(k+3)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb3, gvl); // sum += ALPHA*A*B
                
               vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)*lda+(k+2)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb2, gvl); // sum += ALPHA*A*B
	
                vaalpha02 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)*lda+(k+3)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb3, gvl); // sum += ALPHA*A*B
                

                vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+(k+2)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb2, gvl); // sum += ALPHA*A*B
		
                vaalpha03 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+(k+3)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb3, gvl); // sum += ALPHA*A*B
                
               vaalpha4 = __builtin_epi_vfmv_v_f_2xf32( A[(i+4)*lda+(k+2)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb2, gvl); // sum += ALPHA*A*B
	
               vaalpha04 = __builtin_epi_vfmv_v_f_2xf32(A[(i+4)*lda+(k+3)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb3, gvl); // sum += ALPHA*A*B
                
                vaalpha5 = __builtin_epi_vfmv_v_f_2xf32(A[(i+5)*lda+(k+2)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb2, gvl); // sum += ALPHA*A*B
               
		vaalpha05 = __builtin_epi_vfmv_v_f_2xf32(A[(i+5)*lda+(k+3)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb3, gvl); // sum += ALPHA*A*B
                
                vaalpha6 = __builtin_epi_vfmv_v_f_2xf32(A[(i+6)*lda+(k+2)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb2, gvl); // sum += ALPHA*A*B
                 
		vaalpha06 = __builtin_epi_vfmv_v_f_2xf32(A[(i+6)*lda+(k+3)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb3, gvl); // sum += ALPHA*A*B
               
                 vaalpha7 = __builtin_epi_vfmv_v_f_2xf32(A[(i+7)*lda+(k+2)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb2, gvl); // sum += ALPHA*A*B
		 
                  vaalpha07 = __builtin_epi_vfmv_v_f_2xf32(A[(i+7)*lda+(k+3)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb3, gvl); // sum += ALPHA*A*B
               

                  vaalpha8 = __builtin_epi_vfmv_v_f_2xf32(A[(i+8)*lda+(k+2)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb2, gvl); // sum += ALPHA*A*B
		  
                  vaalpha08 = __builtin_epi_vfmv_v_f_2xf32(A[(i+8)*lda+(k+3)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb3, gvl); // sum += ALPHA*A*B
                

                   vaalpha9= __builtin_epi_vfmv_v_f_2xf32(A[(i+9)*lda+(k+2)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb2, gvl); // sum += ALPHA*A*B
		  
               	 vaalpha09= __builtin_epi_vfmv_v_f_2xf32(A[(i+9)*lda+(k+3)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb3, gvl); // sum += ALPHA*A*B
                

                 vaalpha10 = __builtin_epi_vfmv_v_f_2xf32(A[(i+10)*lda+(k+2)], gvl); // ALPHA*A
                   vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb2, gvl); // sum += ALPHA*A*B
		 
                 vaalpha010 = __builtin_epi_vfmv_v_f_2xf32(A[(i+10)*lda+(k+3)], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb3, gvl); // sum += ALPHA*A*B
                
                vaalpha11 = __builtin_epi_vfmv_v_f_2xf32(A[(i+11)*lda+(k+2)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb2, gvl); // sum += ALPHA*A*B
		 
                 vaalpha011 = __builtin_epi_vfmv_v_f_2xf32(A[(i+11)*lda+(k+3)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb3, gvl); // sum += ALPHA*A*B
                

                vaalpha12 = __builtin_epi_vfmv_v_f_2xf32(A[(i+12)*lda+(k+2)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb2, gvl); // sum += ALPHA*A*B
		
                vaalpha012 = __builtin_epi_vfmv_v_f_2xf32(A[(i+12)*lda+(k+3)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb3, gvl); // sum += ALPHA*A*B
                

                 vaalpha13 = __builtin_epi_vfmv_v_f_2xf32(A[(i+13)*lda+(k+2)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb2, gvl); // sum += ALPHA*A*B
                 
		vaalpha013 = __builtin_epi_vfmv_v_f_2xf32( A[(i+13)*lda+(k+3)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb3, gvl); // sum += ALPHA*A*B
                

                   vaalpha14 = __builtin_epi_vfmv_v_f_2xf32(A[(i+14)*lda+(k+2)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb2, gvl); // sum += ALPHA*A*B
		  
                 vaalpha014 = __builtin_epi_vfmv_v_f_2xf32(A[(i+14)*lda+(k+3)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb3, gvl); // sum += ALPHA*A*B
                

                   vaalpha15 = __builtin_epi_vfmv_v_f_2xf32(A[(i+15)*lda+(k+2)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb2, gvl); // sum += ALPHA*A*B
		   
                   vaalpha015 = __builtin_epi_vfmv_v_f_2xf32(A[(i+15)*lda+(k+3)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb3, gvl); // sum += ALPHA*A*B

	}
	flag++;
	}
       for ( int k1 = k; k1 < K; k1 += 1) {
		__epi_2xf32 vb = __builtin_epi_vload_2xf32(&B[k1*ldb+j], gvl);

                register float alpha =  A[i*lda+k1];
               __epi_2xf32 vaalpha = __builtin_epi_vfmv_v_f_2xf32(alpha, gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb, gvl); // sum += ALPHA*A*B
                register float alpha1 =  A[(i+1)*lda+k1];
               __epi_2xf32 vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(alpha1, gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl); // sum += ALPHA*A*B
                register float alpha2 =  A[(i+2)*lda+k1];
               __epi_2xf32 vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(alpha2, gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl); // sum += ALPHA*A*B
                register float alpha3 =  A[(i+3)*lda+k1];
               __epi_2xf32 vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(alpha3, gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl); // sum += ALPHA*A*B
                register float alpha4 =  A[(i+4)*lda+k1];
               __epi_2xf32 vaalpha4 = __builtin_epi_vfmv_v_f_2xf32(alpha4, gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb, gvl); // sum += ALPHA*A*B
                register float alpha5 =  A[(i+5)*lda+k1];
               __epi_2xf32 vaalpha5 = __builtin_epi_vfmv_v_f_2xf32(alpha5, gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb, gvl); // sum += ALPHA*A*B
                register float alpha6 =  A[(i+6)*lda+k1];
               __epi_2xf32 vaalpha6 = __builtin_epi_vfmv_v_f_2xf32(alpha6, gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb, gvl); // sum += ALPHA*A*B
                register float alpha7 =  A[(i+7)*lda+k1];
               __epi_2xf32 vaalpha7 = __builtin_epi_vfmv_v_f_2xf32(alpha7, gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb, gvl); // sum += ALPHA*A*B
                register float alpha8 =  A[(i+8)*lda+k1];
               __epi_2xf32 vaalpha8 = __builtin_epi_vfmv_v_f_2xf32(alpha8, gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb, gvl); // sum += ALPHA*A*B
                register float alpha9 =  A[(i+9)*lda+k1];
               __epi_2xf32 vaalpha9= __builtin_epi_vfmv_v_f_2xf32(alpha9, gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb, gvl); // sum += ALPHA*A*B
                register float alpha10 =  A[(i+10)*lda+k1];
               __epi_2xf32 vaalpha10 = __builtin_epi_vfmv_v_f_2xf32(alpha10, gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb, gvl); // sum += ALPHA*A*B
		register float alpha11 =  A[(i+11)*lda+k1];
               __epi_2xf32 vaalpha11 = __builtin_epi_vfmv_v_f_2xf32(alpha11, gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb, gvl); // sum += ALPHA*A*B
                register float alpha12 =  A[(i+12)*lda+k1];
               __epi_2xf32 vaalpha12 = __builtin_epi_vfmv_v_f_2xf32(alpha12, gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb, gvl); // sum += ALPHA*A*B
                register float alpha13 =  A[(i+13)*lda+k1];
               __epi_2xf32 vaalpha13 = __builtin_epi_vfmv_v_f_2xf32(alpha13, gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb, gvl); // sum += ALPHA*A*B
                register float alpha14 =  A[(i+14)*lda+k1];
               __epi_2xf32 vaalpha14 = __builtin_epi_vfmv_v_f_2xf32(alpha14, gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb, gvl); // sum += ALPHA*A*B
                register float alpha15 =  A[(i+15)*lda+k1];
               __epi_2xf32 vaalpha15 = __builtin_epi_vfmv_v_f_2xf32(alpha15, gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb, gvl); // sum += ALPHA*A*B

	}
                __builtin_epi_vstore_2xf32(&C[i*ldc+j], vc, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+1)*ldc+j], vc1, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+2)*ldc+j], vc2, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+3)*ldc+j], vc3, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+4)*ldc+j], vc4, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+5)*ldc+j], vc5, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+6)*ldc+j], vc6, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+7)*ldc+j], vc7, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+8)*ldc+j], vc8, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+9)*ldc+j], vc9, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+10)*ldc+j], vc10, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+11)*ldc+j], vc11, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+12)*ldc+j], vc12, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+13)*ldc+j], vc13, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+14)*ldc+j], vc14, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+15)*ldc+j], vc15, gvl);
		//
        }
    j += gvl;
     }}

  int i_left=i;
  for (int j = 0; j < N; ) {
     __epi_2xf32  vaalpha, vaalpha1, vaalpha2, vaalpha3, vc, vc1, vc2, vc3,vb;
     float alpha1, alpha2, alpha3, alpha;
   
     unsigned long int gvl = __builtin_epi_vsetvl(N-j, __epi_e32, __epi_m1);
     for (i=i_left; i < M; i += 4) {    // change according to unroll degree 
        vc= __builtin_epi_vload_2xf32(&C[i*ldc+j], gvl);
       if (i+1 < M) { vc1= __builtin_epi_vload_2xf32(&C[(i+1)*ldc+j], gvl);}
       if (i+2 < M) { vc2= __builtin_epi_vload_2xf32(&C[(i+2)*ldc+j], gvl);}
       if (i+3 < M) {vc3= __builtin_epi_vload_2xf32(&C[(i+3)*ldc+j], gvl);}

        for (int k = 0; k < K; k ++) {
                vaalpha = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+k], gvl); // ALPHA*A
               if (i+1 < M) { vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+k], gvl);} // ALPHA*A
               if (i+2 < M) { vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)*lda+k], gvl);} // ALPHA*A
               if (i+3 < M) { vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+k], gvl);} // ALPHA*A
                vb = __builtin_epi_vload_2xf32(&B[k*ldb+j], gvl);
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb, gvl); // sum += ALPHA*A*B
                  if (i+1 < M) {vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl);} // sum += ALPHA*A*B
                  if (i+2 < M) {vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl);} // sum += ALPHA*A*B
                  if (i+3 < M) {vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl);}// sum += ALPHA*A*B
        }
          __builtin_epi_vstore_2xf32(&C[i*ldc+j], vc, gvl);
          if (i+1 < M)      {__builtin_epi_vstore_2xf32(&C[(i+1)*ldc+j], vc1, gvl);}
          if (i+2 < M)      {__builtin_epi_vstore_2xf32(&C[(i+2)*ldc+j], vc2, gvl);}
          if (i+3 < M)      {__builtin_epi_vstore_2xf32(&C[(i+3)*ldc+j], vc3, gvl);}
     }
     j += gvl;
  }
}


/***********************3. loop interchange with manual vectorization ALPHA!=1 with double buffer****************/
/* Manual vectorization with loop interchange + loop unrolling*/
void gemm_nn_doublebuff(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
  int i=0,j=0,k=0;
  long gvl;
  if(M>15){
  for ( j = 0; j < N; ) {
      gvl = __builtin_epi_vsetvl(N-j, __epi_e32, __epi_m1); 
     for (i = 0; i < M-15; i += 16) {                        
        __epi_2xf32 vb,vb1,vb2,vb3,vb4, vb5, vb6, vb7, vc, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vc10, vc11, vc12, vc13, vc14, vc15,vc16,vc17,vc18,vc19,vc20,vc21,vc22,vc23;
        
        vc= __builtin_epi_vload_2xf32(&C[i*ldc+j], gvl);
        vc1= __builtin_epi_vload_2xf32(&C[(i+1)*ldc+j], gvl);
        vc2= __builtin_epi_vload_2xf32(&C[(i+2)*ldc+j], gvl);
        vc3= __builtin_epi_vload_2xf32(&C[(i+3)*ldc+j], gvl);
        vc4= __builtin_epi_vload_2xf32(&C[(i+4)*ldc+j], gvl);
        vc5= __builtin_epi_vload_2xf32(&C[(i+5)*ldc+j], gvl);
        vc6= __builtin_epi_vload_2xf32(&C[(i+6)*ldc+j], gvl);
        vc7= __builtin_epi_vload_2xf32(&C[(i+7)*ldc+j], gvl);
        vc8= __builtin_epi_vload_2xf32(&C[(i+8)*ldc+j], gvl);
        vc9= __builtin_epi_vload_2xf32(&C[(i+9)*ldc+j], gvl);
        vc10= __builtin_epi_vload_2xf32(&C[(i+10)*ldc+j], gvl);
        vc11= __builtin_epi_vload_2xf32(&C[(i+11)*ldc+j], gvl);
        vc12= __builtin_epi_vload_2xf32(&C[(i+12)*ldc+j], gvl);
        vc13= __builtin_epi_vload_2xf32(&C[(i+13)*ldc+j], gvl);
        vc14= __builtin_epi_vload_2xf32(&C[(i+14)*ldc+j], gvl);
        vc15= __builtin_epi_vload_2xf32(&C[(i+15)*ldc+j], gvl);
	//
	int flag=0;
        for ( k = 0; k < K-3; k +=4) {
		// double buffer scheme implementation -start 
		  if (flag==0){
                vb = __builtin_epi_vload_2xf32(&B[k*ldb+j], gvl);
                vb1 = __builtin_epi_vload_2xf32(&B[(k+1)*ldb+j], gvl);
                vb2 = __builtin_epi_vload_2xf32(&B[(k+2)*ldb+j], gvl);
                vb3 = __builtin_epi_vload_2xf32(&B[(k+3)*ldb+j], gvl);
                 vb4 = __builtin_epi_vload_2xf32(&B[(k+4)*ldb+j], gvl);
                 vb5 = __builtin_epi_vload_2xf32(&B[(k+5)*ldb+j], gvl);
                 vb6 = __builtin_epi_vload_2xf32(&B[(k+6)*ldb+j], gvl);
                 vb7 = __builtin_epi_vload_2xf32(&B[(k+7)*ldb+j], gvl);
                }
                else
                {
                        if(flag & 1)
                        {
                           if(k<K-4)
                           {
                                vb = __builtin_epi_vload_2xf32(&B[(k+4)*ldb+j], gvl);
                                vb1 = __builtin_epi_vload_2xf32(&B[(k+5)*ldb+j], gvl);
                                vb2 = __builtin_epi_vload_2xf32(&B[(k+6)*ldb+j], gvl);
                                vb3 = __builtin_epi_vload_2xf32(&B[(k+7)*ldb+j], gvl);
                           }
                        }
                        else
                        {
                            if(k<K-4)
                           {
                                vb4 = __builtin_epi_vload_2xf32(&B[(k+4)*ldb+j], gvl);
                                vb5 = __builtin_epi_vload_2xf32(&B[(k+5)*ldb+j], gvl);
                                vb6 = __builtin_epi_vload_2xf32(&B[(k+6)*ldb+j], gvl);
                                vb7 = __builtin_epi_vload_2xf32(&B[(k+7)*ldb+j], gvl);
                           }
                        }
                }

		// double buffer scheme implementation - end

		if(flag & 1)
		{
                register float alpha = ALPHA * A[i*lda+k];
               __epi_2xf32 vaalpha = __builtin_epi_vfmv_v_f_2xf32(alpha, gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb4, gvl); // sum += ALPHA*A*B
                register float alpha0 = ALPHA * A[i*lda+(k+1)];
               __epi_2xf32 vaalpha0 = __builtin_epi_vfmv_v_f_2xf32(alpha0, gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha0, vb5, gvl); // sum += ALPHA*A*B

                register float alpha1 = ALPHA * A[(i+1)*lda+k];
               __epi_2xf32 vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(alpha1, gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb4, gvl); // sum += ALPHA*A*B
                register float alpha01 = ALPHA * A[(i+1)*lda+(k+1)];
               __epi_2xf32 vaalpha01 = __builtin_epi_vfmv_v_f_2xf32(alpha01, gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb5, gvl); // sum += ALPHA*A*B
                
		register float alpha2 = ALPHA * A[(i+2)*lda+k];
               __epi_2xf32 vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(alpha2, gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb4, gvl); // sum += ALPHA*A*B
		register float alpha02 = ALPHA * A[(i+2)*lda+(k+1)];
               __epi_2xf32 vaalpha02 = __builtin_epi_vfmv_v_f_2xf32(alpha02, gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb5, gvl); // sum += ALPHA*A*B
                

		register float alpha3 = ALPHA * A[(i+3)*lda+k];
               __epi_2xf32 vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(alpha3, gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb4, gvl); // sum += ALPHA*A*B
		register float alpha03 = ALPHA * A[(i+3)*lda+(k+1)];
               __epi_2xf32 vaalpha03 = __builtin_epi_vfmv_v_f_2xf32(alpha03, gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb5, gvl); // sum += ALPHA*A*B
                
		register float alpha4 = ALPHA * A[(i+4)*lda+k];
               __epi_2xf32 vaalpha4 = __builtin_epi_vfmv_v_f_2xf32(alpha4, gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb4, gvl); // sum += ALPHA*A*B
		register float alpha04 = ALPHA * A[(i+4)*lda+(k+1)];
               __epi_2xf32 vaalpha04 = __builtin_epi_vfmv_v_f_2xf32(alpha04, gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb5, gvl); // sum += ALPHA*A*B
                
		register float alpha5 = ALPHA * A[(i+5)*lda+k];
               __epi_2xf32 vaalpha5 = __builtin_epi_vfmv_v_f_2xf32(alpha5, gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb4, gvl); // sum += ALPHA*A*B
		register float alpha05 = ALPHA * A[(i+5)*lda+(k+1)];
               __epi_2xf32 vaalpha05 = __builtin_epi_vfmv_v_f_2xf32(alpha05, gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb5, gvl); // sum += ALPHA*A*B
                
		register float alpha6 = ALPHA * A[(i+6)*lda+k];
               __epi_2xf32 vaalpha6 = __builtin_epi_vfmv_v_f_2xf32(alpha6, gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb4, gvl); // sum += ALPHA*A*B
		register float alpha06 = ALPHA * A[(i+6)*lda+(k+1)];
               __epi_2xf32 vaalpha06 = __builtin_epi_vfmv_v_f_2xf32(alpha06, gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb5, gvl); // sum += ALPHA*A*B
               
		 register float alpha7 = ALPHA * A[(i+7)*lda+k];
               __epi_2xf32 vaalpha7 = __builtin_epi_vfmv_v_f_2xf32(alpha7, gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb4, gvl); // sum += ALPHA*A*B
		 register float alpha07 = ALPHA * A[(i+7)*lda+(k+1)];
               __epi_2xf32 vaalpha07 = __builtin_epi_vfmv_v_f_2xf32(alpha07, gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb5, gvl); // sum += ALPHA*A*B
               

		 register float alpha8 = ALPHA * A[(i+8)*lda+k];
               __epi_2xf32 vaalpha8 = __builtin_epi_vfmv_v_f_2xf32(alpha8, gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb4, gvl); // sum += ALPHA*A*B
		 register float alpha08 = ALPHA * A[(i+8)*lda+(k+1)];
               __epi_2xf32 vaalpha08 = __builtin_epi_vfmv_v_f_2xf32(alpha08, gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb5, gvl); // sum += ALPHA*A*B
                

		register float alpha9 = ALPHA * A[(i+9)*lda+k];
               __epi_2xf32 vaalpha9= __builtin_epi_vfmv_v_f_2xf32(alpha9, gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb4, gvl); // sum += ALPHA*A*B
		register float alpha09 = ALPHA * A[(i+9)*lda+(k+1)];
               __epi_2xf32 vaalpha09= __builtin_epi_vfmv_v_f_2xf32(alpha09, gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb5, gvl); // sum += ALPHA*A*B
                

		register float alpha10 = ALPHA * A[(i+10)*lda+k];
               __epi_2xf32 vaalpha10 = __builtin_epi_vfmv_v_f_2xf32(alpha10, gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb4, gvl); // sum += ALPHA*A*B
		register float alpha010 = ALPHA * A[(i+10)*lda+(k+1)];
               __epi_2xf32 vaalpha010 = __builtin_epi_vfmv_v_f_2xf32(alpha010, gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb5, gvl); // sum += ALPHA*A*B
                
		register float alpha11 = ALPHA * A[(i+11)*lda+k];
               __epi_2xf32 vaalpha11 = __builtin_epi_vfmv_v_f_2xf32(alpha11, gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb4, gvl); // sum += ALPHA*A*B
		register float alpha011 = ALPHA * A[(i+11)*lda+(k+1)];
               __epi_2xf32 vaalpha011 = __builtin_epi_vfmv_v_f_2xf32(alpha011, gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb5, gvl); // sum += ALPHA*A*B
                

		register float alpha12 = ALPHA * A[(i+12)*lda+k];
               __epi_2xf32 vaalpha12 = __builtin_epi_vfmv_v_f_2xf32(alpha12, gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb4, gvl); // sum += ALPHA*A*B
		register float alpha012 = ALPHA * A[(i+12)*lda+(k+1)];
               __epi_2xf32 vaalpha012 = __builtin_epi_vfmv_v_f_2xf32(alpha012, gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb5, gvl); // sum += ALPHA*A*B
                

		register float alpha13 = ALPHA * A[(i+13)*lda+k];
               __epi_2xf32 vaalpha13 = __builtin_epi_vfmv_v_f_2xf32(alpha13, gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb4, gvl); // sum += ALPHA*A*B
		register float alpha013 = ALPHA * A[(i+13)*lda+(k+1)];
               __epi_2xf32 vaalpha013 = __builtin_epi_vfmv_v_f_2xf32(alpha013, gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb5, gvl); // sum += ALPHA*A*B
                

		register float alpha14 = ALPHA * A[(i+14)*lda+k];
               __epi_2xf32 vaalpha14 = __builtin_epi_vfmv_v_f_2xf32(alpha14, gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb4, gvl); // sum += ALPHA*A*B
		register float alpha014 = ALPHA * A[(i+14)*lda+(k+1)];
               __epi_2xf32 vaalpha014 = __builtin_epi_vfmv_v_f_2xf32(alpha014, gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb5, gvl); // sum += ALPHA*A*B
                

		register float alpha15 = ALPHA * A[(i+15)*lda+k];
               __epi_2xf32 vaalpha15 = __builtin_epi_vfmv_v_f_2xf32(alpha15, gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb4, gvl); // sum += ALPHA*A*B
		register float alpha015 = ALPHA * A[(i+15)*lda+(k+1)];
               __epi_2xf32 vaalpha015 = __builtin_epi_vfmv_v_f_2xf32(alpha015, gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb5, gvl); // sum += ALPHA*A*B
		  //-----


		/* unroll 4*/

                alpha = ALPHA * A[i*lda+(k+2)];
                vaalpha = __builtin_epi_vfmv_v_f_2xf32(alpha, gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb6, gvl); // sum += ALPHA*A*B
                 alpha0 = ALPHA * A[i*lda+(k+3)];
                vaalpha0 = __builtin_epi_vfmv_v_f_2xf32(alpha0, gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha0, vb7, gvl); // sum += ALPHA*A*B

                alpha1 = ALPHA * A[(i+1)*lda+(k+2)];
               vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(alpha1, gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb6, gvl); // sum += ALPHA*A*B
                alpha01 = ALPHA * A[(i+1)*lda+(k+3)];
               vaalpha01 = __builtin_epi_vfmv_v_f_2xf32(alpha01, gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb7, gvl); // sum += ALPHA*A*B
                
		 alpha2 = ALPHA * A[(i+2)*lda+(k+2)];
               vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(alpha2, gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb6, gvl); // sum += ALPHA*A*B
		alpha02 = ALPHA * A[(i+2)*lda+(k+3)];
                vaalpha02 = __builtin_epi_vfmv_v_f_2xf32(alpha02, gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb7, gvl); // sum += ALPHA*A*B
                

		alpha3 = ALPHA * A[(i+3)*lda+(k+2)];
                vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(alpha3, gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb6, gvl); // sum += ALPHA*A*B
		alpha03 = ALPHA * A[(i+3)*lda+(k+3)];
                vaalpha03 = __builtin_epi_vfmv_v_f_2xf32(alpha03, gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb7, gvl); // sum += ALPHA*A*B
                
		alpha4 = ALPHA * A[(i+4)*lda+(k+2)];
               vaalpha4 = __builtin_epi_vfmv_v_f_2xf32(alpha4, gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb6, gvl); // sum += ALPHA*A*B
		alpha04 = ALPHA * A[(i+4)*lda+(k+3)];
               vaalpha04 = __builtin_epi_vfmv_v_f_2xf32(alpha04, gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb7, gvl); // sum += ALPHA*A*B
                
		alpha5 = ALPHA * A[(i+5)*lda+(k+2)];
                vaalpha5 = __builtin_epi_vfmv_v_f_2xf32(alpha5, gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb6, gvl); // sum += ALPHA*A*B
		 alpha05 = ALPHA * A[(i+5)*lda+(k+3)];
               vaalpha05 = __builtin_epi_vfmv_v_f_2xf32(alpha05, gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb7, gvl); // sum += ALPHA*A*B
                
		alpha6 = ALPHA * A[(i+6)*lda+(k+2)];
                vaalpha6 = __builtin_epi_vfmv_v_f_2xf32(alpha6, gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb6, gvl); // sum += ALPHA*A*B
		 alpha06 = ALPHA * A[(i+6)*lda+(k+3)];
                 vaalpha06 = __builtin_epi_vfmv_v_f_2xf32(alpha06, gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb7, gvl); // sum += ALPHA*A*B
               
		 alpha7 = ALPHA * A[(i+7)*lda+(k+2)];
                 vaalpha7 = __builtin_epi_vfmv_v_f_2xf32(alpha7, gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb6, gvl); // sum += ALPHA*A*B
		  alpha07 = ALPHA * A[(i+7)*lda+(k+3)];
                  vaalpha07 = __builtin_epi_vfmv_v_f_2xf32(alpha07, gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb7, gvl); // sum += ALPHA*A*B
               

		  alpha8 = ALPHA * A[(i+8)*lda+(k+2)];
                  vaalpha8 = __builtin_epi_vfmv_v_f_2xf32(alpha8, gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb6, gvl); // sum += ALPHA*A*B
		  alpha08 = ALPHA * A[(i+8)*lda+(k+3)];
                  vaalpha08 = __builtin_epi_vfmv_v_f_2xf32(alpha08, gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb7, gvl); // sum += ALPHA*A*B
                

		   alpha9 = ALPHA * A[(i+9)*lda+(k+2)];
                   vaalpha9= __builtin_epi_vfmv_v_f_2xf32(alpha9, gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb6, gvl); // sum += ALPHA*A*B
		  alpha09 = ALPHA * A[(i+9)*lda+(k+3)];
               	 vaalpha09= __builtin_epi_vfmv_v_f_2xf32(alpha09, gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb7, gvl); // sum += ALPHA*A*B
                

		  alpha10 = ALPHA * A[(i+10)*lda+(k+2)];
                 vaalpha10 = __builtin_epi_vfmv_v_f_2xf32(alpha10, gvl); // ALPHA*A
                   vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb6, gvl); // sum += ALPHA*A*B
		 alpha010 = ALPHA * A[(i+10)*lda+(k+3)];
                 vaalpha010 = __builtin_epi_vfmv_v_f_2xf32(alpha010, gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb7, gvl); // sum += ALPHA*A*B
                
		alpha11 = ALPHA * A[(i+11)*lda+(k+2)];
                vaalpha11 = __builtin_epi_vfmv_v_f_2xf32(alpha11, gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb6, gvl); // sum += ALPHA*A*B
		 alpha011 = ALPHA * A[(i+11)*lda+(k+3)];
                 vaalpha011 = __builtin_epi_vfmv_v_f_2xf32(alpha011, gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb7, gvl); // sum += ALPHA*A*B
                

		 alpha12 = ALPHA * A[(i+12)*lda+(k+2)];
                vaalpha12 = __builtin_epi_vfmv_v_f_2xf32(alpha12, gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb6, gvl); // sum += ALPHA*A*B
		 alpha012 = ALPHA * A[(i+12)*lda+(k+3)];
                vaalpha012 = __builtin_epi_vfmv_v_f_2xf32(alpha012, gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb7, gvl); // sum += ALPHA*A*B
                

		 alpha13 = ALPHA * A[(i+13)*lda+(k+2)];
                 vaalpha13 = __builtin_epi_vfmv_v_f_2xf32(alpha13, gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb6, gvl); // sum += ALPHA*A*B
		 alpha013 = ALPHA * A[(i+13)*lda+(k+3)];
                 vaalpha013 = __builtin_epi_vfmv_v_f_2xf32(alpha013, gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb7, gvl); // sum += ALPHA*A*B
                

		  alpha14 = ALPHA * A[(i+14)*lda+(k+2)];
                   vaalpha14 = __builtin_epi_vfmv_v_f_2xf32(alpha14, gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb6, gvl); // sum += ALPHA*A*B
		  alpha014 = ALPHA * A[(i+14)*lda+(k+3)];
                 vaalpha014 = __builtin_epi_vfmv_v_f_2xf32(alpha014, gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb7, gvl); // sum += ALPHA*A*B
                

		   alpha15 = ALPHA * A[(i+15)*lda+(k+2)];
                   vaalpha15 = __builtin_epi_vfmv_v_f_2xf32(alpha15, gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb6, gvl); // sum += ALPHA*A*B
		   alpha015 = ALPHA * A[(i+15)*lda+(k+3)];
                   vaalpha015 = __builtin_epi_vfmv_v_f_2xf32(alpha015, gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb7, gvl); // sum += ALPHA*A*B
		}
		else
		{
			
                register float alpha = ALPHA * A[i*lda+k];
               __epi_2xf32 vaalpha = __builtin_epi_vfmv_v_f_2xf32(alpha, gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb, gvl); // sum += ALPHA*A*B
                register float alpha0 = ALPHA * A[i*lda+(k+1)];
               __epi_2xf32 vaalpha0 = __builtin_epi_vfmv_v_f_2xf32(alpha0, gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha0, vb1, gvl); // sum += ALPHA*A*B

                register float alpha1 = ALPHA * A[(i+1)*lda+k];
               __epi_2xf32 vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(alpha1, gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl); // sum += ALPHA*A*B
                register float alpha01 = ALPHA * A[(i+1)*lda+(k+1)];
               __epi_2xf32 vaalpha01 = __builtin_epi_vfmv_v_f_2xf32(alpha01, gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb1, gvl); // sum += ALPHA*A*B
                
		register float alpha2 = ALPHA * A[(i+2)*lda+k];
               __epi_2xf32 vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(alpha2, gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl); // sum += ALPHA*A*B
		register float alpha02 = ALPHA * A[(i+2)*lda+(k+1)];
               __epi_2xf32 vaalpha02 = __builtin_epi_vfmv_v_f_2xf32(alpha02, gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb1, gvl); // sum += ALPHA*A*B
                

		register float alpha3 = ALPHA * A[(i+3)*lda+k];
               __epi_2xf32 vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(alpha3, gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl); // sum += ALPHA*A*B
		register float alpha03 = ALPHA * A[(i+3)*lda+(k+1)];
               __epi_2xf32 vaalpha03 = __builtin_epi_vfmv_v_f_2xf32(alpha03, gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb1, gvl); // sum += ALPHA*A*B
                
		register float alpha4 = ALPHA * A[(i+4)*lda+k];
               __epi_2xf32 vaalpha4 = __builtin_epi_vfmv_v_f_2xf32(alpha4, gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb, gvl); // sum += ALPHA*A*B
		register float alpha04 = ALPHA * A[(i+4)*lda+(k+1)];
               __epi_2xf32 vaalpha04 = __builtin_epi_vfmv_v_f_2xf32(alpha04, gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb1, gvl); // sum += ALPHA*A*B
                
		register float alpha5 = ALPHA * A[(i+5)*lda+k];
               __epi_2xf32 vaalpha5 = __builtin_epi_vfmv_v_f_2xf32(alpha5, gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb, gvl); // sum += ALPHA*A*B
		register float alpha05 = ALPHA * A[(i+5)*lda+(k+1)];
               __epi_2xf32 vaalpha05 = __builtin_epi_vfmv_v_f_2xf32(alpha05, gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb1, gvl); // sum += ALPHA*A*B
                
		register float alpha6 = ALPHA * A[(i+6)*lda+k];
               __epi_2xf32 vaalpha6 = __builtin_epi_vfmv_v_f_2xf32(alpha6, gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb, gvl); // sum += ALPHA*A*B
		register float alpha06 = ALPHA * A[(i+6)*lda+(k+1)];
               __epi_2xf32 vaalpha06 = __builtin_epi_vfmv_v_f_2xf32(alpha06, gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb1, gvl); // sum += ALPHA*A*B
               
		 register float alpha7 = ALPHA * A[(i+7)*lda+k];
               __epi_2xf32 vaalpha7 = __builtin_epi_vfmv_v_f_2xf32(alpha7, gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb, gvl); // sum += ALPHA*A*B
		 register float alpha07 = ALPHA * A[(i+7)*lda+(k+1)];
               __epi_2xf32 vaalpha07 = __builtin_epi_vfmv_v_f_2xf32(alpha07, gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb1, gvl); // sum += ALPHA*A*B
               

		 register float alpha8 = ALPHA * A[(i+8)*lda+k];
               __epi_2xf32 vaalpha8 = __builtin_epi_vfmv_v_f_2xf32(alpha8, gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb, gvl); // sum += ALPHA*A*B
		 register float alpha08 = ALPHA * A[(i+8)*lda+(k+1)];
               __epi_2xf32 vaalpha08 = __builtin_epi_vfmv_v_f_2xf32(alpha08, gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb1, gvl); // sum += ALPHA*A*B
                

		register float alpha9 = ALPHA * A[(i+9)*lda+k];
               __epi_2xf32 vaalpha9= __builtin_epi_vfmv_v_f_2xf32(alpha9, gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb, gvl); // sum += ALPHA*A*B
		register float alpha09 = ALPHA * A[(i+9)*lda+(k+1)];
               __epi_2xf32 vaalpha09= __builtin_epi_vfmv_v_f_2xf32(alpha09, gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb1, gvl); // sum += ALPHA*A*B
                

		register float alpha10 = ALPHA * A[(i+10)*lda+k];
               __epi_2xf32 vaalpha10 = __builtin_epi_vfmv_v_f_2xf32(alpha10, gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb, gvl); // sum += ALPHA*A*B
		register float alpha010 = ALPHA * A[(i+10)*lda+(k+1)];
               __epi_2xf32 vaalpha010 = __builtin_epi_vfmv_v_f_2xf32(alpha010, gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb1, gvl); // sum += ALPHA*A*B
                
		register float alpha11 = ALPHA * A[(i+11)*lda+k];
               __epi_2xf32 vaalpha11 = __builtin_epi_vfmv_v_f_2xf32(alpha11, gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb, gvl); // sum += ALPHA*A*B
		register float alpha011 = ALPHA * A[(i+11)*lda+(k+1)];
               __epi_2xf32 vaalpha011 = __builtin_epi_vfmv_v_f_2xf32(alpha011, gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb1, gvl); // sum += ALPHA*A*B
                

		register float alpha12 = ALPHA * A[(i+12)*lda+k];
               __epi_2xf32 vaalpha12 = __builtin_epi_vfmv_v_f_2xf32(alpha12, gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb, gvl); // sum += ALPHA*A*B
		register float alpha012 = ALPHA * A[(i+12)*lda+(k+1)];
               __epi_2xf32 vaalpha012 = __builtin_epi_vfmv_v_f_2xf32(alpha012, gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb1, gvl); // sum += ALPHA*A*B
                

		register float alpha13 = ALPHA * A[(i+13)*lda+k];
               __epi_2xf32 vaalpha13 = __builtin_epi_vfmv_v_f_2xf32(alpha13, gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb, gvl); // sum += ALPHA*A*B
		register float alpha013 = ALPHA * A[(i+13)*lda+(k+1)];
               __epi_2xf32 vaalpha013 = __builtin_epi_vfmv_v_f_2xf32(alpha013, gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb1, gvl); // sum += ALPHA*A*B
                

		register float alpha14 = ALPHA * A[(i+14)*lda+k];
               __epi_2xf32 vaalpha14 = __builtin_epi_vfmv_v_f_2xf32(alpha14, gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb, gvl); // sum += ALPHA*A*B
		register float alpha014 = ALPHA * A[(i+14)*lda+(k+1)];
               __epi_2xf32 vaalpha014 = __builtin_epi_vfmv_v_f_2xf32(alpha014, gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb1, gvl); // sum += ALPHA*A*B
                

		register float alpha15 = ALPHA * A[(i+15)*lda+k];
               __epi_2xf32 vaalpha15 = __builtin_epi_vfmv_v_f_2xf32(alpha15, gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb, gvl); // sum += ALPHA*A*B
		register float alpha015 = ALPHA * A[(i+15)*lda+(k+1)];
               __epi_2xf32 vaalpha015 = __builtin_epi_vfmv_v_f_2xf32(alpha015, gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb1, gvl); // sum += ALPHA*A*B
		  //-----


		/* unroll 4*/

                alpha = ALPHA * A[i*lda+(k+2)];
                vaalpha = __builtin_epi_vfmv_v_f_2xf32(alpha, gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb2, gvl); // sum += ALPHA*A*B
                 alpha0 = ALPHA * A[i*lda+(k+3)];
                vaalpha0 = __builtin_epi_vfmv_v_f_2xf32(alpha0, gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha0, vb3, gvl); // sum += ALPHA*A*B

                alpha1 = ALPHA * A[(i+1)*lda+(k+2)];
               vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(alpha1, gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb2, gvl); // sum += ALPHA*A*B
                alpha01 = ALPHA * A[(i+1)*lda+(k+3)];
               vaalpha01 = __builtin_epi_vfmv_v_f_2xf32(alpha01, gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb3, gvl); // sum += ALPHA*A*B
                
		 alpha2 = ALPHA * A[(i+2)*lda+(k+2)];
               vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(alpha2, gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb2, gvl); // sum += ALPHA*A*B
		alpha02 = ALPHA * A[(i+2)*lda+(k+3)];
                vaalpha02 = __builtin_epi_vfmv_v_f_2xf32(alpha02, gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb3, gvl); // sum += ALPHA*A*B
                

		alpha3 = ALPHA * A[(i+3)*lda+(k+2)];
                vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(alpha3, gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb2, gvl); // sum += ALPHA*A*B
		alpha03 = ALPHA * A[(i+3)*lda+(k+3)];
                vaalpha03 = __builtin_epi_vfmv_v_f_2xf32(alpha03, gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb3, gvl); // sum += ALPHA*A*B
                
		alpha4 = ALPHA * A[(i+4)*lda+(k+2)];
               vaalpha4 = __builtin_epi_vfmv_v_f_2xf32(alpha4, gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb2, gvl); // sum += ALPHA*A*B
		alpha04 = ALPHA * A[(i+4)*lda+(k+3)];
               vaalpha04 = __builtin_epi_vfmv_v_f_2xf32(alpha04, gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb3, gvl); // sum += ALPHA*A*B
                
		alpha5 = ALPHA * A[(i+5)*lda+(k+2)];
                vaalpha5 = __builtin_epi_vfmv_v_f_2xf32(alpha5, gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb2, gvl); // sum += ALPHA*A*B
		 alpha05 = ALPHA * A[(i+5)*lda+(k+3)];
               vaalpha05 = __builtin_epi_vfmv_v_f_2xf32(alpha05, gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb3, gvl); // sum += ALPHA*A*B
                
		alpha6 = ALPHA * A[(i+6)*lda+(k+2)];
                vaalpha6 = __builtin_epi_vfmv_v_f_2xf32(alpha6, gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb2, gvl); // sum += ALPHA*A*B
		 alpha06 = ALPHA * A[(i+6)*lda+(k+3)];
                 vaalpha06 = __builtin_epi_vfmv_v_f_2xf32(alpha06, gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb3, gvl); // sum += ALPHA*A*B
               
		 alpha7 = ALPHA * A[(i+7)*lda+(k+2)];
                 vaalpha7 = __builtin_epi_vfmv_v_f_2xf32(alpha7, gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb2, gvl); // sum += ALPHA*A*B
		  alpha07 = ALPHA * A[(i+7)*lda+(k+3)];
                  vaalpha07 = __builtin_epi_vfmv_v_f_2xf32(alpha07, gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb3, gvl); // sum += ALPHA*A*B
               

		  alpha8 = ALPHA * A[(i+8)*lda+(k+2)];
                  vaalpha8 = __builtin_epi_vfmv_v_f_2xf32(alpha8, gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb2, gvl); // sum += ALPHA*A*B
		  alpha08 = ALPHA * A[(i+8)*lda+(k+3)];
                  vaalpha08 = __builtin_epi_vfmv_v_f_2xf32(alpha08, gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb3, gvl); // sum += ALPHA*A*B
                

		   alpha9 = ALPHA * A[(i+9)*lda+(k+2)];
                   vaalpha9= __builtin_epi_vfmv_v_f_2xf32(alpha9, gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb2, gvl); // sum += ALPHA*A*B
		  alpha09 = ALPHA * A[(i+9)*lda+(k+3)];
               	 vaalpha09= __builtin_epi_vfmv_v_f_2xf32(alpha09, gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb3, gvl); // sum += ALPHA*A*B
                

		  alpha10 = ALPHA * A[(i+10)*lda+(k+2)];
                 vaalpha10 = __builtin_epi_vfmv_v_f_2xf32(alpha10, gvl); // ALPHA*A
                   vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb2, gvl); // sum += ALPHA*A*B
		 alpha010 = ALPHA * A[(i+10)*lda+(k+3)];
                 vaalpha010 = __builtin_epi_vfmv_v_f_2xf32(alpha010, gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb3, gvl); // sum += ALPHA*A*B
                
		alpha11 = ALPHA * A[(i+11)*lda+(k+2)];
                vaalpha11 = __builtin_epi_vfmv_v_f_2xf32(alpha11, gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb2, gvl); // sum += ALPHA*A*B
		 alpha011 = ALPHA * A[(i+11)*lda+(k+3)];
                 vaalpha011 = __builtin_epi_vfmv_v_f_2xf32(alpha011, gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb3, gvl); // sum += ALPHA*A*B
                

		 alpha12 = ALPHA * A[(i+12)*lda+(k+2)];
                vaalpha12 = __builtin_epi_vfmv_v_f_2xf32(alpha12, gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb2, gvl); // sum += ALPHA*A*B
		 alpha012 = ALPHA * A[(i+12)*lda+(k+3)];
                vaalpha012 = __builtin_epi_vfmv_v_f_2xf32(alpha012, gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb3, gvl); // sum += ALPHA*A*B
                

		 alpha13 = ALPHA * A[(i+13)*lda+(k+2)];
                 vaalpha13 = __builtin_epi_vfmv_v_f_2xf32(alpha13, gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb2, gvl); // sum += ALPHA*A*B
		 alpha013 = ALPHA * A[(i+13)*lda+(k+3)];
                 vaalpha013 = __builtin_epi_vfmv_v_f_2xf32(alpha013, gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb3, gvl); // sum += ALPHA*A*B
                

		  alpha14 = ALPHA * A[(i+14)*lda+(k+2)];
                   vaalpha14 = __builtin_epi_vfmv_v_f_2xf32(alpha14, gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb2, gvl); // sum += ALPHA*A*B
		  alpha014 = ALPHA * A[(i+14)*lda+(k+3)];
                 vaalpha014 = __builtin_epi_vfmv_v_f_2xf32(alpha014, gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb3, gvl); // sum += ALPHA*A*B
                

		   alpha15 = ALPHA * A[(i+15)*lda+(k+2)];
                   vaalpha15 = __builtin_epi_vfmv_v_f_2xf32(alpha15, gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb2, gvl); // sum += ALPHA*A*B
		   alpha015 = ALPHA * A[(i+15)*lda+(k+3)];
                   vaalpha015 = __builtin_epi_vfmv_v_f_2xf32(alpha015, gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb3, gvl); // sum += ALPHA*A*B
		}
		flag++;
	}
       for ( int k1 = k; k1 < K; k1 += 1) {
		__epi_2xf32 vb = __builtin_epi_vload_2xf32(&B[k1*ldb+j], gvl);

                register float alpha = ALPHA * A[i*lda+k1];
               __epi_2xf32 vaalpha = __builtin_epi_vfmv_v_f_2xf32(alpha, gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb, gvl); // sum += ALPHA*A*B
                register float alpha1 = ALPHA * A[(i+1)*lda+k1];
               __epi_2xf32 vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(alpha1, gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl); // sum += ALPHA*A*B
                register float alpha2 = ALPHA * A[(i+2)*lda+k1];
               __epi_2xf32 vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(alpha2, gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl); // sum += ALPHA*A*B
                register float alpha3 = ALPHA * A[(i+3)*lda+k1];
               __epi_2xf32 vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(alpha3, gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl); // sum += ALPHA*A*B
                register float alpha4 = ALPHA * A[(i+4)*lda+k1];
               __epi_2xf32 vaalpha4 = __builtin_epi_vfmv_v_f_2xf32(alpha4, gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb, gvl); // sum += ALPHA*A*B
                register float alpha5 = ALPHA * A[(i+5)*lda+k1];
               __epi_2xf32 vaalpha5 = __builtin_epi_vfmv_v_f_2xf32(alpha5, gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb, gvl); // sum += ALPHA*A*B
                register float alpha6 = ALPHA * A[(i+6)*lda+k1];
               __epi_2xf32 vaalpha6 = __builtin_epi_vfmv_v_f_2xf32(alpha6, gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb, gvl); // sum += ALPHA*A*B
                register float alpha7 = ALPHA * A[(i+7)*lda+k1];
               __epi_2xf32 vaalpha7 = __builtin_epi_vfmv_v_f_2xf32(alpha7, gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb, gvl); // sum += ALPHA*A*B
                register float alpha8 = ALPHA * A[(i+8)*lda+k1];
               __epi_2xf32 vaalpha8 = __builtin_epi_vfmv_v_f_2xf32(alpha8, gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb, gvl); // sum += ALPHA*A*B
                register float alpha9 = ALPHA * A[(i+9)*lda+k1];
               __epi_2xf32 vaalpha9= __builtin_epi_vfmv_v_f_2xf32(alpha9, gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb, gvl); // sum += ALPHA*A*B
                register float alpha10 = ALPHA * A[(i+10)*lda+k1];
               __epi_2xf32 vaalpha10 = __builtin_epi_vfmv_v_f_2xf32(alpha10, gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb, gvl); // sum += ALPHA*A*B
		register float alpha11 = ALPHA * A[(i+11)*lda+k1];
               __epi_2xf32 vaalpha11 = __builtin_epi_vfmv_v_f_2xf32(alpha11, gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb, gvl); // sum += ALPHA*A*B
                register float alpha12 = ALPHA * A[(i+12)*lda+k1];
               __epi_2xf32 vaalpha12 = __builtin_epi_vfmv_v_f_2xf32(alpha12, gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb, gvl); // sum += ALPHA*A*B
                register float alpha13 = ALPHA * A[(i+13)*lda+k1];
               __epi_2xf32 vaalpha13 = __builtin_epi_vfmv_v_f_2xf32(alpha13, gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb, gvl); // sum += ALPHA*A*B
                register float alpha14 = ALPHA * A[(i+14)*lda+k1];
               __epi_2xf32 vaalpha14 = __builtin_epi_vfmv_v_f_2xf32(alpha14, gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb, gvl); // sum += ALPHA*A*B
                register float alpha15 = ALPHA * A[(i+15)*lda+k1];
               __epi_2xf32 vaalpha15 = __builtin_epi_vfmv_v_f_2xf32(alpha15, gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb, gvl); // sum += ALPHA*A*B

	}
                __builtin_epi_vstore_2xf32(&C[i*ldc+j], vc, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+1)*ldc+j], vc1, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+2)*ldc+j], vc2, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+3)*ldc+j], vc3, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+4)*ldc+j], vc4, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+5)*ldc+j], vc5, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+6)*ldc+j], vc6, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+7)*ldc+j], vc7, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+8)*ldc+j], vc8, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+9)*ldc+j], vc9, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+10)*ldc+j], vc10, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+11)*ldc+j], vc11, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+12)*ldc+j], vc12, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+13)*ldc+j], vc13, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+14)*ldc+j], vc14, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+15)*ldc+j], vc15, gvl);
		//
        }
    j += gvl;
     }}

  int i_left=i;
  for (int j = 0; j < N; ) {
     __epi_2xf32  vaalpha, vaalpha1, vaalpha2, vaalpha3, vc, vc1, vc2, vc3,vb;
     float alpha1, alpha2, alpha3, alpha;
   
     unsigned long int gvl = __builtin_epi_vsetvl(N-j, __epi_e32, __epi_m1);
     for (i=i_left; i < M; i += 4) {    // change according to unroll degree 
        vc= __builtin_epi_vload_2xf32(&C[i*ldc+j], gvl);
       if (i+1 < M) { vc1= __builtin_epi_vload_2xf32(&C[(i+1)*ldc+j], gvl);}
       if (i+2 < M) { vc2= __builtin_epi_vload_2xf32(&C[(i+2)*ldc+j], gvl);}
       if (i+3 < M) {vc3= __builtin_epi_vload_2xf32(&C[(i+3)*ldc+j], gvl);}

        for (int k = 0; k < K; k ++) {
                alpha = ALPHA * A[i*lda+k];
                if (i+1 < M) {alpha1 = ALPHA * A[(i+1)*lda+k]; }
                if (i+2 < M) { alpha2 = ALPHA * A[(i+2)*lda+k];}
                if (i+3 < M) { alpha3 = ALPHA * A[(i+3)*lda+k];}
                vaalpha = __builtin_epi_vfmv_v_f_2xf32(alpha, gvl); // ALPHA*A
               if (i+1 < M) { vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(alpha1, gvl);} // ALPHA*A
               if (i+2 < M) { vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(alpha2, gvl);} // ALPHA*A
               if (i+3 < M) { vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(alpha3, gvl);} // ALPHA*A
                vb = __builtin_epi_vload_2xf32(&B[k*ldb+j], gvl);
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb, gvl); // sum += ALPHA*A*B
                  if (i+1 < M) {vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl);} // sum += ALPHA*A*B
                  if (i+2 < M) {vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl);} // sum += ALPHA*A*B
                  if (i+3 < M) {vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl);}// sum += ALPHA*A*B
        }
          __builtin_epi_vstore_2xf32(&C[i*ldc+j], vc, gvl);
          if (i+1 < M)      {__builtin_epi_vstore_2xf32(&C[(i+1)*ldc+j], vc1, gvl);}
          if (i+2 < M)      {__builtin_epi_vstore_2xf32(&C[(i+2)*ldc+j], vc2, gvl);}
          if (i+3 < M)      {__builtin_epi_vstore_2xf32(&C[(i+3)*ldc+j], vc3, gvl);}
     }
     j += gvl;
  }
}


/***********************3. loop interchange with manual vectorization with ALPHA=1****************/
/* Manual vectorization with loop interchange + loop unrolling*/
void gemm_nn_noalpha(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
  int i=0,j=0,k=0;
  long gvl;
  if(M>15){
  for ( j = 0; j < N; ) {
      gvl = __builtin_epi_vsetvl(N-j, __epi_e32, __epi_m1); 
     for (i = 0; i < M-15; i += 16) {                        
        __epi_2xf32 vb,vb1,vb2,vb3,vc, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vc10, vc11, vc12, vc13, vc14, vc15,vc16,vc17,vc18,vc19,vc20,vc21,vc22,vc23;
        
        vc= __builtin_epi_vload_2xf32(&C[i*ldc+j], gvl);
        vc1= __builtin_epi_vload_2xf32(&C[(i+1)*ldc+j], gvl);
        vc2= __builtin_epi_vload_2xf32(&C[(i+2)*ldc+j], gvl);
        vc3= __builtin_epi_vload_2xf32(&C[(i+3)*ldc+j], gvl);
        vc4= __builtin_epi_vload_2xf32(&C[(i+4)*ldc+j], gvl);
        vc5= __builtin_epi_vload_2xf32(&C[(i+5)*ldc+j], gvl);
        vc6= __builtin_epi_vload_2xf32(&C[(i+6)*ldc+j], gvl);
        vc7= __builtin_epi_vload_2xf32(&C[(i+7)*ldc+j], gvl);
        vc8= __builtin_epi_vload_2xf32(&C[(i+8)*ldc+j], gvl);
        vc9= __builtin_epi_vload_2xf32(&C[(i+9)*ldc+j], gvl);
        vc10= __builtin_epi_vload_2xf32(&C[(i+10)*ldc+j], gvl);
        vc11= __builtin_epi_vload_2xf32(&C[(i+11)*ldc+j], gvl);
        vc12= __builtin_epi_vload_2xf32(&C[(i+12)*ldc+j], gvl);
        vc13= __builtin_epi_vload_2xf32(&C[(i+13)*ldc+j], gvl);
        vc14= __builtin_epi_vload_2xf32(&C[(i+14)*ldc+j], gvl);
        vc15= __builtin_epi_vload_2xf32(&C[(i+15)*ldc+j], gvl);
	//
        for ( k = 0; k < K-3; k +=4) {
                vb = __builtin_epi_vload_2xf32(&B[k*ldb+j], gvl);
                vb1 = __builtin_epi_vload_2xf32(&B[(k+1)*ldb+j], gvl);
                vb2 = __builtin_epi_vload_2xf32(&B[(k+2)*ldb+j], gvl);
                vb3 = __builtin_epi_vload_2xf32(&B[(k+3)*ldb+j], gvl);


               __epi_2xf32 vaalpha = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+k], gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha0 = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+(k+1)], gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha0, vb1, gvl); // sum += ALPHA*A*B

               __epi_2xf32 vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+k], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl); // sum += ALPHA*A*B
               
		__epi_2xf32 vaalpha01 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+(k+1)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb1, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)*lda+k], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha02 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)*lda+(k+1)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb1, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+k], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha03 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+(k+1)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb1, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha4 = __builtin_epi_vfmv_v_f_2xf32(A[(i+4)*lda+k], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha04 = __builtin_epi_vfmv_v_f_2xf32(A[(i+4)*lda+(k+1)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb1, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha5 = __builtin_epi_vfmv_v_f_2xf32(A[(i+5)*lda+k], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha05 = __builtin_epi_vfmv_v_f_2xf32( A[(i+5)*lda+(k+1)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb1, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha6 = __builtin_epi_vfmv_v_f_2xf32(A[(i+6)*lda+k], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha06 = __builtin_epi_vfmv_v_f_2xf32(A[(i+6)*lda+(k+1)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb1, gvl); // sum += ALPHA*A*B
               
               __epi_2xf32 vaalpha7 = __builtin_epi_vfmv_v_f_2xf32(A[(i+7)*lda+k], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha07 = __builtin_epi_vfmv_v_f_2xf32(A[(i+7)*lda+(k+1)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb1, gvl); // sum += ALPHA*A*B
               

               __epi_2xf32 vaalpha8 = __builtin_epi_vfmv_v_f_2xf32(A[(i+8)*lda+k], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha08 = __builtin_epi_vfmv_v_f_2xf32(A[(i+8)*lda+(k+1)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb1, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha9= __builtin_epi_vfmv_v_f_2xf32(A[(i+9)*lda+k], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha09= __builtin_epi_vfmv_v_f_2xf32(A[(i+9)*lda+(k+1)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb1, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha10 = __builtin_epi_vfmv_v_f_2xf32(A[(i+10)*lda+k], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha010 = __builtin_epi_vfmv_v_f_2xf32(A[(i+10)*lda+(k+1)], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb1, gvl); // sum += ALPHA*A*B
                
               __epi_2xf32 vaalpha11 = __builtin_epi_vfmv_v_f_2xf32(A[(i+11)*lda+k], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha011 = __builtin_epi_vfmv_v_f_2xf32(A[(i+11)*lda+(k+1)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb1, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha12 = __builtin_epi_vfmv_v_f_2xf32(A[(i+12)*lda+k], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha012 = __builtin_epi_vfmv_v_f_2xf32(A[(i+12)*lda+(k+1)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb1, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha13 = __builtin_epi_vfmv_v_f_2xf32(A[(i+13)*lda+k], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha013 = __builtin_epi_vfmv_v_f_2xf32(A[(i+13)*lda+(k+1)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb1, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha14 = __builtin_epi_vfmv_v_f_2xf32(A[(i+14)*lda+k], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha014 = __builtin_epi_vfmv_v_f_2xf32(A[(i+14)*lda+(k+1)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb1, gvl); // sum += ALPHA*A*B
                

               __epi_2xf32 vaalpha15 = __builtin_epi_vfmv_v_f_2xf32(A[(i+15)*lda+k], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb, gvl); // sum += ALPHA*A*B
	
               __epi_2xf32 vaalpha015 = __builtin_epi_vfmv_v_f_2xf32(A[(i+15)*lda+(k+1)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb1, gvl); // sum += ALPHA*A*B
		  //-----


		/* unroll 4*/

                vaalpha = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+(k+2)], gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb2, gvl); // sum += ALPHA*A*B
                
                vaalpha0 = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+(k+3)], gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha0, vb3, gvl); // sum += ALPHA*A*B

               vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+(k+2)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb2, gvl); // sum += ALPHA*A*B
               
               vaalpha01 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+(k+3)], gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb3, gvl); // sum += ALPHA*A*B
                
               vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)*lda+(k+2)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb2, gvl); // sum += ALPHA*A*B
	
                vaalpha02 = __builtin_epi_vfmv_v_f_2xf32(A[(i+2)*lda+(k+3)], gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb3, gvl); // sum += ALPHA*A*B
                

                vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+(k+2)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb2, gvl); // sum += ALPHA*A*B
		
                vaalpha03 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+(k+3)], gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb3, gvl); // sum += ALPHA*A*B
                
               vaalpha4 = __builtin_epi_vfmv_v_f_2xf32( A[(i+4)*lda+(k+2)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb2, gvl); // sum += ALPHA*A*B
	
               vaalpha04 = __builtin_epi_vfmv_v_f_2xf32(A[(i+4)*lda+(k+3)], gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb3, gvl); // sum += ALPHA*A*B
                
                vaalpha5 = __builtin_epi_vfmv_v_f_2xf32(A[(i+5)*lda+(k+2)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb2, gvl); // sum += ALPHA*A*B
               
		vaalpha05 = __builtin_epi_vfmv_v_f_2xf32(A[(i+5)*lda+(k+3)], gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb3, gvl); // sum += ALPHA*A*B
                
                vaalpha6 = __builtin_epi_vfmv_v_f_2xf32(A[(i+6)*lda+(k+2)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb2, gvl); // sum += ALPHA*A*B
                 
		vaalpha06 = __builtin_epi_vfmv_v_f_2xf32(A[(i+6)*lda+(k+3)], gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb3, gvl); // sum += ALPHA*A*B
               
                 vaalpha7 = __builtin_epi_vfmv_v_f_2xf32(A[(i+7)*lda+(k+2)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb2, gvl); // sum += ALPHA*A*B
		 
                  vaalpha07 = __builtin_epi_vfmv_v_f_2xf32(A[(i+7)*lda+(k+3)], gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb3, gvl); // sum += ALPHA*A*B
               

                  vaalpha8 = __builtin_epi_vfmv_v_f_2xf32(A[(i+8)*lda+(k+2)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb2, gvl); // sum += ALPHA*A*B
		  
                  vaalpha08 = __builtin_epi_vfmv_v_f_2xf32(A[(i+8)*lda+(k+3)], gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb3, gvl); // sum += ALPHA*A*B
                

                   vaalpha9= __builtin_epi_vfmv_v_f_2xf32(A[(i+9)*lda+(k+2)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb2, gvl); // sum += ALPHA*A*B
		  
               	 vaalpha09= __builtin_epi_vfmv_v_f_2xf32(A[(i+9)*lda+(k+3)], gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb3, gvl); // sum += ALPHA*A*B
                

                 vaalpha10 = __builtin_epi_vfmv_v_f_2xf32(A[(i+10)*lda+(k+2)], gvl); // ALPHA*A
                   vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb2, gvl); // sum += ALPHA*A*B
		 
                 vaalpha010 = __builtin_epi_vfmv_v_f_2xf32(A[(i+10)*lda+(k+3)], gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb3, gvl); // sum += ALPHA*A*B
                
                vaalpha11 = __builtin_epi_vfmv_v_f_2xf32(A[(i+11)*lda+(k+2)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb2, gvl); // sum += ALPHA*A*B
		 
                 vaalpha011 = __builtin_epi_vfmv_v_f_2xf32(A[(i+11)*lda+(k+3)], gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb3, gvl); // sum += ALPHA*A*B
                

                vaalpha12 = __builtin_epi_vfmv_v_f_2xf32(A[(i+12)*lda+(k+2)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb2, gvl); // sum += ALPHA*A*B
		
                vaalpha012 = __builtin_epi_vfmv_v_f_2xf32(A[(i+12)*lda+(k+3)], gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb3, gvl); // sum += ALPHA*A*B
                

                 vaalpha13 = __builtin_epi_vfmv_v_f_2xf32(A[(i+13)*lda+(k+2)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb2, gvl); // sum += ALPHA*A*B
                 
		vaalpha013 = __builtin_epi_vfmv_v_f_2xf32( A[(i+13)*lda+(k+3)], gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb3, gvl); // sum += ALPHA*A*B
                

                   vaalpha14 = __builtin_epi_vfmv_v_f_2xf32(A[(i+14)*lda+(k+2)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb2, gvl); // sum += ALPHA*A*B
		  
                 vaalpha014 = __builtin_epi_vfmv_v_f_2xf32(A[(i+14)*lda+(k+3)], gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb3, gvl); // sum += ALPHA*A*B
                

                   vaalpha15 = __builtin_epi_vfmv_v_f_2xf32(A[(i+15)*lda+(k+2)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb2, gvl); // sum += ALPHA*A*B
		   
                   vaalpha015 = __builtin_epi_vfmv_v_f_2xf32(A[(i+15)*lda+(k+3)], gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb3, gvl); // sum += ALPHA*A*B

	}
       for ( int k1 = k; k1 < K; k1 += 1) {
		__epi_2xf32 vb = __builtin_epi_vload_2xf32(&B[k1*ldb+j], gvl);

                register float alpha =  A[i*lda+k1];
               __epi_2xf32 vaalpha = __builtin_epi_vfmv_v_f_2xf32(alpha, gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb, gvl); // sum += ALPHA*A*B
                register float alpha1 =  A[(i+1)*lda+k1];
               __epi_2xf32 vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(alpha1, gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl); // sum += ALPHA*A*B
                register float alpha2 =  A[(i+2)*lda+k1];
               __epi_2xf32 vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(alpha2, gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl); // sum += ALPHA*A*B
                register float alpha3 =  A[(i+3)*lda+k1];
               __epi_2xf32 vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(alpha3, gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl); // sum += ALPHA*A*B
                register float alpha4 =  A[(i+4)*lda+k1];
               __epi_2xf32 vaalpha4 = __builtin_epi_vfmv_v_f_2xf32(alpha4, gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb, gvl); // sum += ALPHA*A*B
                register float alpha5 =  A[(i+5)*lda+k1];
               __epi_2xf32 vaalpha5 = __builtin_epi_vfmv_v_f_2xf32(alpha5, gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb, gvl); // sum += ALPHA*A*B
                register float alpha6 =  A[(i+6)*lda+k1];
               __epi_2xf32 vaalpha6 = __builtin_epi_vfmv_v_f_2xf32(alpha6, gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb, gvl); // sum += ALPHA*A*B
                register float alpha7 =  A[(i+7)*lda+k1];
               __epi_2xf32 vaalpha7 = __builtin_epi_vfmv_v_f_2xf32(alpha7, gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb, gvl); // sum += ALPHA*A*B
                register float alpha8 =  A[(i+8)*lda+k1];
               __epi_2xf32 vaalpha8 = __builtin_epi_vfmv_v_f_2xf32(alpha8, gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb, gvl); // sum += ALPHA*A*B
                register float alpha9 =  A[(i+9)*lda+k1];
               __epi_2xf32 vaalpha9= __builtin_epi_vfmv_v_f_2xf32(alpha9, gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb, gvl); // sum += ALPHA*A*B
                register float alpha10 =  A[(i+10)*lda+k1];
               __epi_2xf32 vaalpha10 = __builtin_epi_vfmv_v_f_2xf32(alpha10, gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb, gvl); // sum += ALPHA*A*B
		register float alpha11 =  A[(i+11)*lda+k1];
               __epi_2xf32 vaalpha11 = __builtin_epi_vfmv_v_f_2xf32(alpha11, gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb, gvl); // sum += ALPHA*A*B
                register float alpha12 =  A[(i+12)*lda+k1];
               __epi_2xf32 vaalpha12 = __builtin_epi_vfmv_v_f_2xf32(alpha12, gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb, gvl); // sum += ALPHA*A*B
                register float alpha13 =  A[(i+13)*lda+k1];
               __epi_2xf32 vaalpha13 = __builtin_epi_vfmv_v_f_2xf32(alpha13, gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb, gvl); // sum += ALPHA*A*B
                register float alpha14 =  A[(i+14)*lda+k1];
               __epi_2xf32 vaalpha14 = __builtin_epi_vfmv_v_f_2xf32(alpha14, gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb, gvl); // sum += ALPHA*A*B
                register float alpha15 =  A[(i+15)*lda+k1];
               __epi_2xf32 vaalpha15 = __builtin_epi_vfmv_v_f_2xf32(alpha15, gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb, gvl); // sum += ALPHA*A*B

	}
	
                __builtin_epi_vstore_2xf32(&C[i*ldc+j], vc, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+1)*ldc+j], vc1, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+2)*ldc+j], vc2, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+3)*ldc+j], vc3, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+4)*ldc+j], vc4, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+5)*ldc+j], vc5, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+6)*ldc+j], vc6, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+7)*ldc+j], vc7, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+8)*ldc+j], vc8, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+9)*ldc+j], vc9, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+10)*ldc+j], vc10, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+11)*ldc+j], vc11, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+12)*ldc+j], vc12, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+13)*ldc+j], vc13, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+14)*ldc+j], vc14, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+15)*ldc+j], vc15, gvl);
		//
        }
    j += gvl;
     }}

  int i_left=i;
  for (int j = 0; j < N; ) {
     __epi_2xf32  vaalpha, vaalpha1, vaalpha2, vaalpha3, vc, vc1, vc2, vc3,vb;
     float alpha1, alpha2, alpha3, alpha;
   
     unsigned long int gvl = __builtin_epi_vsetvl(N-j, __epi_e32, __epi_m1);
     for (i=i_left; i < M; i += 4) {    // change according to unroll degree 
        vc= __builtin_epi_vload_2xf32(&C[i*ldc+j], gvl);
       if (i+1 < M) { vc1= __builtin_epi_vload_2xf32(&C[(i+1)*ldc+j], gvl);}
       if (i+2 < M) { vc2= __builtin_epi_vload_2xf32(&C[(i+2)*ldc+j], gvl);}
       if (i+3 < M) {vc3= __builtin_epi_vload_2xf32(&C[(i+3)*ldc+j], gvl);}

        for (int k = 0; k < K; k ++) {
                vaalpha = __builtin_epi_vfmv_v_f_2xf32(A[i*lda+k], gvl); // ALPHA*A
               if (i+1 < M) { vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(A[(i+1)*lda+k], gvl);} // ALPHA*A
               if (i+2 < M) { vaalpha2 = __builtin_epi_vfmv_v_f_2xf32( A[(i+2)*lda+k], gvl);} // ALPHA*A
               if (i+3 < M) { vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(A[(i+3)*lda+k], gvl);} // ALPHA*A
                vb = __builtin_epi_vload_2xf32(&B[k*ldb+j], gvl);
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb, gvl); // sum += ALPHA*A*B
                  if (i+1 < M) {vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl);} // sum += ALPHA*A*B
                  if (i+2 < M) {vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl);} // sum += ALPHA*A*B
                  if (i+3 < M) {vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl);}// sum += ALPHA*A*B
        }
          __builtin_epi_vstore_2xf32(&C[i*ldc+j], vc, gvl);
          if (i+1 < M)      {__builtin_epi_vstore_2xf32(&C[(i+1)*ldc+j], vc1, gvl);}
          if (i+2 < M)      {__builtin_epi_vstore_2xf32(&C[(i+2)*ldc+j], vc2, gvl);}
          if (i+3 < M)      {__builtin_epi_vstore_2xf32(&C[(i+3)*ldc+j], vc3, gvl);}
     }
     j += gvl;
  }
}


/***********************3. loop interchange with manual vectorization ALPHA!=1****************/
/* Manual vectorization with loop interchange + loop unrolling*/
void gemm_nn(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
  int i=0,j=0,k=0;
  long gvl;
  if(M>15){
  for ( j = 0; j < N; ) {
      gvl = __builtin_epi_vsetvl(N-j, __epi_e32, __epi_m1); 
     for (i = 0; i < M-15; i += 16) {                        
        __epi_2xf32 vb,vb1,vb2,vb3,vc, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vc10, vc11, vc12, vc13, vc14, vc15,vc16,vc17,vc18,vc19,vc20,vc21,vc22,vc23;
        
        vc= __builtin_epi_vload_2xf32(&C[i*ldc+j], gvl);
        vc1= __builtin_epi_vload_2xf32(&C[(i+1)*ldc+j], gvl);
        vc2= __builtin_epi_vload_2xf32(&C[(i+2)*ldc+j], gvl);
        vc3= __builtin_epi_vload_2xf32(&C[(i+3)*ldc+j], gvl);
        vc4= __builtin_epi_vload_2xf32(&C[(i+4)*ldc+j], gvl);
        vc5= __builtin_epi_vload_2xf32(&C[(i+5)*ldc+j], gvl);
        vc6= __builtin_epi_vload_2xf32(&C[(i+6)*ldc+j], gvl);
        vc7= __builtin_epi_vload_2xf32(&C[(i+7)*ldc+j], gvl);
        vc8= __builtin_epi_vload_2xf32(&C[(i+8)*ldc+j], gvl);
        vc9= __builtin_epi_vload_2xf32(&C[(i+9)*ldc+j], gvl);
        vc10= __builtin_epi_vload_2xf32(&C[(i+10)*ldc+j], gvl);
        vc11= __builtin_epi_vload_2xf32(&C[(i+11)*ldc+j], gvl);
        vc12= __builtin_epi_vload_2xf32(&C[(i+12)*ldc+j], gvl);
        vc13= __builtin_epi_vload_2xf32(&C[(i+13)*ldc+j], gvl);
        vc14= __builtin_epi_vload_2xf32(&C[(i+14)*ldc+j], gvl);
        vc15= __builtin_epi_vload_2xf32(&C[(i+15)*ldc+j], gvl);
	//
        for ( k = 0; k < K-3; k +=4) {
                vb = __builtin_epi_vload_2xf32(&B[k*ldb+j], gvl);
                vb1 = __builtin_epi_vload_2xf32(&B[(k+1)*ldb+j], gvl);
                vb2 = __builtin_epi_vload_2xf32(&B[(k+2)*ldb+j], gvl);
                vb3 = __builtin_epi_vload_2xf32(&B[(k+3)*ldb+j], gvl);



                register float alpha = ALPHA * A[i*lda+k];
               __epi_2xf32 vaalpha = __builtin_epi_vfmv_v_f_2xf32(alpha, gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb, gvl); // sum += ALPHA*A*B
                register float alpha0 = ALPHA * A[i*lda+(k+1)];
               __epi_2xf32 vaalpha0 = __builtin_epi_vfmv_v_f_2xf32(alpha0, gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha0, vb1, gvl); // sum += ALPHA*A*B

                register float alpha1 = ALPHA * A[(i+1)*lda+k];
               __epi_2xf32 vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(alpha1, gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl); // sum += ALPHA*A*B
                register float alpha01 = ALPHA * A[(i+1)*lda+(k+1)];
               __epi_2xf32 vaalpha01 = __builtin_epi_vfmv_v_f_2xf32(alpha01, gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb1, gvl); // sum += ALPHA*A*B
                
		register float alpha2 = ALPHA * A[(i+2)*lda+k];
               __epi_2xf32 vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(alpha2, gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl); // sum += ALPHA*A*B
		register float alpha02 = ALPHA * A[(i+2)*lda+(k+1)];
               __epi_2xf32 vaalpha02 = __builtin_epi_vfmv_v_f_2xf32(alpha02, gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb1, gvl); // sum += ALPHA*A*B
                

		register float alpha3 = ALPHA * A[(i+3)*lda+k];
               __epi_2xf32 vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(alpha3, gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl); // sum += ALPHA*A*B
		register float alpha03 = ALPHA * A[(i+3)*lda+(k+1)];
               __epi_2xf32 vaalpha03 = __builtin_epi_vfmv_v_f_2xf32(alpha03, gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb1, gvl); // sum += ALPHA*A*B
                
		register float alpha4 = ALPHA * A[(i+4)*lda+k];
               __epi_2xf32 vaalpha4 = __builtin_epi_vfmv_v_f_2xf32(alpha4, gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb, gvl); // sum += ALPHA*A*B
		register float alpha04 = ALPHA * A[(i+4)*lda+(k+1)];
               __epi_2xf32 vaalpha04 = __builtin_epi_vfmv_v_f_2xf32(alpha04, gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb1, gvl); // sum += ALPHA*A*B
                
		register float alpha5 = ALPHA * A[(i+5)*lda+k];
               __epi_2xf32 vaalpha5 = __builtin_epi_vfmv_v_f_2xf32(alpha5, gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb, gvl); // sum += ALPHA*A*B
		register float alpha05 = ALPHA * A[(i+5)*lda+(k+1)];
               __epi_2xf32 vaalpha05 = __builtin_epi_vfmv_v_f_2xf32(alpha05, gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb1, gvl); // sum += ALPHA*A*B
                
		register float alpha6 = ALPHA * A[(i+6)*lda+k];
               __epi_2xf32 vaalpha6 = __builtin_epi_vfmv_v_f_2xf32(alpha6, gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb, gvl); // sum += ALPHA*A*B
		register float alpha06 = ALPHA * A[(i+6)*lda+(k+1)];
               __epi_2xf32 vaalpha06 = __builtin_epi_vfmv_v_f_2xf32(alpha06, gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb1, gvl); // sum += ALPHA*A*B
               
		 register float alpha7 = ALPHA * A[(i+7)*lda+k];
               __epi_2xf32 vaalpha7 = __builtin_epi_vfmv_v_f_2xf32(alpha7, gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb, gvl); // sum += ALPHA*A*B
		 register float alpha07 = ALPHA * A[(i+7)*lda+(k+1)];
               __epi_2xf32 vaalpha07 = __builtin_epi_vfmv_v_f_2xf32(alpha07, gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb1, gvl); // sum += ALPHA*A*B
               

		 register float alpha8 = ALPHA * A[(i+8)*lda+k];
               __epi_2xf32 vaalpha8 = __builtin_epi_vfmv_v_f_2xf32(alpha8, gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb, gvl); // sum += ALPHA*A*B
		 register float alpha08 = ALPHA * A[(i+8)*lda+(k+1)];
               __epi_2xf32 vaalpha08 = __builtin_epi_vfmv_v_f_2xf32(alpha08, gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb1, gvl); // sum += ALPHA*A*B
                

		register float alpha9 = ALPHA * A[(i+9)*lda+k];
               __epi_2xf32 vaalpha9= __builtin_epi_vfmv_v_f_2xf32(alpha9, gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb, gvl); // sum += ALPHA*A*B
		register float alpha09 = ALPHA * A[(i+9)*lda+(k+1)];
               __epi_2xf32 vaalpha09= __builtin_epi_vfmv_v_f_2xf32(alpha09, gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb1, gvl); // sum += ALPHA*A*B
                

		register float alpha10 = ALPHA * A[(i+10)*lda+k];
               __epi_2xf32 vaalpha10 = __builtin_epi_vfmv_v_f_2xf32(alpha10, gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb, gvl); // sum += ALPHA*A*B
		register float alpha010 = ALPHA * A[(i+10)*lda+(k+1)];
               __epi_2xf32 vaalpha010 = __builtin_epi_vfmv_v_f_2xf32(alpha010, gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb1, gvl); // sum += ALPHA*A*B
                
		register float alpha11 = ALPHA * A[(i+11)*lda+k];
               __epi_2xf32 vaalpha11 = __builtin_epi_vfmv_v_f_2xf32(alpha11, gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb, gvl); // sum += ALPHA*A*B
		register float alpha011 = ALPHA * A[(i+11)*lda+(k+1)];
               __epi_2xf32 vaalpha011 = __builtin_epi_vfmv_v_f_2xf32(alpha011, gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb1, gvl); // sum += ALPHA*A*B
                

		register float alpha12 = ALPHA * A[(i+12)*lda+k];
               __epi_2xf32 vaalpha12 = __builtin_epi_vfmv_v_f_2xf32(alpha12, gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb, gvl); // sum += ALPHA*A*B
		register float alpha012 = ALPHA * A[(i+12)*lda+(k+1)];
               __epi_2xf32 vaalpha012 = __builtin_epi_vfmv_v_f_2xf32(alpha012, gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb1, gvl); // sum += ALPHA*A*B
                

		register float alpha13 = ALPHA * A[(i+13)*lda+k];
               __epi_2xf32 vaalpha13 = __builtin_epi_vfmv_v_f_2xf32(alpha13, gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb, gvl); // sum += ALPHA*A*B
		register float alpha013 = ALPHA * A[(i+13)*lda+(k+1)];
               __epi_2xf32 vaalpha013 = __builtin_epi_vfmv_v_f_2xf32(alpha013, gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb1, gvl); // sum += ALPHA*A*B
                

		register float alpha14 = ALPHA * A[(i+14)*lda+k];
               __epi_2xf32 vaalpha14 = __builtin_epi_vfmv_v_f_2xf32(alpha14, gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb, gvl); // sum += ALPHA*A*B
		register float alpha014 = ALPHA * A[(i+14)*lda+(k+1)];
               __epi_2xf32 vaalpha014 = __builtin_epi_vfmv_v_f_2xf32(alpha014, gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb1, gvl); // sum += ALPHA*A*B
                

		register float alpha15 = ALPHA * A[(i+15)*lda+k];
               __epi_2xf32 vaalpha15 = __builtin_epi_vfmv_v_f_2xf32(alpha15, gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb, gvl); // sum += ALPHA*A*B
		register float alpha015 = ALPHA * A[(i+15)*lda+(k+1)];
               __epi_2xf32 vaalpha015 = __builtin_epi_vfmv_v_f_2xf32(alpha015, gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb1, gvl); // sum += ALPHA*A*B
		  //-----


		/* unroll 4*/

                alpha = ALPHA * A[i*lda+(k+2)];
                vaalpha = __builtin_epi_vfmv_v_f_2xf32(alpha, gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb2, gvl); // sum += ALPHA*A*B
                 alpha0 = ALPHA * A[i*lda+(k+3)];
                vaalpha0 = __builtin_epi_vfmv_v_f_2xf32(alpha0, gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha0, vb3, gvl); // sum += ALPHA*A*B

                alpha1 = ALPHA * A[(i+1)*lda+(k+2)];
               vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(alpha1, gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb2, gvl); // sum += ALPHA*A*B
                alpha01 = ALPHA * A[(i+1)*lda+(k+3)];
               vaalpha01 = __builtin_epi_vfmv_v_f_2xf32(alpha01, gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha01, vb3, gvl); // sum += ALPHA*A*B
                
		 alpha2 = ALPHA * A[(i+2)*lda+(k+2)];
               vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(alpha2, gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb2, gvl); // sum += ALPHA*A*B
		alpha02 = ALPHA * A[(i+2)*lda+(k+3)];
                vaalpha02 = __builtin_epi_vfmv_v_f_2xf32(alpha02, gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha02, vb3, gvl); // sum += ALPHA*A*B
                

		alpha3 = ALPHA * A[(i+3)*lda+(k+2)];
                vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(alpha3, gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb2, gvl); // sum += ALPHA*A*B
		alpha03 = ALPHA * A[(i+3)*lda+(k+3)];
                vaalpha03 = __builtin_epi_vfmv_v_f_2xf32(alpha03, gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha03, vb3, gvl); // sum += ALPHA*A*B
                
		alpha4 = ALPHA * A[(i+4)*lda+(k+2)];
               vaalpha4 = __builtin_epi_vfmv_v_f_2xf32(alpha4, gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb2, gvl); // sum += ALPHA*A*B
		alpha04 = ALPHA * A[(i+4)*lda+(k+3)];
               vaalpha04 = __builtin_epi_vfmv_v_f_2xf32(alpha04, gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha04, vb3, gvl); // sum += ALPHA*A*B
                
		alpha5 = ALPHA * A[(i+5)*lda+(k+2)];
                vaalpha5 = __builtin_epi_vfmv_v_f_2xf32(alpha5, gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb2, gvl); // sum += ALPHA*A*B
		 alpha05 = ALPHA * A[(i+5)*lda+(k+3)];
               vaalpha05 = __builtin_epi_vfmv_v_f_2xf32(alpha05, gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha05, vb3, gvl); // sum += ALPHA*A*B
                
		alpha6 = ALPHA * A[(i+6)*lda+(k+2)];
                vaalpha6 = __builtin_epi_vfmv_v_f_2xf32(alpha6, gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb2, gvl); // sum += ALPHA*A*B
		 alpha06 = ALPHA * A[(i+6)*lda+(k+3)];
                 vaalpha06 = __builtin_epi_vfmv_v_f_2xf32(alpha06, gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha06, vb3, gvl); // sum += ALPHA*A*B
               
		 alpha7 = ALPHA * A[(i+7)*lda+(k+2)];
                 vaalpha7 = __builtin_epi_vfmv_v_f_2xf32(alpha7, gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb2, gvl); // sum += ALPHA*A*B
		  alpha07 = ALPHA * A[(i+7)*lda+(k+3)];
                  vaalpha07 = __builtin_epi_vfmv_v_f_2xf32(alpha07, gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha07, vb3, gvl); // sum += ALPHA*A*B
               

		  alpha8 = ALPHA * A[(i+8)*lda+(k+2)];
                  vaalpha8 = __builtin_epi_vfmv_v_f_2xf32(alpha8, gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb2, gvl); // sum += ALPHA*A*B
		  alpha08 = ALPHA * A[(i+8)*lda+(k+3)];
                  vaalpha08 = __builtin_epi_vfmv_v_f_2xf32(alpha08, gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha08, vb3, gvl); // sum += ALPHA*A*B
                

		   alpha9 = ALPHA * A[(i+9)*lda+(k+2)];
                   vaalpha9= __builtin_epi_vfmv_v_f_2xf32(alpha9, gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb2, gvl); // sum += ALPHA*A*B
		  alpha09 = ALPHA * A[(i+9)*lda+(k+3)];
               	 vaalpha09= __builtin_epi_vfmv_v_f_2xf32(alpha09, gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha09, vb3, gvl); // sum += ALPHA*A*B
                

		  alpha10 = ALPHA * A[(i+10)*lda+(k+2)];
                 vaalpha10 = __builtin_epi_vfmv_v_f_2xf32(alpha10, gvl); // ALPHA*A
                   vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb2, gvl); // sum += ALPHA*A*B
		 alpha010 = ALPHA * A[(i+10)*lda+(k+3)];
                 vaalpha010 = __builtin_epi_vfmv_v_f_2xf32(alpha010, gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha010, vb3, gvl); // sum += ALPHA*A*B
                
		alpha11 = ALPHA * A[(i+11)*lda+(k+2)];
                vaalpha11 = __builtin_epi_vfmv_v_f_2xf32(alpha11, gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb2, gvl); // sum += ALPHA*A*B
		 alpha011 = ALPHA * A[(i+11)*lda+(k+3)];
                 vaalpha011 = __builtin_epi_vfmv_v_f_2xf32(alpha011, gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha011, vb3, gvl); // sum += ALPHA*A*B
                

		 alpha12 = ALPHA * A[(i+12)*lda+(k+2)];
                vaalpha12 = __builtin_epi_vfmv_v_f_2xf32(alpha12, gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb2, gvl); // sum += ALPHA*A*B
		 alpha012 = ALPHA * A[(i+12)*lda+(k+3)];
                vaalpha012 = __builtin_epi_vfmv_v_f_2xf32(alpha012, gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha012, vb3, gvl); // sum += ALPHA*A*B
                

		 alpha13 = ALPHA * A[(i+13)*lda+(k+2)];
                 vaalpha13 = __builtin_epi_vfmv_v_f_2xf32(alpha13, gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb2, gvl); // sum += ALPHA*A*B
		 alpha013 = ALPHA * A[(i+13)*lda+(k+3)];
                 vaalpha013 = __builtin_epi_vfmv_v_f_2xf32(alpha013, gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha013, vb3, gvl); // sum += ALPHA*A*B
                

		  alpha14 = ALPHA * A[(i+14)*lda+(k+2)];
                   vaalpha14 = __builtin_epi_vfmv_v_f_2xf32(alpha14, gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb2, gvl); // sum += ALPHA*A*B
		  alpha014 = ALPHA * A[(i+14)*lda+(k+3)];
                 vaalpha014 = __builtin_epi_vfmv_v_f_2xf32(alpha014, gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha014, vb3, gvl); // sum += ALPHA*A*B
                

		   alpha15 = ALPHA * A[(i+15)*lda+(k+2)];
                   vaalpha15 = __builtin_epi_vfmv_v_f_2xf32(alpha15, gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb2, gvl); // sum += ALPHA*A*B
		   alpha015 = ALPHA * A[(i+15)*lda+(k+3)];
                   vaalpha015 = __builtin_epi_vfmv_v_f_2xf32(alpha015, gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha015, vb3, gvl); // sum += ALPHA*A*B

	}
       for ( int k1 = k; k1 < K; k1 += 1) {
		__epi_2xf32 vb = __builtin_epi_vload_2xf32(&B[k1*ldb+j], gvl);

                register float alpha = ALPHA * A[i*lda+k1];
               __epi_2xf32 vaalpha = __builtin_epi_vfmv_v_f_2xf32(alpha, gvl); // ALPHA*A
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb, gvl); // sum += ALPHA*A*B
                register float alpha1 = ALPHA * A[(i+1)*lda+k1];
               __epi_2xf32 vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(alpha1, gvl); // ALPHA*A
                  vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl); // sum += ALPHA*A*B
                register float alpha2 = ALPHA * A[(i+2)*lda+k1];
               __epi_2xf32 vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(alpha2, gvl); // ALPHA*A
                  vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl); // sum += ALPHA*A*B
                register float alpha3 = ALPHA * A[(i+3)*lda+k1];
               __epi_2xf32 vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(alpha3, gvl); // ALPHA*A
                  vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl); // sum += ALPHA*A*B
                register float alpha4 = ALPHA * A[(i+4)*lda+k1];
               __epi_2xf32 vaalpha4 = __builtin_epi_vfmv_v_f_2xf32(alpha4, gvl); // ALPHA*A
                  vc4 = __builtin_epi_vfmacc_2xf32(vc4, vaalpha4, vb, gvl); // sum += ALPHA*A*B
                register float alpha5 = ALPHA * A[(i+5)*lda+k1];
               __epi_2xf32 vaalpha5 = __builtin_epi_vfmv_v_f_2xf32(alpha5, gvl); // ALPHA*A
                  vc5 = __builtin_epi_vfmacc_2xf32(vc5, vaalpha5, vb, gvl); // sum += ALPHA*A*B
                register float alpha6 = ALPHA * A[(i+6)*lda+k1];
               __epi_2xf32 vaalpha6 = __builtin_epi_vfmv_v_f_2xf32(alpha6, gvl); // ALPHA*A
                  vc6= __builtin_epi_vfmacc_2xf32(vc6, vaalpha6, vb, gvl); // sum += ALPHA*A*B
                register float alpha7 = ALPHA * A[(i+7)*lda+k1];
               __epi_2xf32 vaalpha7 = __builtin_epi_vfmv_v_f_2xf32(alpha7, gvl); // ALPHA*A
                  vc7 = __builtin_epi_vfmacc_2xf32(vc7, vaalpha7, vb, gvl); // sum += ALPHA*A*B
                register float alpha8 = ALPHA * A[(i+8)*lda+k1];
               __epi_2xf32 vaalpha8 = __builtin_epi_vfmv_v_f_2xf32(alpha8, gvl); // ALPHA*A
                  vc8 = __builtin_epi_vfmacc_2xf32(vc8, vaalpha8, vb, gvl); // sum += ALPHA*A*B
                register float alpha9 = ALPHA * A[(i+9)*lda+k1];
               __epi_2xf32 vaalpha9= __builtin_epi_vfmv_v_f_2xf32(alpha9, gvl); // ALPHA*A
                  vc9 = __builtin_epi_vfmacc_2xf32(vc9, vaalpha9, vb, gvl); // sum += ALPHA*A*B
                register float alpha10 = ALPHA * A[(i+10)*lda+k1];
               __epi_2xf32 vaalpha10 = __builtin_epi_vfmv_v_f_2xf32(alpha10, gvl); // ALPHA*A
                  vc10 = __builtin_epi_vfmacc_2xf32(vc10, vaalpha10, vb, gvl); // sum += ALPHA*A*B
		register float alpha11 = ALPHA * A[(i+11)*lda+k1];
               __epi_2xf32 vaalpha11 = __builtin_epi_vfmv_v_f_2xf32(alpha11, gvl); // ALPHA*A
                  vc11 = __builtin_epi_vfmacc_2xf32(vc11, vaalpha11, vb, gvl); // sum += ALPHA*A*B
                register float alpha12 = ALPHA * A[(i+12)*lda+k1];
               __epi_2xf32 vaalpha12 = __builtin_epi_vfmv_v_f_2xf32(alpha12, gvl); // ALPHA*A
                  vc12 = __builtin_epi_vfmacc_2xf32(vc12, vaalpha12, vb, gvl); // sum += ALPHA*A*B
                register float alpha13 = ALPHA * A[(i+13)*lda+k1];
               __epi_2xf32 vaalpha13 = __builtin_epi_vfmv_v_f_2xf32(alpha13, gvl); // ALPHA*A
                  vc13 = __builtin_epi_vfmacc_2xf32(vc13, vaalpha13, vb, gvl); // sum += ALPHA*A*B
                register float alpha14 = ALPHA * A[(i+14)*lda+k1];
               __epi_2xf32 vaalpha14 = __builtin_epi_vfmv_v_f_2xf32(alpha14, gvl); // ALPHA*A
                  vc14 = __builtin_epi_vfmacc_2xf32(vc14, vaalpha14, vb, gvl); // sum += ALPHA*A*B
                register float alpha15 = ALPHA * A[(i+15)*lda+k1];
               __epi_2xf32 vaalpha15 = __builtin_epi_vfmv_v_f_2xf32(alpha15, gvl); // ALPHA*A
                  vc15 = __builtin_epi_vfmacc_2xf32(vc15, vaalpha15, vb, gvl); // sum += ALPHA*A*B

	}
                __builtin_epi_vstore_2xf32(&C[i*ldc+j], vc, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+1)*ldc+j], vc1, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+2)*ldc+j], vc2, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+3)*ldc+j], vc3, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+4)*ldc+j], vc4, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+5)*ldc+j], vc5, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+6)*ldc+j], vc6, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+7)*ldc+j], vc7, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+8)*ldc+j], vc8, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+9)*ldc+j], vc9, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+10)*ldc+j], vc10, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+11)*ldc+j], vc11, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+12)*ldc+j], vc12, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+13)*ldc+j], vc13, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+14)*ldc+j], vc14, gvl);
                __builtin_epi_vstore_2xf32(&C[(i+15)*ldc+j], vc15, gvl);
		//
        }
    j += gvl;
     }}
  int i_left=i;
  for (int j = 0; j < N; ) {
     __epi_2xf32  vaalpha, vaalpha1, vaalpha2, vaalpha3, vc, vc1, vc2, vc3,vb;
     float alpha1, alpha2, alpha3, alpha;
   
     unsigned long int gvl = __builtin_epi_vsetvl(N-j, __epi_e32, __epi_m1);
     for (i=i_left; i < M; i += 4) {    // change according to unroll degree 
        vc= __builtin_epi_vload_2xf32(&C[i*ldc+j], gvl);
       if (i+1 < M) { vc1= __builtin_epi_vload_2xf32(&C[(i+1)*ldc+j], gvl);}
       if (i+2 < M) { vc2= __builtin_epi_vload_2xf32(&C[(i+2)*ldc+j], gvl);}
       if (i+3 < M) {vc3= __builtin_epi_vload_2xf32(&C[(i+3)*ldc+j], gvl);}

        for (int k = 0; k < K; k ++) {
                alpha = ALPHA * A[i*lda+k];
                if (i+1 < M) {alpha1 = ALPHA * A[(i+1)*lda+k]; }
                if (i+2 < M) { alpha2 = ALPHA * A[(i+2)*lda+k];}
                if (i+3 < M) { alpha3 = ALPHA * A[(i+3)*lda+k];}
                vaalpha = __builtin_epi_vfmv_v_f_2xf32(alpha, gvl); // ALPHA*A
               if (i+1 < M) { vaalpha1 = __builtin_epi_vfmv_v_f_2xf32(alpha1, gvl);} // ALPHA*A
               if (i+2 < M) { vaalpha2 = __builtin_epi_vfmv_v_f_2xf32(alpha2, gvl);} // ALPHA*A
               if (i+3 < M) { vaalpha3 = __builtin_epi_vfmv_v_f_2xf32(alpha3, gvl);} // ALPHA*A
                vb = __builtin_epi_vload_2xf32(&B[k*ldb+j], gvl);
                  vc = __builtin_epi_vfmacc_2xf32(vc, vaalpha, vb, gvl); // sum += ALPHA*A*B
                  if (i+1 < M) {vc1 = __builtin_epi_vfmacc_2xf32(vc1, vaalpha1, vb, gvl);} // sum += ALPHA*A*B
                  if (i+2 < M) {vc2 = __builtin_epi_vfmacc_2xf32(vc2, vaalpha2, vb, gvl);} // sum += ALPHA*A*B
                  if (i+3 < M) {vc3 = __builtin_epi_vfmacc_2xf32(vc3, vaalpha3, vb, gvl);}// sum += ALPHA*A*B
        }
          __builtin_epi_vstore_2xf32(&C[i*ldc+j], vc, gvl);
          if (i+1 < M)      {__builtin_epi_vstore_2xf32(&C[(i+1)*ldc+j], vc1, gvl);}
          if (i+2 < M)      {__builtin_epi_vstore_2xf32(&C[(i+2)*ldc+j], vc2, gvl);}
          if (i+3 < M)      {__builtin_epi_vstore_2xf32(&C[(i+3)*ldc+j], vc3, gvl);}
     }
     j += gvl;
  }
}


void gemm_nt(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_tn(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}



//6-loops with packA and PackB
void gemm_nn_pack2(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C,  int ldc, int BlockM, int BlockN, int BlockK, float *transposeB, float *transposeA)

{        int ii,jj,kk,i,j,k;
	int ld = __builtin_epi_vsetvlmax(__epi_e32, __epi_m1);//16;
	//128;
	//512;//__builtin_epi_vsetvlmax(__epi_e32, __epi_m1);
	printf("ld my val = %d", ld);
	long gvl;
	for (jj = 0; jj < N; jj+=BlockN) {
        	int Nc = ((jj+BlockN>N)?(N-jj):(BlockN));
        	for (kk = 0; kk < K; kk+=BlockK) {
                	int Kc = ((kk+BlockK > K)?(K-kk):(BlockK));
                	int itr=0;
                	for(int j=0;j<Nc;)
                	{
				gvl=__builtin_epi_vsetvl(Nc-j, __epi_e32, __epi_m1);
                       		for(int k=0;k<Kc;k++)
                        	{
                        	//      transposeB[k*Kc+j] = B[(k+kk)*ldb+(j+jj)];
                                	__epi_2xf32 tmp = __builtin_epi_vload_2xf32( &B[(k+kk)*ldb+(j+jj)], gvl);
                                	//svst1(pg, &transposeB[((k+(Kc*itr))*ld)+0], tmp);
                                	__builtin_epi_vstore_2xf32( &transposeB[((k+(Kc*(j/ld)))*ld)+0], tmp, gvl);
                                	//transposeB[k*Nc+j] = B[(k+kk)*ldb+(j+jj)];
                        	}
                       	 	itr++;
				j+=gvl;
                	}
                	for (ii = 0; ii < M; ii+=BlockM) {
                        	int Mc = ((ii+BlockM >M)?(M-ii):(BlockM)) ;

				//	gvl=__builtin_epi_vsetvl(Kc-k, __epi_e32, __epi_m1);
                                	int itr1=0;
                                	for(int i=0;i<Mc;i++)
                                	{
                        	for(int k=0;k<Kc;k++)
                        	{
                                  //      	__epi_2xf32 tmp = __builtin_epi_vload_2xf32(&A[(i+ii)*lda+(k+kk)], gvl);
                                    //    	__builtin_epi_vstore_strided_2xf32(&transposeA[k*Mc+i], tmp, Mc*4, gvl);
                                        	transposeA[k*Mc+i] = A[(i+ii)*lda+(k+kk)];
                                	}
			//		k+=gvl;
                        	}

                       	gemm_nn_unroll16(ii,jj,kk,transposeA,transposeB, C,ALPHA, Mc,Nc, Kc, Mc,ld,ldc );
                	}
                }
	}
}


void gemm_nn_original(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
//    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB)
    {


	    /*** enable below for the 6-loops packed implementations*/
/*	float *transposeB, *transposeA;
        int blockM = ((16 >M)?M:(16)) ;
        int blockN  =((512>N)?N:(512));
        int blockK = ((128>K)?K:(128));
        transposeB= (float *)malloc(blockM*blockN*blockK*sizeof(float));
        transposeA= (float *)malloc(blockM*blockN*blockK*sizeof(float));

        if (transposeB == NULL) {
       	 	fprintf(stderr, "Fatal: failed to allocate bytes.\n");
        	exit(0);
        }
        if(transposeA == NULL) {
        	fprintf(stderr, "Fatal: failed to allocate  bytes.\n");
       		exit(0);
        }
	//gemm_nn_original(M,N,K,ALPHA,A, lda,B, ldb, C, ldc);	
	gemm_nn_pack2(M, N, K, ALPHA,A, lda, B, ldb,C, ldc, blockM, blockN, blockK, transposeB, transposeA);
	if(transposeB != NULL)
        {
                free(transposeB);
                transposeB = NULL;
        }
        if(transposeA != NULL)
        {
                free(transposeA);
                transposeA = NULL;
        }
*/


	   /*** 3-loop implementation */
	gemm_nn_noalpha_unroll163loops(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
	
    }
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}

#ifdef GPU

#include <math.h>

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc)
{
    cublasHandle_t handle = blas_handle();
    cudaError_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), 
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
    check_error(status);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void time_gpu_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<32; ++i){
        gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}

void time_gpu(int TA, int TB, int m, int k, int n)
{
    int iter = 10;
    float *a = random_matrix(m,k);
    float *b = random_matrix(k,n);

    int lda = (!TA)?k:m;
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);

    float *a_cl = cuda_make_array(a, m*k);
    float *b_cl = cuda_make_array(b, k*n);
    float *c_cl = cuda_make_array(c, m*n);

    int i;
    clock_t start = clock(), end;
    for(i = 0; i<iter; ++i){
        gemm_gpu(TA,TB,m,n,k,1,a_cl,lda,b_cl,ldb,1,c_cl,n);
        cudaThreadSynchronize();
    }
    double flop = ((double)m)*n*(2.*k + 2.)*iter;
    double gflop = flop/pow(10., 9);
    end = clock();
    double seconds = sec(end-start);
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s, %lf GFLOPS\n",m,k,k,n, TA, TB, seconds, gflop/seconds);
    cuda_free(a_cl);
    cuda_free(b_cl);
    cuda_free(c_cl);
    free(a);
    free(b);
    free(c);
}


void test_gpu_accuracy(int TA, int TB, int m, int k, int n)
{
    srand(0);
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    float *c_gpu = random_matrix(m,n);
    memset(c, 0, m*n*sizeof(float));
    memset(c_gpu, 0, m*n*sizeof(float));
    int i;
    //pm(m,k,b);
    gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c_gpu,n);
    //printf("GPU\n");
    //pm(m, n, c_gpu);

    gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    //printf("\n\nCPU\n");
    //pm(m, n, c);
    double sse = 0;
    for(i = 0; i < m*n; ++i) {
        //printf("%f %f\n", c[i], c_gpu[i]);
        sse += pow(c[i]-c_gpu[i], 2);
    }
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %g SSE\n",m,k,k,n, TA, TB, sse/(m*n));
    free(a);
    free(b);
    free(c);
    free(c_gpu);
}

int test_gpu_blas()
{
    /*
       test_gpu_accuracy(0,0,10,576,75); 

       test_gpu_accuracy(0,0,17,10,10); 
       test_gpu_accuracy(1,0,17,10,10); 
       test_gpu_accuracy(0,1,17,10,10); 
       test_gpu_accuracy(1,1,17,10,10); 

       test_gpu_accuracy(0,0,1000,10,100); 
       test_gpu_accuracy(1,0,1000,10,100); 
       test_gpu_accuracy(0,1,1000,10,100); 
       test_gpu_accuracy(1,1,1000,10,100); 

       test_gpu_accuracy(0,0,10,10,10); 

       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,192,729,1600); 
       time_gpu(0,0,384,196,1728); 
       time_gpu(0,0,256,196,3456); 
       time_gpu(0,0,256,196,2304); 
       time_gpu(0,0,128,4096,12544); 
       time_gpu(0,0,128,4096,4096); 
     */
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,576,12544); 
    time_gpu(0,0,256,2304,784); 
    time_gpu(1,1,2304,256,784); 
    time_gpu(0,0,512,4608,196); 
    time_gpu(1,1,4608,512,196); 

    return 0;
}
#endif

