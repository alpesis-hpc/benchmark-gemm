#include "gemm_cpu.h"


void gemm_cpu (int trans_a, int trans_b, 
               int m, int n, int k, 
               float alpha, 
               float * host_a, int lda, 
               float * host_b, int ldb,
               float beta,
               float * host_c, int ldc)
{
  //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
  int i, j;
  for(i = 0; i < m; ++i)
  {
    for(j = 0; j < n; ++j)
    {
      host_c[i*ldc + j] *= beta;
    }
  }

  if (!trans_a && !trans_b)
    gemm_nn (m, n, k, alpha, host_a, lda, host_b, ldb, host_c, ldc);
  else if (trans_a && !trans_b)
    gemm_tn (m, n, k, alpha, host_a, lda, host_b, ldb, host_c, ldc);
  else if(!trans_a && trans_b)
    gemm_nt (m, n, k, alpha, host_a, lda, host_b, ldb, host_c, ldc);
  else
    gemm_tt (m, n, k, alpha, host_a, lda, host_b, ldb, host_c, ldc);
}


void gemm_nn (int m, int n, int k, 
              float alpha, 
              float * host_a, int lda, 
              float * host_b, int ldb,
              float * host_c, int ldc)
{
  int a, b, c;
  for(a = 0; a < m; ++a)
  {
    for(c = 0; c < k; ++c)
    {
      register float A_PART = alpha * host_a[a*lda+c];
      for(b = 0; b < n; ++b)
      {
        host_c[a*ldc+b] += A_PART * host_b[c*ldb+b];
      }
    }
  }
}


void gemm_nt (int m, int n, int k, 
              float alpha, 
              float * host_a, int lda, 
              float * host_b, int ldb,
              float * host_c, int ldc)
{
  int a, b, c;
  for(a = 0; a < m; ++a)
  {
    for(b = 0; b < n; ++b)
    {
      register float sum = 0;
      for(c = 0; c < k; ++c)
      {
        sum += alpha * host_a[a*lda+c] * host_b[b*ldb + c];
      }
      host_c[a*ldc+b] += sum;
    }
  }
}


void gemm_tn (int m, int n, int k, 
              float alpha, 
              float * host_a, int lda, 
              float * host_b, int ldb,
              float * host_c, int ldc)
{
  int a, b, c;
  for(a = 0; a < m; ++a)
  {
    for(c = 0; c < k; ++c)
    {
      register float A_PART = alpha * host_a[c*lda+a];
      for(b = 0; b < n; ++b)
      {
        host_c[a*ldc+b] += A_PART * host_b[c*ldb+b];
      }
    }
  }
}


void gemm_tt (int m, int n, int k, 
              float alpha, 
              float * host_a, int lda, 
              float * host_b, int ldb,
              float * host_c, int ldc)
{
  int a, b, c;
  for(a = 0; a < m; ++a)
  {
    for(b = 0; b < n; ++b)
    {
      register float sum = 0;
      for(c = 0; c < k; ++c)
      {
        sum += alpha * host_a[a+c*lda] * host_b[c+b*ldb];
      }
      host_c[a*ldc+b] += sum;
    }
  }
}
