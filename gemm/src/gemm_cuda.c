#include "gemm_cuda.h"


void gemm_cuda (int trans_a, int trans_b,
                int m, int n, int k,
                float alpha,
                float * device_a, int lda,
                float * device_b, int ldb,
                float beta,
                float * device_c, int ldc)
{
  int i, j;
  for (i = 0; i < m; ++i)
  {
    for (j = 0; j < n; ++j)
    {
      device_c[i*ldc+j] *= beta;
    }
  }

  if (!trans_a && !trans_b)
    gemm_nn_cuda (m, n, k, alpha, device_a, lda, device_b, ldb, device_c, ldc);
  else if (trans_a && !trans_b)
    gemm_tn_cuda (m, n, k, alpha, device_a, lda, device_b, ldb, device_c, ldc);
  else if (!trans_a && trans_b)
    gemm_nt_cuda (m, n, k, alpha, device_a, lda, device_b, ldb, device_c, ldc);
  else
    gemm_tt_cuda (m, n, k, alpha, device_a, lda, device_b, ldb, device_c, ldc);

}


void gemm_nn_cuda (int m, int n, int k, 
                   float alpha,
                   float * device_a, int lda,
                   float * device_b, int ldb,
                   float * device_c, int ldc)
{

}


void gemm_nt_cuda (int m, int n, int k, 
                   float alpha,
                   float * device_a, int lda,
                   float * device_b, int ldb,
                   float * device_c, int ldc)
{

}


void gemm_tn_cuda (int m, int n, int k, 
                   float alpha,
                   float * device_a, int lda,
                   float * device_b, int ldb,
                   float * device_c, int ldc)
{

}


void gemm_tt_cuda (int m, int n, int k, 
                   float alpha,
                   float * device_a, int lda,
                   float * device_b, int ldb,
                   float * device_c, int ldc)
{

}
