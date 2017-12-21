#ifndef GEMM_CPU_H
#define GEMM_CPU_H

#ifdef __cplusplus
extern "C"
{
#endif

/* --------------------------------------------------------------------------------------------- */

void gemm_cpu(int trans_a, int trans_b, 
              int m, int n, int k, 
              float alpha, 
              float * host_a, int lda, 
              float * host_b, int ldb,
              float beta,
              float * host_c, int ldc);

void gemm_nn (int m, int n, int k, 
              float alpha,
              float * host_a, int lda,
              float * host_b, int ldb,
              float * host_c, int ldc);

void gemm_nt (int m, int n, int k, 
              float alpha,
              float * host_a, int lda,
              float * host_b, int ldb,
              float * host_c, int ldc);

void gemm_tn (int m, int n, int k, 
              float alpha,
              float * host_a, int lda,
              float * host_b, int ldb,
              float * host_c, int ldc);

void gemm_tt (int m, int n, int k, 
              float alpha,
              float * host_a, int lda,
              float * host_b, int ldb,
              float * host_c, int ldc);

/* --------------------------------------------------------------------------------------------- */

#ifdef __cplusplus
}
#endif

#endif
