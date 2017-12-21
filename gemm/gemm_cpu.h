#ifndef GEMM_CPU_H
#define GEMM_CPU_H

#ifdef __cplusplus
extern "C"
{
#endif

#include "config.h"

/* --------------------------------------------------------------------------------------------- */

void gemm_cpu(int trans_a, int trans_b, 
              int m, int n, int k, 
              DTYPE alpha, 
              DTYPE * host_a, int lda, 
              DTYPE * host_b, int ldb,
              DTYPE beta,
              DTYPE * host_c, int ldc);

void gemm_nn (int m, int n, int k, 
              DTYPE alpha,
              DTYPE * host_a, int lda,
              DTYPE * host_b, int ldb,
              DTYPE * host_c, int ldc);

void gemm_nt (int m, int n, int k, 
              DTYPE alpha,
              DTYPE * host_a, int lda,
              DTYPE * host_b, int ldb,
              DTYPE * host_c, int ldc);

void gemm_tn (int m, int n, int k, 
              DTYPE alpha,
              DTYPE * host_a, int lda,
              DTYPE * host_b, int ldb,
              DTYPE * host_c, int ldc);

void gemm_tt (int m, int n, int k, 
              DTYPE alpha,
              DTYPE * host_a, int lda,
              DTYPE * host_b, int ldb,
              DTYPE * host_c, int ldc);

/* --------------------------------------------------------------------------------------------- */

#ifdef __cplusplus
}
#endif

#endif
