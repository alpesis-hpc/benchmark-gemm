#ifndef GEMM_CUDA_H
#define GEMM_CUDA_H

#ifdef __cplusplus
extern "C"
{
#endif

/* --------------------------------------------------------------------------------------------- */

void gemm_cuda (int trans_a, int trans_b, 
                int m, int n, int k, 
                float alpha, 
                float * device_a, int lda, 
                float * device_b, int ldb,
                float beta,
                float * device_c, int ldc);

void gemm_nn_cuda (int m, int n, int k, 
                   float alpha,
                   float * device_a, int lda,
                   float * device_b, int ldb,
                   float * device_c, int ldc);

void gemm_nt_cuda (int m, int n, int k, 
                   float alpha,
                   float * device_a, int lda,
                   float * device_b, int ldb,
                   float * device_c, int ldc);

void gemm_tn_cuda (int m, int n, int k, 
                   float alpha,
                   float * device_a, int lda,
                   float * device_b, int ldb,
                   float * device_c, int ldc);

void gemm_tt_cuda (int m, int n, int k, 
                   float alpha,
                   float * device_a, int lda,
                   float * device_b, int ldb,
                   float * device_c, int ldc);

/* --------------------------------------------------------------------------------------------- */

#ifdef __cplusplus
}
#endif

#endif
