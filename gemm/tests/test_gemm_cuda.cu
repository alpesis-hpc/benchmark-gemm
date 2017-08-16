#include <stdlib.h>

#include <unity.h>
#include <cuda_runtime.h>

#include "gemm_cpu.h"
#include "gemm_cuda.h"


void test_gemm_cuda (void)
{
  const unsigned int m = 20;
  const unsigned int n = 20;
  const unsigned int k = 20;
  const float alpha = 0.0f;
  const float beta = 2.0f;
  const unsigned int lda = k;
  const unsigned int ldb = n;
  const unsigned int ldc = n;
  float * host_a = (float*)malloc(m*k*sizeof(float));
  float * host_b = (float*)malloc(k*n*sizeof(float));
  float * host_c_cpu = (float*)malloc(m*n*sizeof(float));
  float * host_c_gpu = (float*)malloc(m*n*sizeof(float));

  size_t i;
  for (i = 0; i < m*k; ++i) { host_a[i] = (float)rand() / (float)RAND_MAX; };
  for (i = 0; i < k*n; ++i) { host_b[i] = (float)rand() / (float)RAND_MAX; };
  for (i = 0; i < m*n; ++i) { host_c_cpu[i] = 1.0f; host_c_gpu[i] = 1.0f; };

  float * device_a;
  float * device_b;
  float * device_c;
  cudaMalloc (&device_a, m*k*sizeof(float));
  cudaMalloc (&device_b, k*n*sizeof(float));
  cudaMalloc (&device_c, m*n*sizeof(float));
  cudaMemcpy (device_a, host_a, m*k*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy (device_b, host_b, k*n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy (device_c, host_c_gpu, m*n*sizeof(float), cudaMemcpyHostToDevice);
  gemm_cuda (0, 0, m, n, k, alpha, device_a, lda, device_b, ldb, beta, device_c, ldc);
  cudaMemcpy (host_c_gpu, device_c, m*n*sizeof(float), cudaMemcpyDeviceToHost);

  gemm_cpu (0, 0, m, n, k, alpha, host_a, lda, host_b, ldb, beta, host_c_cpu, ldc);

  for (int i = 0; i < m*n; ++i)
    TEST_ASSERT_EQUAL_FLOAT (host_c_cpu[i], host_c_gpu[i]);

  cudaFree (device_a);
  cudaFree (device_b);
  cudaFree (device_c);
  free (host_a);
  free (host_b);
  free (host_c_cpu);
  free (host_c_gpu);
}


int main (void)
{
  UNITY_BEGIN ();
  RUN_TEST (test_gemm_cuda);
  return UNITY_END ();
}

