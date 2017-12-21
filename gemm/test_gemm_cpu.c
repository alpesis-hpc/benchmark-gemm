#include <stdlib.h>

#include <unity.h>


void test_gemm_cpu (void)
{
  const unsigned int m = 20;
  const unsigned int n = 20;
  const unsigned int k = 20;
  const float alpha = 1.0f;
  const float beta = 1.0f;
  const unsigned int lda = k;
  const unsigned int ldb = n;
  const unsigned int ldc = n;
  float * host_a = (float*)malloc(m*k*sizeof(float));
  float * host_b = (float*)malloc(k*n*sizeof(float));
  float * host_c = (float*)malloc(m*n*sizeof(float));

  size_t i;
  for (i = 0; i < m*k; ++i) { host_a[i] = (float)rand() / (float)RAND_MAX; };
  for (i = 0; i < k*n; ++i) { host_b[i] = (float)rand() / (float)RAND_MAX; };
  for (i = 0; i < m*n; ++i) { host_c[i] = 0.0f; };

  gemm_cpu (0, 0, m, n, k, alpha, host_a, lda, host_b, ldb, beta, host_c, ldc);

  free (host_a);
  free (host_b);
  free (host_c);
}


int main (void)
{
  UNITY_BEGIN ();
  RUN_TEST (test_gemm_cpu);
  return UNITY_END ();
}

