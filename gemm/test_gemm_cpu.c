#include <stdlib.h>
#include <unity.h>

#include "config.h"


void test_gemm_cpu (void)
{
  const unsigned int m = 20;
  const unsigned int n = 20;
  const unsigned int k = 20;
  const DTYPE alpha = 1.0f;
  const DTYPE beta = 1.0f;
  const unsigned int lda = k;
  const unsigned int ldb = n;
  const unsigned int ldc = n;
  DTYPE * host_a = (DTYPE*)malloc(m*k*sizeof(DTYPE));
  DTYPE * host_b = (DTYPE*)malloc(k*n*sizeof(DTYPE));
  DTYPE * host_c = (DTYPE*)malloc(m*n*sizeof(DTYPE));

  size_t i;
  for (i = 0; i < m*k; ++i) { host_a[i] = (DTYPE)rand() / (DTYPE)RAND_MAX; };
  for (i = 0; i < k*n; ++i) { host_b[i] = (DTYPE)rand() / (DTYPE)RAND_MAX; };
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

