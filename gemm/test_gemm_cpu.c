#include <stdio.h>
#include <stdlib.h>
#include <unity.h>

#include "config.h"

char * get_format()
{
  if (sizeof(DTYPE) == sizeof(int)) return "%d ";
  else if (sizeof(DTYPE) == sizeof(long)) return "%ld ";
  else if (sizeof(DTYPE) == sizeof(float)) return "%f ";
  else if (sizeof(DTYPE) == sizeof(double)) return "%f ";
  else return "";
}

void data_print(DTYPE * data, unsigned int n)
{
  size_t i;
  for (i = 0; i < n; ++i) printf(get_format(), data[i]);
  printf("\n");
}

void test_gemm_cpu (void)
{
  const unsigned int m = 10;
  const unsigned int n = 10;
  const unsigned int k = 10;
  const DTYPE alpha = 1.0f;
  const DTYPE beta = 1.0f;
  const unsigned int lda = k;
  const unsigned int ldb = n;
  const unsigned int ldc = n;
  DTYPE * host_a = (DTYPE*)malloc(m*k*sizeof(DTYPE));
  DTYPE * host_b = (DTYPE*)malloc(k*n*sizeof(DTYPE));
  DTYPE * host_c = (DTYPE*)malloc(m*n*sizeof(DTYPE));

  size_t i;
  for (i = 0; i < m*k; ++i) { host_a[i] = (DTYPE)i*1024; };
  for (i = 0; i < k*n; ++i) { host_b[i] = (DTYPE)i*2048; };
  for (i = 0; i < m*n; ++i) { host_c[i] = 0; };

  gemm_cpu (0, 0, m, n, k, alpha, host_a, lda, host_b, ldb, beta, host_c, ldc);
  printf("host_a:\n"); data_print(host_a, m*k);
  printf("host_b:\n"); data_print(host_b, k*n);
  printf("host_c:\n"); data_print(host_c, m*n);

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

