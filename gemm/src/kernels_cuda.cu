#include "kernels_cuda.cuh"

/*
 * C := alpha * C
 */
__global__ void VectorMulKernel(int m, int n,
                                float alpha,
                                float * device_a, int lda)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
 
  if (i < m*n)
  {
    device_a[i] = alpha * device_a[i];
  }
}
