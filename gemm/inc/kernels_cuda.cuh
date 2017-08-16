#ifndef KERNELS_CUDA_H
#define KERNELS_CUDA_H

#include "cuda.h"
#include "cuda_runtime.h"


__global__ void VectorMulKernel(int m, int n,
                                 float alpha,
                                 float * device_a, int lda);

#endif
