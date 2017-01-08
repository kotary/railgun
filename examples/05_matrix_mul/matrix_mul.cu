#include <stdio.h>
#include <cuda_runtime.h>
#include "railgun.h"

#define BLOCK_SIZE 16

__global__ void
matrixMul(int m, int n, int k, float* A, float* B, float *C)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int sum = 0;
  if(col < k && row < m) {
    for(int i = 0; i < n; i++) {
      sum += A[row * n + i] * B[i * k + col];
    }
    C[row * k + col] = sum;
  }
}

int
main(void)
{
  int m, n, k;
  float *h_A, *h_B, *h_C;

  m = 32; n = 64; k = 128;
  cudaHostAlloc((void**)&h_A, sizeof(float)*m*n, cudaHostAllocDefault);
  cudaHostAlloc((void**)&h_B, sizeof(float)*n*k, cudaHostAllocDefault);
  cudaHostAlloc((void**)&h_C, sizeof(float)*m*k, cudaHostAllocDefault);

  // initialize A
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      h_A[i*n+j] = rand() / (float)RAND_MAX;
    }
  }

  // initialize B
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < k; j++) {
      h_B[i*k+j] = rand() / (float)RAND_MAX;
    }
  }

  unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;

  dim3 dimGrid(grid_cols, grid_rows);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  railgun_t *rg;
  railgun_args *args;

  rg = get_railgun();
  args = rg->wrap_args("IIIff|f", m, 1, n, 1, k, 1, h_A, m*n, h_B, n*k, h_C, m*k);
  rg->schedule((void*)matrixMul, args, dim3(grid_cols, grid_rows), dim3(BLOCK_SIZE, BLOCK_SIZE));
  rg->execute();

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      printf("%f\n", h_C[i * k + j]);
    }
  }

  cudaFreeHost(h_A);
  cudaFreeHost(h_B);
  cudaFreeHost(h_C);

  reset_railgun();
  cudaDeviceReset();

  return 0;
}
