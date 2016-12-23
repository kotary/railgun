#include "railgun.h"
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void
vector_sqrt(int n, const float *A, float *B)
{
  int i, j;
  for (i = 0; i < n; i++) {
      B[i] = sqrtf(A[i]);
  }
}

int
main(void)
{
  const int k = 128;
  int *lens, *sizes;
  float **ha, **hb;
  int i, j;
  railgun_t *rg;
  railgun_args *args;
  lens = (int*)malloc(sizeof(int) * k);
  sizes = (int*)malloc(sizeof(int) * k);


  for (i = 0; i < k; i++) {
    lens[i] = 1000000;
    sizes[i] = sizeof(float) * lens[i];
  }

  rg = get_railgun();
  ha = (float**)malloc(sizeof(float*) * k);
  hb = (float**)malloc(sizeof(float*) * k);
  for (i = 0; i < k; i++) {
    cudaHostAlloc((void**)&ha[i], sizes[i], cudaHostAllocDefault);
    cudaHostAlloc((void**)&hb[i], sizes[i], cudaHostAllocDefault);
    for (j = 0; j < lens[i]; j++) {
      ha[i][j] = (float)j;
    }
    args = rg->wrap_args("If|f", lens[i], 1, ha[i], lens[i], hb[i], lens[i]);
    rg->schedule((void*)vector_sqrt, args, dim3(1, 1, 1), dim3(1, 1, 1));
  }

  rg->execute();

  for (i = 0; i < k; i++) {
    printf("Result of vector %d:\n", i);
    for (j = 0; j < 10; j++) {
      printf("  B[%d] = %f\n", j, hb[i][j]);
    }
  }

  for (i = 0; i < k; i++) {
    cudaFreeHost(ha[i]);
    cudaFreeHost(hb[i]);
  }
  free(ha);
  free(hb);
  free(lens);
  free(sizes);

  reset_railgun();
  cudaDeviceReset();

  return 0;
}
