#include "railgun.h"
#include <stdio.h>
#include <time.h>

#define N 2000000
#define M 4000000
#define L 8000000

__global__ void
kernel(int size, double *a, double *b, double* c)
{
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < size) {
    c[i] = a[i] * b[i];
    i += blockDim.x * gridDim.x;
  }
}

int
main(void)
{
  railgun_t *rg;
  railgun_args *args;
  int i;

  double *ha0, *hb0, *hc0;
  double *ha1, *hb1, *hc1;
  double *ha2, *hb2, *hc2;

  cudaEvent_t start, stop;
  float elps;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaHostAlloc((void**)&ha0, N * sizeof(double), cudaHostAllocDefault);
  cudaHostAlloc((void**)&hb0, N * sizeof(double), cudaHostAllocDefault);
  cudaHostAlloc((void**)&hc0, N * sizeof(double), cudaHostAllocDefault);
  cudaHostAlloc((void**)&ha1, M * sizeof(double), cudaHostAllocDefault);
  cudaHostAlloc((void**)&hb1, M * sizeof(double), cudaHostAllocDefault);
  cudaHostAlloc((void**)&hc1, M * sizeof(double), cudaHostAllocDefault);
  cudaHostAlloc((void**)&ha2, L * sizeof(double), cudaHostAllocDefault);
  cudaHostAlloc((void**)&hb2, L * sizeof(double), cudaHostAllocDefault);
  cudaHostAlloc((void**)&hc2, L * sizeof(double), cudaHostAllocDefault);

  for (i = 0; i < N; i++) {
    ha0[i] = i;
    hb0[i] = i % 6;
  }

  for (i = 0; i < M; i++) {
    ha1[i] = i % 100;
    hb1[i] = i * 2;
  }

  for (i = 0; i < L; i++) {
    ha2[i] = i * 10;
    hb2[i] = i % 1000;
  }

  rg = get_railgun();

  args = rg->wrap_args("Idd|d", N, 1, ha0, N, hb0, N, hc0, N);
  rg->schedule((void*)kernel, args, dim3(1024, 1, 1), dim3(1024, 1, 1));

  args = rg->wrap_args("Idd|d", M, 1, ha1, M, hb1, M, hc1, M);
  rg->schedule((void*)kernel, args, dim3(1024, 1, 1), dim3(1024, 1, 1));

  args = rg->wrap_args("Idd|d", L, 1, ha2, L, hb2, L, hc2, L);
  rg->schedule((void*)kernel, args, dim3(1024, 1, 1), dim3(1024, 1, 1));

  cudaEventRecord(start, 0);
  rg->execute();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elps, start, stop);

  printf("Time taken: %3.1f ms\n", elps);
  for (i = 0; i < 10; i++) {
    printf("hc0[%d]=%lf\n", i, hc0[i]);
  }

  return 0;
}
