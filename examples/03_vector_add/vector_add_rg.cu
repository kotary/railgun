#include <stdio.h>
#include <cuda_runtime.h>
#include "railgun.h"

__global__ void
vectorAdd(int numElements, const float *A, const float *B, float *C)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
      for (int j = 0; j < 100; j++) {
        C[i] = A[i] + B[i];
      }
    }
}

int
main(void)
{
    int numElements = 5000000;
    printf("[Vector addition of %d elements]\n", numElements);
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    size_t size = numElements * sizeof(float);
    float *h_A, *h_B, *h_C, *h_D, *h_E, *h_F;
    cudaHostAlloc((void**)&h_A, size, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_B, size, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_C, size, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_D, size, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_E, size, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_F, size, cudaHostAllocDefault);
    // float *h_A = (float *)malloc(size);
    // float *h_B = (float *)malloc(size);
    // float *h_C = (float *)malloc(size);
    // float *h_D = (float *)malloc(size);
    // float *h_E = (float *)malloc(size);
    // float *h_F = (float *)malloc(size);


    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    for (int i = 0; i < numElements; ++i) {
        h_D[i] = i;
        h_E[i] = i;
    }

    railgun_t *rg;
    railgun_args *args;

    rg = get_railgun();
    args = rg->wrap_args("Iff|f", numElements, 1, h_A, numElements, h_B, numElements, h_C, numElements);
    rg->schedule((void*)vectorAdd, args, dim3(blocksPerGrid, 1, 1), dim3(threadsPerBlock, 1, 1));

    args = rg->wrap_args("Iff|f", numElements, 1, h_D, numElements, h_E, numElements, h_F, numElements);
    rg->schedule((void*)vectorAdd, args, dim3(blocksPerGrid, 1, 1), dim3(threadsPerBlock, 1, 1));

    rg->execute();

    for (int i = 0; i < 20; ++i)
    {
      printf("%f\n", h_C[i]);
      printf("%f\n", h_F[i]);
    }

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // free(h_A);
    // free(h_B);
    // free(h_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFreeHost(h_D);
    cudaFreeHost(h_E);
    cudaFreeHost(h_D);

    cudaDeviceReset();

    printf("Done\n");
    return 0;
}
