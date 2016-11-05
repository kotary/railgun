#include <stdio.h>
#include "railgun.h"

#define N 100

__global__ void
matrix_add(int* a, int* b, int* c)
{
  int i;

  i = threadIdx.x;
  if (i < N) {
    c[i] = a[i] + b[i];
  }
}

int main(void)
{
  int *a, *b, *c, i;
  railgun_t *rg;

  a = (int*)malloc(sizeof(int) * N);
  b = (int*)malloc(sizeof(int) * N);
  c = (int*)malloc(sizeof(int) * N);

  for (i = 0; i < N; i++) {
    a[i] = i;
    b[i] = i % 10;
  }

  rg = get_railgun();
  railgun->init();

  args = rg->wrap_args("ii|i", *a, N, *b, N, *c, N);
  tid = rg->schedule(matrix_add, args);
  rg->sync();
}
