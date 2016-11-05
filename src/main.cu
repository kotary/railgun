#include <stdio.h>
#include "railgun.h"

#define N 20

__global__ void
matrix_add(int* a, int* b, int* c)
{
  int i;

  i = threadIdx.x;
  if (i < N) {
    c[i] = a[i] + b[i];
  }
}

int
main(void)
{
  int i;
  int a[N], b[N], c[N];
  railgun_t *rg;
  railgun_args *args;

  rg = get_railgun();

  for (i = 0; i < N; i++) {
    a[i] = i;
    b[i] = i % 10;
  }

  printf("Testing...\n");

  args = rg->wrap_args("ii|i", a, N, b, N, c, N);

  dump_args(args);
  rg->schedule((void*)matrix_add, args);

  return 0;
}
