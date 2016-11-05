#include <stdio.h>
#include "railgun.h"

#define N 20

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

  args = rg->wrap_args("ii|i", a, N, b, N, c, N);

  printf("Testing...\n");

  dump_args(args);

  return 0;
}
