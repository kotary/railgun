#include "railgun.h"
#include <stdio.h>

void
dump_args(railgun_args* args)
{
  int i;

  printf("railgun_args:\n");
  printf("argc: %d\n", args->argc);
  for (i = 0; i < args->argc; i++) {
    railgun_data v;

    printf("argv[%d]:\n", i);
    v = args->argv[i];
    switch (v.type) {
    case TYPE_INT_P:
      int *p;

      printf("\ttype: int*\n");
      printf("\tlength: %d\n", v.n);
      p = v.d.ip;
      printf("\tvalues:\n");
      for (int i = 0; i < v.n; i++) {
        printf("\t\t%d\n", p[i]);
      }
      break;
    default:
      break;
    }
  }
}
