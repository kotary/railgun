#include "railgun.h"
#include <stdio.h>

#define print_header(typename, len)\
  printf("\ttype: %s\n", typename);\
  printf("\tlength: %d\n", len);
#define print_values(format, ptr, len)\
  printf("\tvalues:\n");\
  for (int i = 0; i < len; i++) {\
    printf("\t\t");\
    printf(format, ptr[i]);\
    printf("\n");}

void
dump_args(railgun_args* args)
{
  printf("railgun_args:\n");
  printf("argc: %d\n", args->argc);
  for (int i = 0; i < args->argc; i++) {
    railgun_data v;

    printf("argv[%d]:\n", i);
    v = args->argv[i];
    switch (v.type) {
    case TYPE_INT_P:
      print_header("int*", v.n);
      print_values("%d", v.d.ip, v.n);
      break;
    default:
      break;
    }
  }
}
