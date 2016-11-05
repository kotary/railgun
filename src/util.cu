#include "railgun.h"
#include <stdio.h>

#define print_header(typename, dir, len)\
  printf("\ttype: %s\n", typename);\
  if (dir == RG_DIR_DOWNLOAD) printf("\tdirection:download\n");\
  else printf("\tdirection:readback\n");\
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
  char* cp;
  printf("railgun_args:\n");
  printf("format:%s\n", args->fmt);
  printf("argc: %d\n", args->argc);
  for (int i = 0; i < args->argc; i++) {
    railgun_data v;

    printf("argv[%d]:\n", i);
    v = args->argv[i];
    switch (v.type) {
    case RG_TYPE_INT_P:
      print_header("int*", v.dir, v.n);
      print_values("%d", v.d.ip, v.n);
      break;
    default:
      break;
    }
  }
}
