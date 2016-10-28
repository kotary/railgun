#include "railgun.h"
#include <stdarg.h>
#include <string.h>

railgun_args*
_wrap_args(const char *fmt, ...)
{
  int len;
  char c;
  va_list ap;
  railgun_args* args;
  railgun_value* argv;

  args = (railgun_args*)malloc(sizeof(railgun_args));

  len = strlen(fmt);
  args->argc = len;
  argv = (railgun_value*)malloc(sizeof(railgun_value) * len);
  args->argv = argv;

  va_start(ap, fmt);
  while ((c = *fmt++)) {
    switch (c) {
    case 'i':
      argv->type = TYPE_INT_P;
      argv->d.ip = va_arg(ap, int*);
      argv->n = va_arg(ap, int);
      break;
    default:
      break;
    }
    argv++;
  }

  return args;
}
