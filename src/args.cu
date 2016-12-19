#include "railgun.h"
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

void
remove_char(const char* src, const char c, char* dst)
{
  const char *p;
  for (p = src; *p != '\0'; p++) {
    if (*p != c)
      *dst++ = *p;
  }
}

railgun_args*
_wrap_args(const char *fmt, ...)
{
  int argc;
  char c, *fmt_new;
  const char *p;
  va_list ap;
  railgun_data_dir dir;
  railgun_args* args;
  railgun_data* argv;

  args = (railgun_args*)malloc(sizeof(railgun_args));
  argc = strlen(fmt);
  if (strchr(fmt, '|') != NULL) {
    argc = strlen(fmt) - 1;
  }
  args->argc = argc;
  argv = (railgun_data*)malloc(sizeof(railgun_data) * argc);
  args->argv = argv;
  dir = RG_DIR_DOWNLOAD;

  p = fmt;
  va_start(ap, fmt);
  while ((c = *p++)) {
    if (c == '|') {
      dir = RG_DIR_READBACK;
      c = *p++;
    }
    switch (c) {
    case 'i':
      argv->type = RG_TYPE_INT_P;
      argv->dir = dir;
      argv->d.ip = va_arg(ap, int*);
      argv->n = va_arg(ap, int);
      break;
    case 'd':
      argv->type = RG_TYPE_DOUBLE_P;
      argv->dir = dir;
      argv->d.dp = va_arg(ap, double*);
      argv->n = va_arg(ap, int);
      break;
    case 'I':
      argv->type = RG_TYPE_INT;
      argv->dir = dir;
      argv->d.i = va_arg(ap, int);
      argv->n = va_arg(ap, int);
      break;
    case 'D':
      argv->type = RG_TYPE_DOUBLE;
      argv->dir = dir;
      argv->d.i = va_arg(ap, double);
      argv->n = va_arg(ap, int);
      break;
    case 'F':
      argv->type = RG_TYPE_FLOAT;
      argv->dir = dir;
      argv->d.f = (float)va_arg(ap, double);
      argv->n = va_arg(ap, int);
      break;
    case 'f':
      argv->type = RG_TYPE_FLOAT_P;
      argv->dir = dir;
      argv->d.fp = (float*)va_arg(ap, double*);
      argv->n = va_arg(ap, int);
      break;
    default:
      break;
    }
    argv++;
  }

  fmt_new = (char*)malloc((argc + 1) * sizeof(char));
  remove_char(fmt, '|', fmt_new);
  args->fmt = fmt_new;

  return args;
}
