#ifndef _RAILGUN_H_
#define _RAILGUN_H_

#include <stdarg.h>

#define SCHEDULE_INTERVAL 10000 // micro second

/* ----- type::args ----- */
typedef enum {
  RG_TYPE_INT_P,
} railgun_data_type;

typedef enum {
  RG_DIR_DOWNLOAD,
  RG_DIR_READBACK,
} railgun_data_dir;

typedef struct {
  railgun_data_type type;
  railgun_data_dir dir;
  union {
    int *ip;
    float *fp;
    double *dp;
  } d;
  int n;
} railgun_data;

typedef struct {
  const char *fmt;
  int argc;
  railgun_data *argv;
} railgun_args;

typedef struct {
  int (*init)(void);
  int (*get_count)(void);
  railgun_args* (*wrap_args)(const char*, ...);
} railgun_t;

/* ----- protoypes ----- */
railgun_t* get_railgun(void);

railgun_args* _wrap_args(const char* fmt, ...);

void dump_args(railgun_args* args);

#endif
