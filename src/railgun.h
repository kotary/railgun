#ifndef _RAILGUN_H_
#define _RAILGUN_H_

#include <stdarg.h>

#define SCHEDULE_INTERVAL 10000 // micro second

/* ----- type::args ----- */
typedef enum {
  TYPE_INT_P,
} railgun_value_type;

typedef struct {
  railgun_value_type type;
  union {
    int* ip;
    float* fp;
    double* dp;
  } d;
  int n;
} railgun_value;

typedef struct {
  int argc;
  railgun_value* argv;
} railgun_args;

typedef struct {
  int (*init)(void);
  int (*get_count)(void);
  railgun_args* (*wrap_args)(const char* fmt, ...);
} railgun_t;

/* ----- protoypes ----- */
railgun_t* get_railgun(void);

railgun_args* _wrap_args(const char* fmt, ...);

void dump_args(railgun_args* args);

#endif
