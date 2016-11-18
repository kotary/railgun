#include "railgun.h"
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <queue>

std::queue<railgun_task*> tq;

int
_schedule(void* f, railgun_args* args, dim3 blocks, dim3 threads)
{
  railgun_task* t;

  t = (railgun_task*)malloc(sizeof(railgun_task));

  t->f = f;
  t->args = args;
  t->blocks = blocks;
  t->threads = threads;

  tq.push(t);

  return 0;
}

size_t
get_data_size(railgun_data_type t)
{
  switch (t) {
  case RG_TYPE_INT_P:
  case RG_TYPE_INT:
    return sizeof(int);
  case RG_TYPE_DOUBLE_P:
  case RG_TYPE_DOUBLE:
    return sizeof(double);
  default:
    return 0;
  }
}

typedef void (*iii_f)(int*, int*, int*);
typedef void (*ii_f)(int*, int*);
typedef void (*dd_f)(double*, double*);
typedef void (*ddd_f)(double*, double*, double*);
typedef void (*Iddd_f)(int, double*, double*, double*);

void
execute_task(railgun_task* task, railgun_memory* mem, cudaStream_t* strm)
{
  int i, argc;
  size_t size;
  const char *fmt;
  railgun_args *args;
  railgun_data *d, *argv;

  args = task->args;
  argc = args->argc;
  argv = args->argv;
  // allocate and download
  for (i = 0; i < argc; i++) {
    d = &argv[i];
    size = d->n * get_data_size(d->type);
    switch (d->type) {
    case RG_TYPE_INT_P:
      cudaMalloc((void**)&(mem[i].ip), size);
      if (d->dir == RG_DIR_DOWNLOAD)
        cudaMemcpyAsync(mem[i].ip, d->d.ip, size, cudaMemcpyHostToDevice, *strm);
      break;
    case RG_TYPE_DOUBLE_P:
      cudaMalloc((void**)&(mem[i].dp), size);
      if (d->dir == RG_DIR_DOWNLOAD)
        cudaMemcpyAsync(mem[i].dp, d->d.dp, size, cudaMemcpyHostToDevice, *strm);
      break;
    case RG_TYPE_INT:
      mem[i].i = d->d.i;
      break;
    case RG_TYPE_DOUBLE:
      mem[i].d = d->d.d;
      break;
    default:
      break;
    }
  }

  // execute
  fmt = args->fmt;
  if (!strcmp(fmt, "iii")) {
    ((iii_f)task->f)<<<task->blocks, task->threads>>>(mem[0].ip, mem[1].ip, mem[2].ip);
  } else if (!strcmp(fmt, "ii")) {
    ((ii_f)task->f)<<<task->blocks, task->threads>>>(mem[0].ip, mem[1].ip);
  } else if (!strcmp(fmt, "dd")) {
    ((dd_f)task->f)<<<task->blocks, task->threads>>>(mem[0].dp, mem[1].dp);
  } else if (!strcmp(fmt, "ddd")) {
    ((ddd_f)task->f)<<<task->blocks, task->threads>>>(mem[0].dp, mem[1].dp, mem[2].dp);
  } else if (!strcmp(fmt, "Iddd")) {
    ((Iddd_f)task->f)<<<task->blocks, task->threads>>>(mem[0].i, mem[1].dp, mem[2].dp, mem[3].dp);
  }

  // readback
  for (i = 0; i < argc; i++) {
    d = &argv[i];
    if (d->dir == RG_DIR_READBACK) {
      size = d->n * get_data_size(d->type);
      switch (d->type) {
      case RG_TYPE_INT_P:
        cudaMemcpyAsync(d->d.ip, mem[i].ip, size, cudaMemcpyDeviceToHost, *strm);
        break;
      case RG_TYPE_DOUBLE_P:
        // printf("device: d->d.dp = %p\n", d->d.dp);
        cudaMemcpyAsync(d->d.dp, mem[i].dp, size, cudaMemcpyDeviceToHost, *strm);
        break;
      case RG_TYPE_INT:
        d->d.i = mem[i].i;
        break;
      case RG_TYPE_DOUBLE:
        d->d.d = mem[i].d;
      default:
        break;
      }
    }
  }

  return;
}

void
wait_streams(cudaStream_t* strms, int n)
{
  int i;

  for (i = 0; i < n; i++) {
    cudaStreamSynchronize(strms[i]);
  }

  return;
}

int
_execute()
{
  int i, j, n, total, max_i;
  int totals[10];
  railgun_data d;
  railgun_task *t;
  railgun_task *tasks[10];
  railgun_memory **mems;
  cudaStream_t *strms;

  j = 0;
  while (!tq.empty()) {
    t = tq.front();
    tq.pop();
    tasks[j] = t;

    total = 0;
    for (i = 0; i < t->args->argc; i++) {
      d = t->args->argv[i];
      total += get_data_size(d.type) * d.n;
    }
    totals[j] = total;
    j++;
  }
  n = j;
  mems = (railgun_memory**)malloc(n * sizeof(railgun_memory*));
  strms = (cudaStream_t*)malloc(n * sizeof(cudaStream_t));
  for (j = 0; j < n; j++) {
    max_i = 0;
    for (i = 0; i < n; i++) {
      if (totals[i] > totals[max_i]) {
        max_i = i;
      }
    }

    // printf("%d\n", totals[max_i]);
    // printf("%d\n", max_i);

    cudaStreamCreate(&strms[max_i]);
    t = tasks[max_i];
    mems[max_i] = (railgun_memory*)malloc(t->args->argc * sizeof(railgun_memory*));
    execute_task(t, mems[max_i], &strms[max_i]);

    totals[max_i] = 0;
  }

  wait_streams(strms, n);

  // free railgun_memory(on GPU) and stream
  for (i = 0; i < n; i++) {
    // free(mems[i]);
    cudaStreamDestroy(strms[i]);
  }
  // free(mems);
  free(strms);

  return 0;
}
