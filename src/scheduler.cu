#include "railgun.h"
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <queue>
#include <gc.h>

std::queue<railgun_task*> tq;
#define malloc GC_malloc
#define realloc GC_realloc
#define calloc(m,n) GC_malloc((m)*(n))
#define free

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
  case RG_TYPE_FLOAT_P:
  case RG_TYPE_FLOAT:
    return sizeof(float);
  case RG_TYPE_DOUBLE_P:
  case RG_TYPE_DOUBLE:
    return sizeof(double);
  default:
    return 0;
  }
}

// typedef void (*iii_f)(int*, int*, int*);
// typedef void (*ii_f)(int*, int*);
// typedef void (*dd_f)(double*, double*);
// typedef void (*ddd_f)(double*, double*, double*);
// typedef void (*Iddd_f)(int, double*, double*, double*);

void
execute_task(railgun_task* t, railgun_memory* mem, cudaStream_t* strm)
// execute_task(railgun_task* t, railgun_memory* mem)
{
  cudaError_t err = cudaSuccess;
  int i, argc;
  size_t size;
  railgun_args *args;
  railgun_data *argv, *d;

  args = t->args;
  argc = args->argc;
  argv = args->argv;

  for (i = 0; i < argc; i++) {
    d = &argv[i];
    size = d->n * get_data_size(d->type);
    switch (d->type) {
      case RG_TYPE_FLOAT_P:
        cudaMalloc((void**)&(mem[i].fp), size);
        if (d->dir == RG_DIR_DOWNLOAD)
          cudaMemcpyAsync(mem[i].fp, d->d.fp, size, cudaMemcpyHostToDevice, *strm);
          // cudaMemcpy(mem[i].fp, d->d.fp, size, cudaMemcpyHostToDevice);
        break;
      case RG_TYPE_DOUBLE_P:
        cudaMalloc((void**)&(mem[i].dp), size);
        if (d->dir == RG_DIR_DOWNLOAD)
          cudaMemcpyAsync(mem[i].dp, d->d.dp, size, cudaMemcpyHostToDevice, *strm);
          // cudaMemcpy(mem[i].dp, d->d.dp, size, cudaMemcpyHostToDevice);
        break;
      case RG_TYPE_INT:
        mem[i].i = d->d.i;
        break;
      default:
        break;
    }
  }

  // err = cudaMalloc((void**)&da, argv[1].n * sizeof(float));
  // err = cudaMalloc((void**)&db, argv[2].n * sizeof(float));
  // err = cudaMalloc((void**)&dc, argv[3].n * sizeof(float));
  //
  // err = cudaMemcpy(da, argv[1].d.fp, argv[1].n * sizeof(float), cudaMemcpyHostToDevice);
  // err = cudaMemcpy(db, argv[2].d.fp, argv[2].n * sizeof(float), cudaMemcpyHostToDevice);

  // ((void (*)(int,float*,float*,float*))t.f)<<<t.blocks, t.threads>>>(mem[0].i, mem[1].fp, mem[2].fp, mem[3].fp);
  printf("now, the execution will start\n");
  // _execute_kernel(args->fmt, t, mem, strm);
  // _execute_kernel(args->fmt, t, mem);
  ((void (*)(int,float*,float*,float*))t->f)<<<t->blocks, t->threads, 0, *strm>>>(mem[0].i, mem[1].fp, mem[2].fp, mem[3].fp);
  // ((void (*)(int,float*,float*,float*))t->f)<<<t->blocks, t->threads>>>(mem[0].i, mem[1].fp, mem[2].fp, mem[3].fp);
  // ((void (*)(int,double*,double*,double*))t->f)<<<t->blocks, t->threads>>>(mem[0].i, mem[1].dp, mem[2].dp, mem[3].dp);

  for (i = 0; i < argc; i++) {
    d = &argv[i];
    if (d->dir == RG_DIR_READBACK) {
      size = d->n * get_data_size(d->type);
      switch (d->type) {
        case RG_TYPE_FLOAT_P:
          cudaMemcpyAsync(d->d.fp, mem[i].fp, size, cudaMemcpyDeviceToHost, *strm);
          // cudaMemcpy(d->d.fp, mem[i].fp, size, cudaMemcpyDeviceToHost);
          break;
        case RG_TYPE_DOUBLE_P:
          cudaMemcpyAsync(d->d.dp, mem[i].dp, size, cudaMemcpyDeviceToHost, *strm);
          // cudaMemcpy(d->d.dp, mem[i].dp, size, cudaMemcpyDeviceToHost);
          break;
        default:
          break;
      }
    }
  }
  // err = cudaMemcpy(argv[3].d.fp, dc, argv[3].n * sizeof(float), cudaMemcpyDeviceToHost);


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
