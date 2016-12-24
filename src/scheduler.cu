#include "railgun.h"
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <queue>
#include <gc.h>

#define malloc GC_malloc
#define realloc GC_realloc
#define calloc(m,n) GC_malloc((m)*(n))
#define free

std::queue<railgun_task> tq;

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

int
_schedule(void* f, railgun_args* args, dim3 blocks, dim3 threads)
{
  railgun_task *t;
  railgun_data *d;
  int i;

  t = (railgun_task*)malloc(sizeof(railgun_task));
  t->f = f;
  t->args = args;
  t->blocks = blocks;
  t->threads = threads;
  t->total = 0;

  d = t->args->argv;
  for (i = 0; i < t->args->argc; i++) {
    t->total += d[i].n * get_data_size(d[i].type);
  }

  bheap_push(task_q, t->total, t);

  return 0;
}

void
execute_task(railgun_task* t, railgun_memory* mem, cudaStream_t* strm)
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

void execute_tasks_df(int n, railgun_task* ts, railgun_memory** mems, cudaStream_t *strms)
{
  int i, j, size;
  railgun_data *d, *argv;

  for (i = 0; i < n; i++) {
    // Phase 00: Pre-Processing
    argv = ts[i].args->argv;

    // Phase 01: Data Transfer
    for (j = 0; j < ts[i].args->argc; j++) {
      d = &argv[j];
      size = d->n * get_data_size(d->type);
      switch (d->type) {
        case RG_TYPE_FLOAT_P:
          cudaMalloc((void**)&(mems[i][j].fp), size);
          if (d->dir == RG_DIR_DOWNLOAD)
            cudaMemcpyAsync(mems[i][j].fp, d->d.fp, size, cudaMemcpyHostToDevice, strms[i]);
          break;
        case RG_TYPE_DOUBLE_P:
          cudaMalloc((void**)&(mems[i][j].dp), size);
          if (d->dir == RG_DIR_DOWNLOAD)
            cudaMemcpyAsync(mems[i][j].dp, d->d.dp, size, cudaMemcpyHostToDevice, strms[i]);
          break;
        case RG_TYPE_INT:
          mems[i][j].i = d->d.i;
          break;
        default:
          break;
      }
    }

    // Phase 02: Kernel Execution
    ((void (*)(int,float*,float*,float*))ts[i].f)<<<ts[i].blocks, ts[i].threads, 0, strms[i]>>>(mems[i][0].i, mems[i][1].fp, mems[i][2].fp, mems[i][3].fp);

    // Phase 03: Data Transfer(GPU -> CPU)
    for (j = 0; j < ts[i].args->argc; j++) {
      d = &argv[j];
      if (d->dir == RG_DIR_READBACK) {
        size = d->n * get_data_size(d->type);
        switch (d->type) {
          case RG_TYPE_FLOAT_P:
            cudaMemcpyAsync(d->d.fp, mems[i][j].fp, size, cudaMemcpyDeviceToHost, strms[i]);
            break;
          case RG_TYPE_DOUBLE_P:
            cudaMemcpyAsync(d->d.dp, mems[i][j].dp, size, cudaMemcpyDeviceToHost, strms[i]);
            break;
          default:
            break;
        }
      }
    }

  }
}

void
execute_tasks_bf(int n, railgun_task* ts, railgun_memory** mems, cudaStream_t* strms)
{
  int i, j, argc;
  cudaError_t err = cudaSuccess;
  size_t size;
  railgun_args *args;
  railgun_data *argv, *d;

  // Phase 00: Pre-Processing

  // Phase 01: Data Transfer(CPU -> GPU)
  for (i = 0; i < n; i++) {
    args = ts[i].args;
    argc = args->argc;
    argv = args->argv;
    for (j = 0; j < argc; j++) {
      d = &argv[j];
      size = d->n * get_data_size(d->type);
      switch (d->type) {
        case RG_TYPE_FLOAT_P:
          cudaMalloc((void**)&(mems[i][j].fp), size);
          if (d->dir == RG_DIR_DOWNLOAD)
            cudaMemcpyAsync(mems[i][j].fp, d->d.fp, size, cudaMemcpyHostToDevice, strms[i]);
          break;
        case RG_TYPE_DOUBLE_P:
          cudaMalloc((void**)&(mems[i][j].dp), size);
          if (d->dir == RG_DIR_DOWNLOAD)
            cudaMemcpyAsync(mems[i][j].dp, d->d.dp, size, cudaMemcpyHostToDevice, strms[i]);
          break;
        case RG_TYPE_INT:
          mems[i][j].i = d->d.i;
          break;
        default:
          break;
      }
    }
  }

  // Phase 02: Kernel Execution
  for (i = 0; i < n; i++) {
    // printf("Kernel Execution No.%d: %s\n", i, ts[i].args->fmt);
    if (!strcmp(ts[i].args->fmt, "Ifff")) {
      ((void (*)(int,float*,float*,float*))ts[i].f)<<<ts[i].blocks, ts[i].threads, 0, strms[i]>>>(mems[i][0].i, mems[i][1].fp, mems[i][2].fp, mems[i][3].fp);
    } else if (!strcmp(ts[i].args->fmt, "Iff")) {
      ((void (*)(int,float*,float*))ts[i].f)<<<ts[i].blocks, ts[i].threads, 0, strms[i]>>>(mems[i][0].i, mems[i][1].fp, mems[i][2].fp);
    }
  }

  // Phase 03: Data Transfer(GPU -> CPU)
  for (i = 0; i < n; i++) {
    args = ts[i].args;
    argc = args->argc;
    argv = args->argv;
    for (j = 0; j < argc; j++) {
      d = &argv[j];
      if (d->dir == RG_DIR_READBACK) {
        size = d->n * get_data_size(d->type);
        switch (d->type) {
          case RG_TYPE_FLOAT_P:
            cudaMemcpyAsync(d->d.fp, mems[i][j].fp, size, cudaMemcpyDeviceToHost, strms[i]);
            break;
          case RG_TYPE_DOUBLE_P:
            cudaMemcpyAsync(d->d.dp, mems[i][j].dp, size, cudaMemcpyDeviceToHost, strms[i]);
            break;
          default:
            break;
        }
      }
    }
  }

  // Phase 04: Post-Processing

  return;
}

void
wait_streams(cudaStream_t* strms, int n)
{
  int i;

  for (i = 0; i < n; i++) {
    printf("waiting...:%p\n", &(strms[i]));
    cudaStreamSynchronize(strms[i]);
  }

  return;
}

int
_execute()
{
  railgun_task *tasks;
  railgun_data *d;
  railgun_memory **mems;
  cudaStream_t *strms;
  int i, j, task_n, total;

  task_n = task_q->tail + 1;
  tasks = (railgun_task*)malloc(task_n * sizeof(railgun_task));
  for (i = 0; i < task_n; i++) {
    tasks[i] = *((railgun_task*)bheap_pop(task_q).opt);
  }

  mems = (railgun_memory**)malloc(task_n * sizeof(railgun_memory*));
  strms = (cudaStream_t*)malloc(task_n * sizeof(cudaStream_t));

  for (i = 0; i < task_n; i++) {
    mems[i] = (railgun_memory*)malloc(tasks[i].args->argc * sizeof(railgun_memory));
    cudaStreamCreate(&(strms[i]));
  }

  // execute_tasks_df(task_n, tasks, mems, strms);
  execute_tasks_bf(task_n, tasks, mems, strms);

  wait_streams(strms, task_n);
  for (i = 0; i < task_n; i++) {
    cudaStreamDestroy(strms[i]);
  }

  return 0;
}
