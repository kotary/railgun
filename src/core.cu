#include "railgun.h"
#include <cuda_runtime.h>

railgun_t gRailgun;
bheap* task_q;

railgun_t*
get_railgun(void)
{
  if (!task_q) {
    task_q = bheap_new(RG_TASK_Q_LEN);
  }
  gRailgun.wrap_args = _wrap_args;
  gRailgun.schedule = _schedule;
  gRailgun.execute = _execute;
  return &gRailgun;
}

void
reset_railgun(void)
{
  if (task_q) {
    bheap_free(task_q);
  }
}
