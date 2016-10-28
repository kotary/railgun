#include "railgun.h"

#include <cuda_runtime.h>

railgun_t gRailgun;

railgun_t*
get_railgun(void)
{
  gRailgun.wrap_args = _wrap_args;
  return &gRailgun;
}
