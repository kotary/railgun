#include "railgun.h"
#include <stdio.h>
#include <queue>

std::queue<railgun_task*> tq;

int
_schedule(void* f, railgun_args* args)
{
  railgun_task* t;

  t = (railgun_task*)malloc(sizeof(railgun_task));

  t->f = f;
  t->args = args;

  tq.push(t);

  return 0;
}
