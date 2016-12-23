#ifndef _BINARY_HEAP_LIB_H_
#define _BINARY_HEAP_LIB_H_

typedef struct {
  int val;
  void* opt;
} bheap_node;

typedef struct {
  bheap_node *nodes;
  int len;
  int tail;
} bheap;

bheap* bheap_new(int);
void bheap_free(bheap*);

int bheap_push(bheap*, int, void*);
bheap_node bheap_pop(bheap*);
int bheap_is_empty(bheap*);

void bheap_print(bheap* heap);

#endif
