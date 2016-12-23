#include "bheap.h"
#include <stdio.h>
#include <stdlib.h>

#define TRUE 1
#define FALSE 0
#define SWAP(type,a,b)  {type work = a; a = b; b = work;}

void bheap_remove_root(bheap*);
void bheap_bubble_up(bheap*);
void bheap_bubble_down(bheap*);
int bheap_extend(bheap*);

bheap*
bheap_new(int len)
{
  bheap *heap;

  heap = (bheap*)malloc(sizeof(bheap));
  heap->nodes = (bheap_node*)malloc(sizeof(bheap_node) * len);
  heap->len = len;
  heap->tail = -1;

  return heap;
}

void
bheap_free(bheap* heap)
{
  free(heap->nodes);
  free(heap);
  return;
}

int
bheap_extend(bheap* heap)
{
  bheap_node* tmp;
  int new_len;

  new_len = heap->len * 2;
  tmp = (bheap_node*)realloc(heap->nodes, sizeof(bheap_node) * new_len);
  if (tmp) {
    heap->len = new_len;
    heap->nodes = tmp;
    return TRUE;
  } else {
    return FALSE;
  }
}

int
bheap_push(bheap* heap, int val, void* opt)
{
  bheap_node *node;
  if (heap->tail == heap->len - 1) bheap_extend(heap);
  node = &heap->nodes[++heap->tail];
  node->val = val;
  node->opt = opt;
  bheap_bubble_up(heap);
  return TRUE;
}

int
bheap_is_empty(bheap* heap)
{
  if (heap->tail < 0) {
    return TRUE;
  } else {
    return FALSE;
  }
}

void
bheap_remove_root(bheap* heap)
{
  heap->nodes[0] = heap->nodes[heap->tail--];
  bheap_bubble_down(heap);
}

bheap_node
bheap_pop(bheap* heap)
{
  bheap_node node;
  node = heap->nodes[0];
  bheap_remove_root(heap);
  return node;
}

void
bheap_print(bheap* heap)
{
  int i;
  for (i = 0; i <= heap->tail; i++) {
    printf("%d\n", heap->nodes[i].val);
  }
}

void
bheap_bubble_up(bheap* heap)
{
  int i;
  bheap_node *nodes, tail_node;
  nodes = heap->nodes;
  tail_node = nodes[heap->tail];

  for (i = heap->tail; i > 0 && tail_node.val > nodes[(i-1)/2].val; i = (i-1)/2) {
    nodes[i] = nodes[(i-1)/2];
  }
  nodes[i] = tail_node;
}

void
bheap_bubble_down(bheap* heap)
{
  int parent, child;
  bheap_node *nodes, root_node;
  nodes = heap->nodes;
  root_node = nodes[0];

  parent = 0;
  while (parent <= heap->tail) {
    child = (parent * 2) + 1;
    if (child > heap->tail) break;
    if (child + 1 <= heap->tail && nodes[child].val < nodes[child+1].val)
      child++;
    if (root_node.val < nodes[child].val) {
      nodes[parent] = nodes[child];
      parent = child;
    } else break;
  }
  nodes[parent] = root_node;
}
