#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <vector_types.h>

// calculate the number of threads in a block
#define NUM_THREAD_IN_BLOCK (blockDim.x * blockDim.y * blockDim.z)

// calculate the thread id of given block
#define TID_IN_2D_BLOCK (blockDim.x * threadIdx.y) + threadIdx.x
#define TID_IN_3D_BLOCK                                                        \
  ((blockDim.y * blockDim.x) * threadIdx.z) + TID_IN_2D_BLOCK

// calculate the thread id of given grid
#define TID_IN_1D_GRID (blockIdx.x * NUM_THREAD_IN_BLOCK) + TID_IN_3D_BLOCK
#define TID_IN_2D_GRID                                                         \
  (blockIdx.y * (gridDim.x * NUM_THREAD_IN_BLOCK)) + TID_IN_1D_GRID
#define TID_IN_3D_GRID                                                         \
  (blockIdx.z * (gridDim.y * gridDim.x * NUM_THREAD_IN_BLOCK)) + TID_IN_2D_GRID

#define NUM_DATA 1024000

__global__ void vecAdd(int *_a, int *_b, int *_c) {
  // int tID = threadIdx.x;
  // int current_idx = (blockIdx.x * blockDim.x) + tID;
  int tID = TID_IN_3D_GRID;
  // printf("current_idx: %d\n", current_idx);
  _c[tID] = _a[tID] + _b[tID];
}

int main(void) {
  struct timeval startTime, endTime;
  struct timeval startTime2, endTime2;
  double diffTime;

  int *a, *b, *c, *h_c;
  int *d_a, *d_b, *d_c;

  int memSize = sizeof(int) * NUM_DATA;
  printf("%d elements, memSize = %d bytes \n", NUM_DATA, memSize);
  a = new int[NUM_DATA];
  memset(a, 0, memSize);
  b = new int[NUM_DATA];
  memset(b, 0, memSize);
  c = new int[NUM_DATA];
  memset(c, 0, memSize);
  h_c = new int[NUM_DATA];
  memset(h_c, 0, memSize);

  for (int i = 0; i < NUM_DATA; i++) {
    a[i] = rand() % 10;
    b[i] = rand() % 10;
  }

  gettimeofday(&startTime, NULL);
  for (int i = 0; i < NUM_DATA; i++) {
    h_c[i] = a[i] + b[i];
  }
  gettimeofday(&endTime, NULL);
  diffTime = (endTime.tv_sec - startTime.tv_sec) +
             ((endTime.tv_usec - startTime.tv_usec) / 1000000.0);
  printf("VectorSum on Host: %lf ms\n", diffTime);
  
  cudaMalloc(&d_a, memSize);
  cudaMalloc(&d_b, memSize);
  cudaMalloc(&d_c, memSize);

  gettimeofday(&startTime2, NULL);

  gettimeofday(&startTime, NULL);
  cudaMemcpy(d_a, a, memSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, memSize, cudaMemcpyHostToDevice);
  gettimeofday(&endTime, NULL);
  diffTime = (endTime.tv_sec - startTime.tv_sec) +
             ((endTime.tv_usec - startTime.tv_usec) / 1000000.0);
  printf("Data Trans. : Host -> Device: %lf ms\n", diffTime);

  dim3 block(128);
  dim3 grid((NUM_DATA + block.x - 1) / block.x);

  gettimeofday(&startTime, NULL);
  vecAdd<<<grid, block>>>(d_a, d_b, d_c);
  cudaDeviceSynchronize();
  gettimeofday(&endTime, NULL);
  diffTime = (endTime.tv_sec - startTime.tv_sec) +
             ((endTime.tv_usec - startTime.tv_usec) / 1000000.0);
  printf("Computation(Kernel): %lf ms\n", diffTime);

  gettimeofday(&startTime, NULL);
  cudaMemcpy(c, d_c, memSize, cudaMemcpyDeviceToHost);
  gettimeofday(&endTime, NULL);
  diffTime = (endTime.tv_sec - startTime.tv_sec) +
             ((endTime.tv_usec - startTime.tv_usec) / 1000000.0);
  printf("Data Trans. : Device -> Host: %lf ms\n", diffTime);

  gettimeofday(&endTime2, NULL);
  diffTime = (endTime2.tv_sec - startTime2.tv_sec) +
             ((endTime2.tv_usec - startTime2.tv_usec) / 1000000.0);
  printf("CUDA Total: %lf ms\n", diffTime);

  // check results
  bool result = true;
  for (int i = 0; i < NUM_DATA; i++) {
    if ((a[i] + b[i]) != c[i]) {
      printf("[%d] The resutls is not matched! (%d, %d) \n", i, a[i] + b[i],
             c[i]);
      result = false;
    }
  }
  if (result)
    printf("GPU works well!\n");
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  delete[] a;
  delete[] b;
  delete[] c;
  return 0;
}