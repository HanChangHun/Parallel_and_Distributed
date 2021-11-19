#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/time.h>

using namespace std;

#define NUM_CPU_THREADS 4
#define ROW_SIZE 32
#define K_SIZE 512

#define COL_SIZE 32
#define WORK_LOAD (1024)
#define MAT_SIZE_A (ROW_SIZE * K_SIZE)
#define MAT_SIZE_B (K_SIZE * COL_SIZE)
#define MAT_SIZE_C (ROW_SIZE * COL_SIZE)

// input matrix
float A[ROW_SIZE][K_SIZE];
float B[K_SIZE][COL_SIZE];

float hostC[ROW_SIZE][COL_SIZE];   // host result
float deviceC[COL_SIZE][COL_SIZE]; // device result

#define memsetZero(_P, _type, _size) memset(_P, 0, sizeof(_type) * _size);
#define dMemAlloc(_P, _type, _size) cudaMalloc(&_P, sizeof(_type) * _size);

void genInputMatrices(void);

__global__ void matMul_kernel(float *_A, float *_B, float *_C) {
  int row = threadIdx.y;
  int col = threadIdx.x;
  int index = row * blockDim.x + col;

  _C[index] = 0;

  for (int k = 0; k < K_SIZE; k++)
    for (int i = 0; i < WORK_LOAD; i++)
      _C[index] += _A[row * K_SIZE + k] * _B[col + k * COL_SIZE];
}

// __global__ void matMul_kernel_shared(float *_A, float *_B, float *_C) {
//   int row = threadIdx.y;
//   int col = threadIdx.x;
//   int index = row * blockDim.x + col;

//   __shared__ float sA[ROW_SIZE][K_SIZE]; // 32 * 256 * 4 bytes = 16 KB
//   __shared__ float sB[K_SIZE][COL_SIZE]; // 16 KB

//   for (int k = 0; k < K_SIZE; k++) {
//     sA[row][k] = _A[row * K_SIZE + k];
//     sB[k][col] = _B[col + k * COL_SIZE];
//   }
//   __syncthreads(); // wait until all thread load the matrix
//   _C[index] = 0;
//   for (int k = 0; k < K_SIZE; k++)
//     for (int i = 0; i < WORK_LOAD; i++)
//       _C[index] += sA[row][k] * sB[k][col];
// }

__global__ void matMul_kernel_shared_C(float *_A, float *_B, float *_C) {
  int row = threadIdx.y;
  int col = threadIdx.x;
  int index = row * blockDim.x + col;

  float sC = 0;  // add register

  for (int k = 0; k < K_SIZE; k++)
    for (int i = 0; i < WORK_LOAD; i++)
      sC += _A[row * K_SIZE + k] * _B[col + k * COL_SIZE];

  _C[index] = sC;
}

void calculate_time(struct timeval startTime, string msg) {
  double diffTime;
  struct timeval endTime;

  gettimeofday(&endTime, NULL);

  diffTime = (endTime.tv_sec - startTime.tv_sec) +
             ((endTime.tv_usec - startTime.tv_usec) / 1000000.0);

  printf("%s: %lf ms\n", msg.c_str(), diffTime);
}

int main(void) {
  struct timeval startTime;

  float *dA, *dB, *dC;
  dA = dB = dC = NULL;

  memsetZero(A, float, MAT_SIZE_A);
  memsetZero(B, float, MAT_SIZE_B);
  memsetZero(hostC, float, MAT_SIZE_C);
  memsetZero(deviceC, float, MAT_SIZE_C);

  // device memory allocaiton
  dMemAlloc(dA, float, MAT_SIZE_A);
  dMemAlloc(dB, float, MAT_SIZE_B);
  dMemAlloc(dC, float, MAT_SIZE_C);

  genInputMatrices();

  // Host code
  gettimeofday(&startTime, NULL);
  for (int r = 0; r < ROW_SIZE; r++)
    for (int c = 0; c < COL_SIZE; c++)
      for (int k = 0; k < K_SIZE; k++)
        for (int i = 0; i < WORK_LOAD; i++)
          hostC[r][c] += A[r][k] * B[k][c];
  calculate_time(startTime, "CPU code");

  // Copy input matrices : H -> D
  gettimeofday(&startTime, NULL);
  cudaMemcpy(dA, A, sizeof(float) * MAT_SIZE_A, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sizeof(float) * MAT_SIZE_B, cudaMemcpyHostToDevice);
  // calculate_time(startTime, "[Data transter] host->device");

  dim3 blockDim(COL_SIZE, ROW_SIZE);

  // cudaFuncSetCacheConfig(matMul_kernel, cudaFuncCachePreferNone);
  // cudaFuncSetCacheConfig(matMul_kernel, cudaFuncCachePreferShared);
  // cudaFuncSetCacheConfig(matMul_kernel, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(matMul_kernel, cudaFuncCachePreferEqual);

  gettimeofday(&startTime, NULL);
  matMul_kernel<<<1, blockDim>>>(dA, dB, dC);
  cudaDeviceSynchronize();
  calculate_time(startTime, "Kernel launch");

  // // Kernel call (shared memory)
  // gettimeofday(&startTime, NULL);
  // matMul_kernel_shared<<<1, blockDim>>>(dA, dB, dC);
  // cudaDeviceSynchronize();
  // calculate_time(startTime, "Kernel launch(shared version)");

  // Kernel call (shared memory C version)
  gettimeofday(&startTime, NULL);
  matMul_kernel_shared_C<<<1, blockDim>>>(dA, dB, dC);
  cudaDeviceSynchronize();
  calculate_time(startTime, "Kernel launch(shared version)");

  // Get back result : D -> H
  gettimeofday(&startTime, NULL);
  cudaMemcpy(deviceC, dC, sizeof(float) * MAT_SIZE_C, cudaMemcpyDeviceToHost);
  // calculate_time(startTime, "[Data transfer] device->host");

  // check the results
  bool isCorrect = true;
  float *pHostC = &hostC[0][0];
  float *pDeviceC = &deviceC[0][0];
  for (int i = 0; i < MAT_SIZE_C; i++) {
    if (pHostC[i] != pDeviceC[i]) {
      printf("[%d] %.2f, %.2f\n", i, pHostC[i], pDeviceC[i]);
      isCorrect = false;
      break;
    }
  }

  if (isCorrect)
    printf("Result is correct!\n");
  else
    printf("Result is not correct!!!!!!\n");
}

void genInputMatrices(void) {
  for (int r = 0; r < ROW_SIZE; r++)
    for (int k = 0; k < K_SIZE; k++)
      A[r][k] = rand() % 100;
  for (int k = 0; k < K_SIZE; k++)
    for (int c = 0; c < COL_SIZE; c++)
      B[k][c] = rand() % 100;
}

// void setTimer(void) {
//   timer = new DS_timer(NUM_TIMER);
//   timer->initTimers();
//   timer->setTimerName(TIMER_HOST, "CPU code");
//   timer->setTimerName(TIMER_KERNEL, "Kernel launch");
//   timer->setTimerName(TIMER_KERNEL_SH, "Kernel launch (shared ver.)");
//   timer->setTimerName(TIMER_HtoD, "[Data transter] host->device");
//   timer->setTimerName(TIMER_DtoH, "[Data transfer] device->host");
// }