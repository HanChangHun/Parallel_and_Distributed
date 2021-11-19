/******************************************************
  File Name [matrixmul.cu]
  Synopsis [This file defines the main function to do
  matrix-matrixmultiplication.]
  Description []
 *******************************************************/
//––––––––––––––––––––––––––––––––––––––––––––––––––––––
// Included C libraries
//––––––––––––––––––––––––––––––––––––––––––––––––––––––
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//––––––––––––––––––––––––––––––––––––––––––––––––––––––
// Included CUDA libraries
//––––––––––––––––––––––––––––––––––––––––––––––––––––––
#include <helper_functions.h>

//––––––––––––––––––––––––––––––––––––––––––––––––––––––
// Included helper functions
//––––––––––––––––––––––––––––––––––––––––––––––––––––––
#include "assist.h"
//––––––––––––––––––––––––––––––––––––––––––––––––––––––
// Included host matrix-matrix multiplication function prototype
//––––––––––––––––––––––––––––––––––––––––––––––––––––––
#include "matrixmul.h"
/*-----------------------------------------*/
/* */
/* Synopsis [Main function] */
/* Description [] */
/* */
/*-----------------------------------------*/

#define TILE_WIDTH 2

__global__ void MatrixMul_kernel(float *Md, float *Nd, float *Pd, int Width) {
  int row = blockDim.x * blockIdx.x + threadIdx.x;
  int col = blockDim.y * blockIdx.y + threadIdx.y;

  float Pvalue = 0;

  for (int i = 0; i < Width; i++) {
    Pvalue += Md[row * Width + i] * Nd[i * Width + col];
  }

  Pd[row * Width + col] = Pvalue;
}

__global__ void MatrixMul_kernel_shared(float *Md, float *Nd, float *Pd,
                                        int Width) {
  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Identify the row and column of the Pd element to work on
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;
  float Pvalue = 0;

  // Loop over the Md and Nd tiles required to compute the Pd element
  for (int m = 0; m < Width / TILE_WIDTH; ++m) {
    // Coolaborative loading of Md and Nd tiles into shared memory
    Mds[ty][tx] = Md[Row * Width + (m * TILE_WIDTH + tx)];
    Nds[ty][tx] = Nd[Col + (m * TILE_WIDTH + ty) * Width];
    __syncthreads();
    for (int k = 0; k < TILE_WIDTH; ++k)
      Pvalue += Mds[ty][k] * Nds[k][tx];
    __syncthreads();
  }

  Pd[Row * Width + Col] = Pvalue;
}

int main(int argc, char **argv) {
  bool if_quiet = false;
  int i, j;
  char *matrix_id = NULL, *input_fn = NULL, *gold_fn = NULL;
  int Mw = 0, Mh = 0, Nw = 0, Nh = 0, Pw = 0, Ph = 0;

  if (argc == 2) {
    matrix_id = strdup(argv[1]);
  } else {
    fprintf(stderr, "Error: Wrong input parameter numbers.\n");
    fprintf(stderr, "Usage:\n"
                    "$> ./lab1.1-matrixmul <8, 128, 512, 3072, 4096>\n"
                    "Examples:\n"
                    " $> ./lab1.1-matrixmul 128\n");
    exit(1);
  }

  Mw = Mh = Nw = Nh = Pw = Ph = atoi(matrix_id);
  input_fn = (char *)malloc(30 * sizeof(char));
  gold_fn = (char *)malloc(30 * sizeof(char));
  sprintf(input_fn, "matrix_%s.bin", matrix_id);
  sprintf(gold_fn, "matrix_%s.gold", matrix_id);

  if (Pw * Ph > 15 * 15) {
    if_quiet = true; // If not display matrix contents
  }

  printf("Input matrix size: %d by %d\n", Mw, Mh);

  //––––––––––––––––––––––––––––––––––––––––––––––––––––
  // Setup host side
  //––––––––––––––––––––––––––––––––––––––––––––––––––––
  printf("Setup host side environment:\n");

  // allocate host memory for matrices M and N
  printf(" Allocate host memory for matrices M and N.\n");
  printf(" M: %d x %d\n", Mw, Mh);
  printf(" N: %d x %d\n", Nw, Nh);

  unsigned int size_M = Mw * Mh;
  unsigned int mem_size_M = sizeof(float) * size_M;
  float *hostM = (float *)malloc(mem_size_M);

  unsigned int size_N = Nw * (Nh);
  unsigned int mem_size_N = sizeof(float) * size_N;
  float *hostN = (float *)malloc(mem_size_N);

  // allocate memory for the result on host side
  printf(" Allocate memory for the result on host side.\n");
  unsigned int size_P = Pw * Ph;
  unsigned int mem_size_P = sizeof(float) * size_P;
  float *hostP = (float *)malloc(mem_size_P);

  // Initialize the input matrices.
  printf(" Generate input matrix data for matrix M and N.\n");
  GenMatrixFile(input_fn, Pw, Ph, if_quiet);
  unsigned int *matrix = ReadMatrixFile(input_fn, Pw, Ph, true);
  for (i = 0; i < Mw; i++)
    for (j = 0; j < Nw; j++)
      hostM[i * Mw + j] = hostN[i * Mw + j] = (float)matrix[i * Mw + j];
  free(matrix);
  matrix = NULL;

  //---------------------------------------
  // Do matrix-matrix multiplication
  //---------------------------------------
  printf(" Computing matrix multiplication M x N:\n");
  if (Pw * Ph > 512 * 512) {
    printf(" (It takes time since matrix is larger than 512by512.\n");
  }
  // Measurement executtion time
  StopWatchInterface *cpu_timer = 0, *gpu_timer1 = 0, *gpu_timer2 = 0,
                     *copy_timer1 = 0, *copy_timer2 = 0;

  sdkCreateTimer(&cpu_timer);
  sdkStartTimer(&cpu_timer);

  float *reference = (float *)malloc(mem_size_P);
  computeGold(reference, hostM, hostN, Mh, Mw, Nw);
  sdkStopTimer(&cpu_timer);

  printf(" CPU Processing time : %f (ms)\n", sdkGetTimerValue(&cpu_timer));

  sdkDeleteTimer(&cpu_timer);
  printf(" Matrix data checksum : %g\n", CheckSum(reference, Mw, Nw));
  if (!if_quiet) {
    printf(" Matrix data contents :\n");
    printf(" ");
  }
  matrix = (unsigned int *)malloc(Pw * Ph * sizeof(unsigned int));
  for (i = 0; i < Ph; i++) {
    for (j = 0; j < Pw; j++) {
      matrix[i * Pw + j] = (unsigned int)reference[i * Pw + j];
      if (!if_quiet)
        printf("%u ", matrix[i * Pw + j]);
    }
    if (!if_quiet)
      printf("\n ");
  }
  if (!if_quiet)
    printf("\n");

  WriteMatrixFile(gold_fn, matrix, Pw, Ph, 1);
  free(matrix);
  matrix = NULL;
  free(reference);

  //---------------------------------------
  // Do matrix-matrix multiplication with CUDA
  //---------------------------------------
  float *deviceM, *deviceN, *deviceP;

  cudaMalloc(&deviceM, size_M * sizeof(float));
  cudaMalloc(&deviceN, size_N * sizeof(float));
  cudaMalloc(&deviceP, size_P * sizeof(float));

  cudaMemset(deviceM, 0, size_M * sizeof(float));
  cudaMemset(deviceN, 0, size_N * sizeof(float));
  cudaMemset(deviceP, 0, size_P * sizeof(float));

  sdkCreateTimer(&copy_timer1);
  sdkStartTimer(&copy_timer1);
  cudaMemcpy(deviceM, hostM, size_M * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceN, hostN, size_N * sizeof(float), cudaMemcpyHostToDevice);
  sdkStopTimer(&copy_timer1);
  printf("\n\n host to device memory copy time : %f (ms)\n",
         sdkGetTimerValue(&copy_timer1));
  sdkDeleteTimer(&copy_timer1);

  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  dim3 dimGrid(ceil((float)Mw / TILE_WIDTH), ceil((float)Nh / TILE_WIDTH));

  float *reference1 = (float *)malloc(mem_size_P);
  sdkCreateTimer(&gpu_timer1);
  sdkStartTimer(&gpu_timer1);
  MatrixMul_kernel<<<dimGrid, dimBlock>>>(deviceM, deviceN, deviceP, Mw);
  cudaDeviceSynchronize();
  sdkStopTimer(&gpu_timer1);
  printf(" GPU Processing time(kernel) : %f (ms)\n",
         sdkGetTimerValue(&gpu_timer1));
  sdkDeleteTimer(&gpu_timer1);

  float *reference2 = (float *)malloc(mem_size_P);
  sdkCreateTimer(&gpu_timer2);
  sdkStartTimer(&gpu_timer2);
  MatrixMul_kernel_shared<<<dimGrid, dimBlock>>>(deviceM, deviceN, deviceP, Mw);
  cudaDeviceSynchronize();
  sdkStopTimer(&gpu_timer2);
  printf(" GPU Processing time(kernel_shared tile: %d) : %f (ms)\n", TILE_WIDTH,
         sdkGetTimerValue(&gpu_timer2));
  sdkDeleteTimer(&gpu_timer2);
  cudaMemcpy(reference2, deviceP, size_P * sizeof(float),
             cudaMemcpyDeviceToHost); // skip check this memory copy time.

  sdkCreateTimer(&copy_timer2);
  sdkStartTimer(&copy_timer2);
  cudaMemcpy(reference1, deviceP, size_P * sizeof(float),
             cudaMemcpyDeviceToHost);
  sdkStopTimer(&copy_timer2);
  printf(" device to host memory copy time : %f (ms)\n",
         sdkGetTimerValue(&copy_timer2));
  sdkDeleteTimer(&copy_timer2);

  printf(" reference1, reference2 checksum : %g, %g\n",
         CheckSum(reference1, Mw, Nw), CheckSum(reference2, Mw, Nw));

  // clean up memory
  free(hostM);
  free(hostN);
  free(hostP);
  free(input_fn);
  free(gold_fn);

  return 0;
}
