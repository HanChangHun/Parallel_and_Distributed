/****************************************************************************
* Assignment 1: Pthread implementation of parallel sorting algorithms
* Hoeseok Yang
* 2014. 10
****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <pthread.h>

#include <sys/time.h>
#include "quicksort.h"
#include "mergesort.h"
#include "bucketsort.h"

#define INPUT_SIZE_LOG 24
#define INPUT_SIZE (1 << INPUT_SIZE_LOG)
#define THREAD_LEVEL 10

int *input_array;
int *output_array;

int verify()
{
  int i;
  for (i = 1; i < INPUT_SIZE; i++)
  {
    if (output_array[i - 1] > output_array[i])
    {
      printf("oops, verification failed at index %d (%d's value is %d and %d's value is %d)\n",
             i - 1, i - 1, output_array[i - 1], i, output_array[i]);
      return 0;
    }
  }
  return 1;
}

void gen_sequence()
{
  int i;
  srand(time(NULL));
  for (i = 0; i < INPUT_SIZE; i++)
    output_array[i] = input_array[i] = rand();
  return;
}

int main(int argc, char *argv[])
{
  struct timeval start, end;
  double diff;

  printf("========================================\n");
  printf("SORTING ASSIGNMENT, SIZE=%d\n", INPUT_SIZE);
  printf("========================================\n");
  printf("Sort algorithms: %s\n", argv[1]);
  /* memory allocation */
  input_array = (int *)malloc(sizeof(int) * INPUT_SIZE);
  output_array = (int *)malloc(sizeof(int) * INPUT_SIZE);

  /* random number generation */
  gen_sequence();

  /* your sorting algorithm */
  gettimeofday(&start, NULL);
  if (strcmp(argv[1], "quicksort") == 0)
    quicksort(output_array, 0, INPUT_SIZE - 1);
  else if (strcmp(argv[1], "quicksort_th") == 0)
    quicksort_th(output_array, 0, INPUT_SIZE - 1, THREAD_LEVEL);
  else if (strcmp(argv[1], "mergesort") == 0)
    mergesort(output_array, 0, INPUT_SIZE - 1);
  else if (strcmp(argv[1], "mergesort_th") == 0)
    mergesort_th(output_array, 0, INPUT_SIZE - 1, THREAD_LEVEL);
  else if (strcmp(argv[1], "bucketsort") == 0)
    bucketsort(output_array, 0, INPUT_SIZE - 1);
  else
  {
    printf("There is no selected method.\n");
    return 1;
  }
  gettimeofday(&end, NULL);

  /* verification */
  if (verify())
    printf("sorting verified!\n");
  else
    printf("verification failed!\n");

  /* memory deallocation */
  free(input_array);
  free(output_array);

  return 0;
}
