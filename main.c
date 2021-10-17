/****************************************************************************
* Assignment 1: Pthread implementation of parallel sorting algorithms
* Hoeseok Yang
* 2014. 10
****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "sorts.h"

#define INPUT_SIZE_LOG 24
#define INPUT_SIZE (1<<INPUT_SIZE_LOG)

int * input_array;
int * output_array;

int verify()
{
  int i;
  for(i=1;i<INPUT_SIZE;i++) {
    if(output_array[i-1]>output_array[i]) {
      printf("oops, verification failed at index %d (%d's value is %d and %d's value is %d)\n",
             i-1,i-1,output_array[i-1],i,output_array[i]);
      return 0;
    }
  }
  return 1;
}

void gen_sequence()
{
  int i;
  srand(time(NULL));
  for(i=0;i<INPUT_SIZE;i++)
    output_array[i] = input_array[i] = rand();
  return;
}

int quicksort(int * arr, int start, int end)
{
  int pivot_idx = end;
  int pivot;
  int i=start,j;

  //printf("quicksort called: start %d end %d\n",start, end);
  //for(j=start;j<=end;j++) printf("%d ",arr[j]); printf("\n");

  if (start>=end) return 0;

  /* partition: "i" indicates current position*/
  while(i<pivot_idx) {
    /* pivot selection */
    pivot = arr[pivot_idx];
    //printf("-----------------------------------------\n");
    //printf("i is %d, pivot index is %d\n",i,pivot_idx);

    if(arr[i]<pivot) i++;
    else {
      /* triangle swap of arr[i], arr[pivot_idx-1], and pivot */
      arr[pivot_idx--] = arr[i];
      arr[i] = arr[pivot_idx];
      arr[pivot_idx] = pivot;
    }
    //printf("at iteration %d\n",i);
    //for(j=start;j<=end;j++) if(j==pivot_idx)printf("\"%d\" ",arr[j]);else printf("%d ",arr[j]); printf("\n");
  }
  
  //printf("after partitioning: i %d\n",i);
  //for(j=0;j<INPUT_SIZE;j++) printf("%d ",arr[j]); printf("\n");

  /* recursive quicksort */
  if (start < i) quicksort(arr, start, i-1);
  if (i+1 < end) quicksort(arr, i+1, end);

  return i;
}


int main(int argc, char* argv[])
{
  int i;
  double time_spent = 0.0;

  for(i=0; i<argc; ++i)
    printf("Argument %d : %s\n", i, argv[i]);

  printf("========================================\n");
  printf("SORTING ASSIGNMENT, SIZE=%d\n",INPUT_SIZE);
  printf("========================================\n");
  /* memory allocation */
  input_array = (int *) malloc(sizeof(int)*INPUT_SIZE);
  output_array = (int *) malloc(sizeof(int)*INPUT_SIZE);

  /* random number generation */
  gen_sequence();
    
  /* your sorting algorithm */
  clock_t begin = clock();
  if (strcmp(argv[1], "quicksort") == 0)
    quicksort(output_array, 0, INPUT_SIZE -1);
  else
  {
    printf("There is no selected method.\n");
    return 1;
  }
  clock_t end = clock();

  time_spent += (double)(end - begin) / CLOCKS_PER_SEC;

  printf("The elapsed time is %f seconds\n", time_spent);

  /* verification */
  if(verify()) printf("sorting verified!\n");
  else printf("verification failed!\n");

  /* memory deallocation */
  free(input_array);
  free(output_array);

  return 0;
  
}
