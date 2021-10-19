#include "bucketsort.h"

void bucketsort(int *arr, int start, int end)
{
    int i, j=0;
    int size = end - start + 1;
    int* bucket = (int*)malloc(sizeof(int)*size);

    // initialize counters.
    for( i=0 ; i<size ; i++ )
        bucket[i] = 0;
}