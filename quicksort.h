#pragma once
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

struct thread_data{
	int *arr;
	int start;
	int end;
    int level;
};

int quicksort(int *arr, int start, int end);
int quicksort_th_dy(int *arr, int start, int end, int tlevel);
void *quicksort_th_dy_worker(void *threadarg);