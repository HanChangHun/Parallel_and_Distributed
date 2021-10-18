#pragma once
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>


struct q_thread_data{
	int *arr;
	int start;
	int end;
    int level;
};

int quicksort(int *arr, int start, int end);
int quicksort_th(int *arr, int start, int end, int tlevel);
void *quicksort_th_worker(void *threadarg);