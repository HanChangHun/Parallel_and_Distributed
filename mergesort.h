#pragma once
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>


struct m_thread_data{
	int *arr;
	int start;
	int end;
};

void merge(int* arr, int start, int mid, int end);
void mergesort(int* arr, int start, int end);
void mergesort_th(int *arr, int start, int end, int num_th);
void *mergesort_th_worker(void *threadargs);
