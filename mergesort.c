#include "mergesort.h"

void merge(int *arr, int start, int mid, int end)
{
    int n1 = mid - start + 1, n2 = end - mid, i, j;
    int *left = (int *)malloc(sizeof(int) * n1);
    int *right = (int *)malloc(sizeof(int) * n2);

    for (i = 0; i < n1; i++)
        left[i] = arr[i + start];

    for (i = 0; i < n2; i++)
        right[i] = arr[i + mid + 1];

    int k = start;
    i = j = 0;

    while (i < n1 && j < n2)
    {
        if (left[i] <= right[j])
            arr[k++] = left[i++];
        else
            arr[k++] = right[j++];
    }

    while (i < n1)
        arr[k++] = left[i++];

    while (j < n2)
        arr[k++] = right[j++];
}

void mergesort(int *arr, int start, int end)
{
    int mid = start + (end - start) / 2;
    if (start < end)
    {
        mergesort(arr, start, mid);
        mergesort(arr, mid + 1, end);
        merge(arr, start, mid, end);
    }
}

void mergesort_th(int *arr, int start, int end, int num_th)
{
    int t, rc;
    void *status;

    printf("1\n");

    struct m_thread_data td_arr[num_th];
    for (t = 0; t < num_th; t++)
    {
        td_arr[t].arr = arr;
        td_arr[t].start = t * ((end - start) / 4);
        td_arr[t].end = (t + 1) * ((end - start) / 4) - 1;
    }

    printf("2\n");

    pthread_t threads[num_th];
    for (t = 0; t < num_th; t++)
    {
        printf("Thread start: %d", t);
        rc = pthread_create(&threads[t], NULL, mergesort_th_worker,
                            (void *)&td_arr[t]);
        if (rc)
            printf("ERROR; return code from pthread_create() is %d\n", rc);
    }

    printf("3\n");

    for (t = 0; t < num_th; t++)
    {
        rc = pthread_join(threads[t], &status);
        if (rc)
            printf("ERROR; return code from pthread_join() is %d\n", rc);
    }
}

void *mergesort_th_worker(void *threadargs)
{
    struct m_thread_data *targs;
    targs = (struct m_thread_data *)threadargs;

    mergesort(targs->arr, targs->start, targs->end);

    pthread_exit(NULL);
}