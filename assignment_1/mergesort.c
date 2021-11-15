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

void mergesort_th(int *arr, int start, int end, int tlevel)
{
    int t, rc, start2, end2;
    void *status;

    struct m_thread_data td;
    td.arr = arr;
    td.start = start;
    td.end = end;
    td.tlevel = tlevel;

    pthread_t main_thread;
    rc = pthread_create(&main_thread, NULL, mergesort_th_worker,
                        (void *)&td);
    if (rc)
        printf("ERROR; return code from pthread_create() is %d\n", rc);

    rc = pthread_join(main_thread, &status);
    if (rc)
        printf("ERROR; return code from pthread_join() is %d\n", rc);
}

void *mergesort_th_worker(void *threadargs)
{
    int t, rc;
    void *status;

    struct m_thread_data *targs;
    targs = (struct m_thread_data *)threadargs;
    int *arr = targs->arr;
    int start = targs->start;
    int end = targs->end;
    int tlevel = targs->tlevel;

    if (tlevel <= 0 || start == end)
    {
        mergesort(arr, start, end);
        pthread_exit(NULL);
    }

    int mid = start + (end - start) / 2;

    struct m_thread_data td_arr[2];
    for (t = 0; t < 2; t++)
    {
        td_arr[t].arr = arr;
        td_arr[t].tlevel = tlevel - 1;
    }
    td_arr[0].start = start;
    td_arr[0].end = mid;
    td_arr[1].start = mid + 1;
    td_arr[1].end = end;

    pthread_t threads[2];
    for (t = 0; t < 2; t++)
    {
        rc = pthread_create(&threads[t], NULL, mergesort_th_worker,
                            (void *)&td_arr[t]);
        if (rc)
        {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    for (int t = 0; t < 2; t++)
    {
        rc = pthread_join(threads[t], &status);
        if (rc)
        {
            printf("ERROR; return code from pthread_join() is %d\n", rc);
            exit(-1);
        }
    }

    merge(arr, start, mid, end);

    pthread_exit(NULL);
}