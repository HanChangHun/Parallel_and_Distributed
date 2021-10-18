#include "quicksort.h"

int quicksort(int *arr, int start, int end)
{
    int pivot_idx = end;
    int pivot;
    int i = start, j;

    //printf("quicksort called: start %d end %d\n",start, end);
    //for(j=start;j<=end;j++) printf("%d ",arr[j]); printf("\n");

    if (start >= end)
        return 0;

    /* partition: "i" indicates current position*/
    while (i < pivot_idx)
    {
        /* pivot selection */
        pivot = arr[pivot_idx];
        //printf("-----------------------------------------\n");
        //printf("i is %d, pivot index is %d\n",i,pivot_idx);

        if (arr[i] < pivot)
            i++;
        else
        {
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
    if (start < i)
        quicksort(arr, start, i - 1);
    if (i + 1 < end)
        quicksort(arr, i + 1, end);

    return i;
}

int quicksort_th(int *arr, int start, int end, int tlevel)
{
    int rc;
    void *status;

    struct q_thread_data td;
    td.arr = arr;
    td.start = start;
    td.end = end;
    td.level = tlevel;

    pthread_t main_thread;
    rc = pthread_create(&main_thread, NULL, quicksort_th_worker,
                        (void *)&td);
    if (rc)
        printf("ERROR; return code from pthread_create() is %d\n", rc);

    rc = pthread_join(main_thread, &status);
    if (rc)
        printf("ERROR; return code from pthread_join() is %d\n", rc);
}

void *quicksort_th_worker(void *threadargs)
{
    int t, rc;
    void *status;

    struct q_thread_data *targs;
    targs = (struct q_thread_data *)threadargs;

    if (targs->level <= 0 || targs->start == targs->end)
    {
        //We have plenty of threads, finish with sequential.
        quicksort(targs->arr, targs->start, targs->end);
        pthread_exit(NULL);
    }

    int pivot_idx = targs->end;
    int pivot;
    int i = targs->start, j;
    int *arr = targs->arr;

    if (i >= pivot_idx)
        return 0;

    /* partition: "i" indicates current position*/
    while (i < pivot_idx)
    {
        /* pivot selection */
        pivot = arr[pivot_idx];

        if (arr[i] < pivot)
            i++;
        else
        {
            /* triangle swap of arr[i], arr[pivot_idx-1], and pivot */
            arr[pivot_idx--] = arr[i];
            arr[i] = arr[pivot_idx];
            arr[pivot_idx] = pivot;
        }
    }

    struct q_thread_data td_arr[2];
    for (t = 0; t < 2; t++)
    {
        td_arr[t].arr = targs->arr;
        td_arr[t].level = targs->level - 1;
    }
    td_arr[0].start = targs->start;
    td_arr[0].end = i - 1;
    td_arr[1].start = i + 1;
    td_arr[1].end = targs->end;

    /* recursive quicksort with thread */
    pthread_t threads[2];
    for (t = 0; t < 2; t++)
    {
        rc = pthread_create(&threads[t], NULL, quicksort_th_worker,
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

    pthread_exit(NULL);
}