from threading import Thread
import numpy as np
from quicksort import quicksort


def bucketsort(arr, start, end):
    # make buckets
    buckets =  [[] for _ in range(len(arr))]
    
    # assign values
    for value in arr:
        bucket_index = value * len(arr) // (max(arr) + 1)
        buckets[bucket_index].append(value)
    
    # sort & merge
    idx = 0
    sorted_list = []
    for bucket in buckets:
        if bucket is not []:
            quicksort(bucket, 0, len(bucket)-1)
            for i in bucket:
                arr[idx] = i
                idx += 1


def bucketsort_th_st(arr, start, end, num_th):
    # make buckets
    MAX = max(arr)
    size = end - start + 1
    buckets =  [[] for _ in range(len(arr))]
    
    # assign values
    for i in range(num_th):
        start2 = i * int(size / num_th)
        end2 = (i + 1) * int(size / num_th) - 1
        arr2 = arr[start2:end2+1]

        for value in arr2:
            bucket_index = value * len(arr) // (MAX + 1)
            buckets[bucket_index].append(value)
    
    # sort & merge
    idx = 0
    # for i in range(num_th):
    #     start2 = i * int(size / num_th)
    #     end2 = (i + 1) * int(size / num_th) - 1
    for i in range(len(buckets)):
        quicksort(buckets[i], 0, len(buckets[i])-1)
        for value in buckets[i]:
            arr[idx] = value
            idx += 1
