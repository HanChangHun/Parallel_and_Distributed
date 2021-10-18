from threading import Thread
import numpy as np


def merge(arr, start, mid, end):
    left = np.zeros([mid - start + 1], dtype=int)
    right = np.zeros([end - mid], dtype=int)

    n1 = mid - start + 1
    n2 = end - mid

    for i in range(n1):
        left[i] = arr[i + start]

    for i in range(n2):
        right[i] = arr[i + mid + 1]

    k = start
    i = j = 0

    while i < n1 and j < n2:
        if left[i] <= right[j]:
            arr[k] = left[i]
            k += 1
            i += 1
        else:
            arr[k] = right[j]
            k += 1
            j += 1

    while i < n1:
        arr[k] = left[i]
        k += 1
        i += 1

    while j < n2:
        arr[k] = right[j]
        k += 1
        j += 1


def merge_sort(arr, start, end):
    mid = int(start + (end - start) // 2)
    if start < end:
        merge_sort(arr, start, mid)
        merge_sort(arr, mid + 1, end)
        merge(arr, start, mid, end)


def merge_sort_th_st(arr, start, end, num_th):
    size = end - start + 1
    mid = int(start + size // 2)

    threads = []
    for i in range(num_th):
        start2 = i * int(size / num_th)
        end2 = (i + 1) * int(size / num_th) - 1

        th = Thread(target=lambda: merge_sort(arr, start2, end2))
        th.start()
        threads.append(th)

    for th in threads:
        th.join()

    if start < end:
        merge_sort(arr, start, mid)
        merge_sort(arr, mid + 1, end)
        merge(arr, start, mid, end)


def merge_sort_th_dy(arr, start, end):
    size = end - start + 1
    mid = int(start + size // 2)

    if start < end:
        lthread = Thread(target=lambda: merge_sort(arr, start, mid))
        lthread.start()
        rthread = Thread(target=lambda: merge_sort(arr, mid + 1, end))
        rthread.start()
    
    lthread.join()
    rthread.join()
    
    merge(arr, start, mid, end)