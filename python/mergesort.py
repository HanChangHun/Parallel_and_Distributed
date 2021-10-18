from threading import Thread
import numpy as np


def merge(arr, low, mid, high):
    left = np.zeros([mid - low + 1], dtype=int)
    right = np.zeros([high - mid], dtype=int)

    n1 = mid - low + 1
    n2 = high - mid

    for i in range(n1):
        left[i] = arr[i + low]

    for i in range(n2):
        right[i] = arr[i + mid + 1]

    k = low
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


def merge_sort(arr, low, high):
    mid = int(low + (high - low) // 2)
    if low < high:
        merge_sort(arr, low, mid)
        merge_sort(arr, mid + 1, high)
        merge(arr, low, mid, high)


def merge_sort_th_st(arr, low, high, num_th):
    size = high - low
    mid = int(low + size // 2)

    threads = []
    for i in range(num_th):
        low2 = i * int(size / num_th)
        high2 = (i + 1) * int(size / num_th) - 1

        th = Thread(target=lambda: merge_sort(arr, low2, high2))
        th.start()

        threads.append(th)

    for th in threads:
        th.join()

    if low < high:
        merge_sort(arr, low, mid)
        merge_sort(arr, mid + 1, high)
        merge(arr, low, mid, high)


def merge_sort_th_dy(arr, low, high):
    size = high - low
    mid = int(low + size // 2)

    if low < high:
        lthread = Thread(target=lambda: merge_sort(arr, low, mid))
        lthread.start()
        rthread = Thread(target=lambda: merge_sort(arr, mid + 1, high))
        rthread.start()
    
    lthread.join()
    rthread.join()
    
    merge(arr, low, mid, high)