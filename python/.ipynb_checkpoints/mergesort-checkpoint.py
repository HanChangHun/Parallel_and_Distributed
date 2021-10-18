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
    for i in range(num_th):
        low = thread_part * (MAX / 4)
        high = (thread_part + 1) * (MAX / 4) - 1

    mid = low + (high - low) / 2
    if low < high:
        merge_sort(low, mid)
        merge_sort(mid + 1, high)
        merge(low, mid, high)