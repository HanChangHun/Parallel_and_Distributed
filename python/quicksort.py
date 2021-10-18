from threading import Thread


def quicksort(arr, start, end):
    pivot_idx = end
    i = start

    if start >= end:
        return 0

    while i < pivot_idx:
        pivot = arr[pivot_idx]

        if arr[i] < pivot:
            i += 1
        else:
            arr[pivot_idx] = arr[i]
            pivot_idx -= 1
            arr[i] = arr[pivot_idx]
            arr[pivot_idx] = pivot

    if start < i:
        quicksort(arr, start, i - 1)
    if i + 1 < end:
        quicksort(arr, i + 1, end)

    return i


def quicksort_th_dy(arr, start, end):
    pivot_idx = end
    i = start

    if start >= end:
        return 0

    while i < pivot_idx:
        pivot = arr[pivot_idx]

        if arr[i] < pivot:
            i += 1
        else:
            arr[pivot_idx] = arr[i]
            pivot_idx -= 1
            arr[i] = arr[pivot_idx]
            arr[pivot_idx] = pivot

    if start < i:
        lthread = Thread(target=lambda: quicksort(arr, start, i - 1))
        lthread.start()
    if i + 1 < end:
        rthread = Thread(target=lambda: quicksort(arr, i + 1, end))
        rthread.start()

    lthread.join()
    rthread.join()

    return i