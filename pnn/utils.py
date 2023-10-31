import numpy as np


def range_chunks(length, n):
    for i in range(0, length, n):
        if i + n < length:
            yield i, i + n
        else:
            yield i, length


def save_array(filepath, array):
    np.save(filepath, array, allow_pickle=False)
