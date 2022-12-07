from libc.math cimport round
import numpy as np
cimport numpy as cnp

def rounding(float n):
    return round(n)

# return type is a C struct of 2 values - this should be quick...

def minmax(double [:] arr):
    cdef double min = np.inf
    cdef double max = -np.inf
    cdef int i
    for i in range(arr.shape[0]):
        if arr[i] < min:
            min = arr[i]
        if arr[i] > max:
            max = arr[i]
    return min, max


# cdef (double, double) minmax(double [:] arr):
#     cdef double min = np.inf
#     cdef double max = -np.inf
#     cdef int i
#     for i in range(arr.shape[0]):
#         if arr[i] < min:
#             min = arr[i]
#         if arr[i] > max:
#             max = arr[i]
#     return min, max
