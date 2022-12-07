#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np
from Tools.MyUtils.my_utils import rounding, minmax
cimport cython
from cython.parallel import prange

cimport numpy as cnp

@cython.boundscheck(False)
@cython.wraparound(False)
def original_lines(cnp.ndarray masked_arena_pixels,
          cnp.ndarray m,
          cnp.ndarray c,
          cnp.ndarray xmin,
          cnp.ndarray xmax,
          cnp.ndarray ymin,
          cnp.ndarray ymax):

    cdef cnp.ndarray[long, ndim=1] x_len = np.rint(xmax[:, 0] - xmin[:, 0]).astype(int)

    cdef cnp.ndarray[long, ndim=1] y_len = np.rint(ymax[:, 0] - ymin[:, 0]).astype(int)

    cdef long n_photoreceptors = m.shape[0]
    cdef Py_ssize_t i
    cdef cnp.ndarray[double, ndim=2] readings = np.zeros((120, 3))

    # Declarations
    cdef cnp.ndarray[double, ndim=2] x_ranges
    cdef cnp.ndarray[double, ndim=2] y_ranges
    cdef cnp.ndarray[double, ndim=2] y_values
    cdef cnp.ndarray[double, ndim=3] set_1
    cdef cnp.ndarray[double, ndim=2] x_values
    cdef cnp.ndarray[double, ndim=3] set_2
    cdef cnp.ndarray[double, ndim=3] full_set
    cdef cnp.ndarray[long, ndim=2] full_set_reshaped
    cdef cnp.ndarray[double, ndim=2] used_pixels
    cdef cnp.ndarray[double, ndim=1] total_sum

    for i in range(n_photoreceptors): #prange(n_photoreceptors, nogil=True):
        x_ranges = np.linspace(xmin[i], xmax[i], x_len[i])
        y_ranges = np.linspace(ymin[i], ymax[i], y_len[i])

        y_values = (m[i] * x_ranges) + c[i]
        y_values = np.floor(y_values)
        set_1 = np.stack((x_ranges, y_values), axis=-1)

        x_values = (y_ranges - c[i]) / m[i]
        x_values = np.floor(x_values)
        set_2 = np.stack((x_values, y_ranges), axis=-1)

        full_set = np.vstack((set_1, set_2))
        full_set_reshaped = full_set.reshape(full_set.shape[0] * full_set.shape[1], 2).astype(int)
        used_pixels = masked_arena_pixels[full_set_reshaped[:, 0], full_set_reshaped[:, 1]]
        total_sum = used_pixels.sum(axis=0)
        readings[i] = total_sum

    return readings

@cython.nogil
cdef inline void worker(double [:, :, :] masked_arena_pixels,
          double [:] m,
          double [:] c,
          double [:] x_ranges,
          double [:] y_ranges,
          ) nogil:

    cdef double [:] y_values
    cdef double [:, :] set_1

    cdef double [:, :] x_values
    cdef double [:, :, :] set_2

    cdef double [:, :, :] full_set
    cdef long [:, :] full_set_reshaped
    cdef double [:, :] used_pixels
    cdef double [:] total_sum

    cdef int i

    for i in range(len(m)):


    y_values = (m * x_ranges) + c
    y_values = np.floor(y_values)
    set_1 = np.stack((x_ranges, y_values), axis=-1)
    x_values = (y_ranges - c) / m
    x_values = np.floor(x_values)
    set_2 = np.stack((x_values, y_ranges), axis=-1)
    full_set = np.vstack((set_1, set_2))
    full_set_reshaped = full_set.reshape(full_set.shape[0] * full_set.shape[1], 2).astype(int)
    used_pixels = masked_arena_pixels[full_set_reshaped[:, 0], full_set_reshaped[:, 1]]
    total_sum = used_pixels.sum(axis=0)
    return total_sum

@cython.boundscheck(False)
@cython.wraparound(False)
def parallel_lines(cnp.ndarray masked_arena_pixels,
          cnp.ndarray m,
          cnp.ndarray c,
          cnp.ndarray xmin,
          cnp.ndarray xmax,
          cnp.ndarray ymin,
          cnp.ndarray ymax,
          ):
    cdef cnp.ndarray[long, ndim=1] x_len = np.rint(xmax[:, 0] - xmin[:, 0]).astype(int)

    cdef cnp.ndarray[long, ndim=1] y_len = np.rint(ymax[:, 0] - ymin[:, 0]).astype(int)

    cdef long n_photoreceptors = m.shape[0]
    cdef Py_ssize_t i
    cdef double [:, :] readings = np.zeros((120, 3))

    cdef double [:, :, :] x_ranges
    cdef double [:, :, :] y_ranges

    cdef double [:, :] y_values
    cdef double [:, :, :] set_1

    cdef double [:, :] x_values
    cdef double [:, :, :] set_2

    cdef double [:, :, :] full_set
    cdef long [:, :] full_set_reshaped
    cdef double [:, :] used_pixels
    cdef double [:] total_sum


    x_ranges = [np.linspace(xmin[i], xmax[i], d) for i, d in enumerate(x_len)]
    y_ranges = [np.linspace(ymin[i], ymax[i], d) for i, d in enumerate(y_len)]

    for i in prange(n_photoreceptors, nogil=True):
        readings[i] = worker(masked_arena_pixels, m[i], c[i], x_ranges[i], y_ranges[i])

    return readings





# lines = parallel_lines
lines = original_lines