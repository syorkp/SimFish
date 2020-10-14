#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np

cimport numpy as cnp


cdef _ray(Py_ssize_t r0, Py_ssize_t c0, Py_ssize_t r1, Py_ssize_t c1, double[:, :, :] im, long nrows, long ncols):
    """Generate line pixel coordinates.
    Parameters
    ----------
    r0, c0 : int
        Starting position (row, column).
    r1, c1 : int
        End position (row, column).
    Returns
    -------
    rr, cc : (N,) ndarray of int
        Indices of pixels that belong to the line.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.
    See Also
    --------
    line_aa : Anti-aliased line generator
    """

    cdef char steep = 0
    cdef Py_ssize_t r = r0
    cdef Py_ssize_t c = c0
    cdef Py_ssize_t dr = abs(r1 - r0)
    cdef Py_ssize_t dc = abs(c1 - c0)
    cdef Py_ssize_t sr, sc, d, i
    #cdef Py_ssize_t[::1] pv = np.zeros(3, dtype=np.float16)
    #cdef Py_ssize_t[::1] rr = np.zeros(max(dc, dr) + 1, dtype=np.intp)
    #cdef Py_ssize_t[::1] cc = np.zeros(max(dc, dr) + 1, dtype=np.intp)
    cdef Py_ssize_t rr, cc
    cdef double cc1=0, cc2=0, cc3=0, tp=0
    cdef cnp.ndarray res = np.zeros(6)
    
    with nogil:
        if (c1 - c) > 0:
            sc = 1
        else:
            sc = -1
        if (r1 - r) > 0:
            sr = 1
        else:
            sr = -1
        if dr > dc:
            steep = 1
            c, r = r, c
            dc, dr = dr, dc
            sc, sr = sr, sc
        d = (2 * dr) - dc
    
        for i in range(dc):
            if steep:
                rr = c
                cc = r
            else:
                rr = r
                cc = c
            while d >= 0:
                r = r + sr
                d = d - (2 * dc)
            c = c + sc
            d = d + (2 * dr)
            if rr>(nrows-1) or cc>(ncols-1) or rr<0 or cc<0: # hitting a wall
                cc1 = 1
                cc2 = 0
                cc3 = 0
                tp = 1
                break
            if (im[rr, cc, 0] + im[rr, cc, 1] + im[rr, cc, 2]) > 0: # hitting an object
                cc1 = im[rr, cc, 0]
                cc2 = im[rr, cc, 1]
                cc3 = im[rr, cc, 2]
                tp = 2
                break
#        if dark_col > 0 and cc < dark_col:
#            cc1 *= 0.01
#            cc2 *= 0.01
#            cc3 *= 0.01

    res[0] = cc1  # three color channels, type of collision, collision coordinates  
    res[1] = cc2
    res[2] = cc3
    res[3] = tp
    res[4] = rr
    res[5] = cc
        
    return res
    #     rr[dc] = r1
    #     cc[dc] = c1

    # prfl = im[rr, cc]
    # ps = np.sum(prfl, 1)
    # if len(np.nonzero(ps)[0]) > 0:
    #     return prfl[np.nonzero(ps)[0][0], :]
    # else:
    #     return np.array([0,0,0], dtype=np.float)

def rays(long [:, :] xmat, long [:, :] ymat, double [:, :, :] im, long nrows, long ncols, double dark_gain, double light_gain, double bkg_scatter, long dark_col=0):
    cdef Py_ssize_t i, c
    cdef Py_ssize_t num_rays = xmat.shape[0]
    
    res = np.zeros((num_rays, 3))
    cdef double[:, :] result_view = res
    r = np.zeros(3)
    cdef double[:] r_view = r
    
    for i in range(num_rays):
        r_view = _ray(ymat[i,0], xmat[i,0], ymat[i,1], xmat[i,1], im, nrows, ncols)
        
        if r_view[3] == 0:  # empty space
            if r_view[5] < dark_col:
                for c in range(3):
                    result_view[i, c] = bkg_scatter * dark_gain
            else:
                for c in range(3):
                    result_view[i, c] = bkg_scatter * light_gain
        else:                # wall or object
            if r_view[5] < dark_col:
                for c in range(3):
                    result_view[i, c] = r_view[c] * dark_gain
            else:
                for c in range(3):
                    result_view[i, c] = r_view[c] * light_gain
            
    return res
