from __future__ import division

import six
import cython
import numpy as np
cimport numpy as np
from libc.math cimport abs

DOUBLE = np.float64
ctypedef np.float64_t DOUBLE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def distfun(np.ndarray[DOUBLE_t, ndim=2] x):
    cdef int i = 0
    cdef int j = 0
    cdef int ip = 0
    cdef int ii = 0
    cdef int n = x.shape[0]
    cdef int p = x.shape[1]
    cdef int n2 = n * (n - 1) // 2
    cdef np.ndarray[DOUBLE_t, ndim = 2] d = np.zeros((n2, p), dtype=DOUBLE)
    for i in six.moves.range(n):
        for ip in six.moves.range(i + 1, n):
            for j in six.moves.range(p):
                d[ii, j] = abs(x[i, j] - x[ip, j])
            ii += 1
    return d

@cython.boundscheck(False)
@cython.wraparound(False)
def multfun(np.ndarray[DOUBLE_t, ndim=2] x):
    cdef int i = 0
    cdef int j = 0
    cdef int ip = 0
    cdef int ii = 0
    cdef int n = x.shape[0]
    cdef int p = x.shape[1]
    cdef int n2 = n * (n - 1) // 2
    cdef np.ndarray[DOUBLE_t, ndim = 2] d = np.zeros((n2, p), dtype=DOUBLE)
    for i in six.moves.range(n):
        for ip in six.moves.range(i + 1, n):
            for j in six.moves.range(p):
                d[ii, j] = x[i, j] * x[ip, j]
            ii += 1
    return d
