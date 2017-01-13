import numpy as np
cimport numpy as np
from libc.math cimport fabs

FLOAT = np.float32
ctypedef np.float32_t FLOAT_t


def distfun(np.ndarray[FLOAT_t, ndim=2] x):
    cdef int i = 0
    cdef int j = 0
    cdef int ip = 0
    cdef int ii = 0
    cdef int n = x.shape[0]
    cdef int p = x.shape[1]
    cdef int n2 = <int>(n * (n - 1) / 2.)
    cdef np.ndarray[FLOAT_t, ndim = 2] d = np.zeros((n2, p), dtype=FLOAT)
    for i in xrange(n):
        for ip in xrange(i + 1, n):
            for j in xrange(p):
                d[ii, j] = fabs(x[i, j] - x[ip, j])
            ii += 1
    return d


def multfun(np.ndarray[FLOAT_t, ndim=2] x):
    cdef int i = 0
    cdef int j = 0
    cdef int ip = 0
    cdef int ii = 0
    cdef int n = x.shape[0]
    cdef int p = x.shape[1]
    cdef int n2 = <int>(n * (n - 1) / 2.)
    cdef np.ndarray[FLOAT_t, ndim = 2] d = np.zeros((n2, p), dtype=FLOAT)
    for i in xrange(n):
        for ip in xrange(i + 1, n):
            for j in xrange(p):
                d[ii, j] = x[i, j] * x[ip, j]
            ii += 1
    return d
