from __future__ import division

import cython

import numpy
cimport numpy

from libc.math cimport abs


@cython.boundscheck(False)
@cython.wraparound(False)
def absolute(double[:, :] x):
    cdef int i, j, ip
    cdef int ii = 0
    cdef int n = x.shape[0]
    cdef int p = x.shape[1]
    cdef int n2 = n * (n - 1) // 2
    cdef double[:, :] d = numpy.zeros((n2, p), dtype=numpy.double)

    for i in range(n):
        for ip in range(i + 1, n):
            for j in range(p):
                d[ii, j] = abs(x[i, j] - x[ip, j])
            ii += 1

    return numpy.array(d)


@cython.boundscheck(False)
@cython.wraparound(False)
def multiply(double[:, :] x):
    cdef int i, j, ip
    cdef int ii = 0
    cdef int n = x.shape[0]
    cdef int p = x.shape[1]
    cdef int n2 = n * (n - 1) // 2
    cdef double[:, :] d = numpy.zeros((n2, p), dtype=numpy.double)

    for i in range(n):
        for ip in range(i + 1, n):
            for j in range(p):
                d[ii, j] = x[i, j] * x[ip, j]
            ii += 1

    return numpy.array(d)
