cimport numpy as np
from cython.parallel import prange
cimport cython
import numpy

@cython.boundscheck(False)
@cython.cdivision(True)
def update(np.ndarray[float, ndim=2] u, np.ndarray[float, ndim=2] v):
    cdef:
        unsigned int i, j, N = u.shape[0], M = u.shape[1]
    
    for i in prange(1, N-2, nogil=True):
        for j in xrange(1, M-2):
            v[i,j] = (u[i+1,j] + u[i-1,j] + u[i,j-1] + u[i,j+1])*0.25
            
        
    