"""
for i in range(1, nx-1):
   2     for j in range(1, ny-1):
   3         u[i,j] = ((u[i-1, j] + u[i+1, j])*dy**2 +
   4                   (u[i, j-1] + u[i, j+1])*dx**2)/(2.0*(dx**2 + dy**2))
"""

cimport numpy as np
from cython.parallel import prange
cimport cython
import numpy

@cython.boundscheck(False)
@cython.cdivision(True)
def update(np.ndarray[double, ndim=2] u, #shape=(N,M,2)
            np.ndarray[int, ndim=2] mask,
           unsigned int h, #height of substrate (in mesh cells)
           double Esub, #permittivity of substrate
           unsigned int strip_left, #position of left hand edge of microstrip
           unsigned int width, #width of microstrip in mesh cells
           unsigned int thickness, #thickness of microstrip in mesh cells
           unsigned int die_left,
           unsigned int die_right,
           unsigned int die_bottom,
           unsigned int die_top,
           double Edie, #permittivity of die
           double dx, double dy, double omega):
    
    cdef:
        unsigned int i, j, N=u.shape[0], M=u.shape[1], ct
        double dx2, dy2, delta, max_delta, update
        np.ndarray[double] adelta=numpy.zeros(u.shape[0]/2, 'd')
    
    dx2 = dx*dx
    dy2 = dy*dy
            
    for ct in xrange(100):
        max_delta = 0.0
        #relax everywhere
        for i in prange(1, N/2, nogil=True):
            adelta[i] = 0
            for j in xrange(1+(i%2), M-1,2):
                if mask[i,j] == 1:
                    continue
                else:
                    update = ((u[i+1,j] + u[i-1,j]) * dy2 +
                              (u[i,j+1] + u[i,j-1]) * dx2) / (2*(dx2+dy2))
                    delta = update - u[i,j]
                    u[i,j] = (1-omega) * u[i,j] + omega * update
                    if delta > adelta[i]:
                        adelta[i] = delta
                        
        for i in prange(1, N/2, nogil=True):
            adelta[i] = 0
            for j in xrange(2-(i%2), M-1,2):
                if mask[i,j] == 1:
                    continue
                else:
                    update = ((u[i+1,j] + u[i-1,j]) * dy2 +
                              (u[i,j+1] + u[i,j-1]) * dx2) / (2*(dx2+dy2))
                    delta = update - u[i,j]
                    u[i,j] = (1-omega) * u[i,j] + omega * update
                    if delta > adelta[i]:
                        adelta[i] = delta
                        
        for i in xrange(1,N/2):
            delta = adelta[i]
            if delta > max_delta:
                max_delta = delta
        
        #apply substrate top boundary condition
        j=h
        for i in xrange(1,strip_left):
            u[i,j] = ((u[i+1,j] + u[i-1,j]) * dy2 +
                        2*( u[i,j+1] + Esub*u[i,j-1] ) * dx2/(Esub+1) ) / (2*(dx2+dy2))
        for i in xrange(strip_left+width, N-1):
            u[i,j] = ((u[i+1,j] + u[i-1,j]) * dy2 +
                        2*( u[i,j+1] + Esub*u[i,j-1] ) * dx2/(Esub+1) ) / (2*(dx2+dy2))
        
        #apply die boundary conditions on edges
        j = die_bottom
        for i in xrange(die_left, die_right):
            u[i,j] = ((u[i+1,j] + u[i-1,j]) * dy2 +
                        2*(Edie*u[i,j+1] + u[i,j-1] ) * dx2/(Edie+1) ) / (2*(dx2+dy2))
        j = die_top
        for i in xrange(die_left, die_right):
            u[i,j] = ((u[i+1,j] + u[i-1,j]) * dy2 +
                        2*( u[i,j+1] + Edie*u[i,j-1] ) * dx2/(Edie+1) ) / (2*(dx2+dy2))
        i = die_left
        for j in xrange(die_bottom, die_top):
            u[i,j] = (2*(Edie*u[i+1,j] + u[i-1,j]) * dy2/(Edie+1) +
                        (u[i,j+1] + u[i,j-1] ) * dx2 ) / (2*(dx2+dy2))
        i = die_right
        for j in xrange(die_bottom, die_top):
            u[i,j] = (2*(u[i+1,j] + Edie*u[i-1,j]) * dy2/(Edie+1) +
                        (u[i,j+1] + u[i,j-1] ) * dx2 ) / (2*(dx2+dy2))
        #apply die corner boundary conditions
        i = die_left
        j = die_bottom
        u[i,j] = ((0.5*(Edie+1)*u[i+1,j] + u[i-1,j]) * dy2/(0.5*(Edie+3)) +
                    (0.5*(Edie+1)*u[i,j+1] + u[i,j-1] ) * dx2/(0.5*(Edie+3)) ) / (dx2+dy2)
        i = die_left
        j = die_top
        u[i,j] = ((0.5*(Edie+1)*u[i+1,j] + u[i-1,j]) * dy2/(0.5*(Edie+3)) +
                    (u[i,j+1] + 0.5*(Edie+1)*u[i,j-1] ) * dx2/(0.5*(Edie+3)) ) / (dx2+dy2)
        i = die_right
        j = die_bottom
        u[i,j] = ((u[i+1,j] + 0.5*(Edie+1)*u[i-1,j]) * dy2/(0.5*(Edie+3)) +
                    (0.5*(Edie+1)*u[i,j+1] + u[i,j-1] ) * dx2/(0.5*(Edie+3)) ) / (dx2+dy2)
        i = die_right
        j = die_top
        u[i,j] = ((u[i+1,j] + 0.5*(Edie+1)*u[i-1,j]) * dy2/(0.5*(Edie+3)) +
                    (u[i,j+1] + 0.5*(Edie+1)*u[i,j-1] ) * dx2/(0.5*(Edie+3)) ) / (dx2+dy2)
                                
        for j in xrange(1, M-1):
            u[N/2,j] = u[(N/2)-1,j]
    
    return max_delta *0.25*2*N*N*M*M*(dx2+dy2) / (3.141592654*(M*M*dx2 + N*N*dy2))