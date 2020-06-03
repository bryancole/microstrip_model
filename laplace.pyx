"""
for i in range(1, nx-1):
   2     for j in range(1, ny-1):
   3         u[i,j] = ((u[i-1, j] + u[i+1, j])*dy**2 +
   4                   (u[i, j-1] + u[i, j+1])*dx**2)/(2.0*(dx**2 + dy**2))
"""

cimport numpy as np
from cython.parallel import prange
cimport cython
from libc.math cimport fmax
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
    
    dx2 = dx*dx
    dy2 = dy*dy 
            
    for ct in xrange(100):
        max_delta = 0.0
        #relax everywhere
        for i in prange(1, N/2, nogil=True):
            for j in xrange(1+(i%2), M-1,2):
                if mask[i,j] == 1:
                    continue
                else:
                    update = ((u[i+1,j] + u[i-1,j]) * dy2 +
                              (u[i,j+1] + u[i,j-1]) * dx2) / (2*(dx2+dy2))
                    delta = update - u[i,j]
                    u[i,j] = (1-omega) * u[i,j] + omega * update
                    max_delta = fmax(max_delta, delta)
                        
        for i in prange(1, N/2, nogil=True):
            for j in xrange(2-(i%2), M-1,2):
                if mask[i,j] == 1:
                    continue
                else:
                    update = ((u[i+1,j] + u[i-1,j]) * dy2 +
                              (u[i,j+1] + u[i,j-1]) * dx2) / (2*(dx2+dy2))
                    delta = update - u[i,j]
                    u[i,j] = (1-omega) * u[i,j] + omega * update
                    max_delta = fmax(max_delta, delta)
        
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

@cython.boundscheck(False)
@cython.cdivision(True)
def semi_update(np.ndarray[double, ndim=2] r, #IN
                np.ndarray[double, ndim=2] b, #OUT
                np.ndarray[int, ndim=2] op,
                double dx2, double dy2):
    cdef:
        unsigned int i, j, J, N=r.shape[0], M=r.shape[1]
        double *epsilon = [1.0, 2.2, 13.0] #dielectric constants
        int OP
        double E1,E2, top, bottom, left, right, newval, omega=1.9
        
    """Checker board half-update scheme
    op - >=0 = FD update, epsilon indexed from *epsilon
         -1 = constant potential
         -2 = horizontal dielectric boundary
         -3 = vertical dielectric boundary
    """
    for J in prange(1, M/2, nogil=True):
        for i in xrange(1, N):
            j = 2*J-1
            OP = op[i-1,j+1]
            top = r[i-1,j+2]
            bottom = r[i-1,j]
            left = r[i-1,j+1]
            right = r[i,j+1]
            if OP>=0:
                newval = 0.5*((bottom + top)*dx2+\
                                  (left + right)*dy2)/(dx2+dy2)
                #delta = newval - b[i-1,j+1]
                b[i-1,j+1] = (1-omega)*b[i-1,j+1] + omega*newval
#            elif OP==-2:
#                E1 = epsilon[op[i-1,j]] #bottom
#                E2 = epsilon[op[i-1,j+2]] #top
#                b[i-1,j+1] = 0.5*( (left + right)*dy2 +\
#                                   2*(E1*bottom + E2*top)*dx2/(E1+E2)
#                                   ) / (dx2 + dy2)
#            elif OP==-3:
#                E1 = epsilon[op[i-1,j+1]] #left
#                E2 = epsilon[op[i,j+1]] #right
#                b[i-1,j+1] = 0.5*( (bottom + top)*dx2 +\
#                                   2*(E1*left + E2*right)*dy2/(E1+E2)
#                                  ) / (dx2 + dy2)
                
            OP = op[i,j]
            top = r[i,j+1]
            bottom = r[i,j-1]
            left = r[i-1,j]
            right = r[i,j]
            if OP>=0:
                newval = 0.5*((bottom + top)*dx2+\
                              (left + right)*dy2)/(dx2+dy2)
                b[i,j] = (1-omega)*b[i,j] + omega*newval
#            elif OP==-2:
#                E1 = epsilon[op[i,j-1]] #bottom
#                E2 = epsilon[op[i,j+1]] #top
#                b[i-1,j+1] = 0.5*( (left + right)*dy2 +\
#                                   2*(E1*bottom + E2*top)*dx2/(E1+E2)
#                                   ) / (dx2 + dy2)
#            elif OP==-3:
#                E1 = epsilon[op[i-1,j]] #left
#                E2 = epsilon[op[i,j]] #right
#                b[i-1,j+1] = 0.5*( (bottom + top)*dx2 +\
#                                   2*(E1*left + E2*right)*dy2/(E1+E2)
#                                  ) / (dx2 + dy2)
                                  