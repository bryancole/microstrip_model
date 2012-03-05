"""
for i in range(1, nx-1):
   2     for j in range(1, ny-1):
   3         u[i,j] = ((u[i-1, j] + u[i+1, j])*dy**2 +
   4                   (u[i, j-1] + u[i, j+1])*dx**2)/(2.0*(dx**2 + dy**2))
"""

cimport numpy as np
cimport cython

@cython.boundscheck(False)
def update(np.ndarray[double, ndim=3] u, #shape=(N,M,2)
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
        unsigned int i, j, a, b
        double dx2, dy2, Q=0.0, Er
    
    dx2 = dx*dx
    dy2 = dy*dy
    
    #set equipotential in layer 1
    for i in xrange(strip_left, strip_left+thickness):
        for j in xrange(h, h+thickness):
            u[i,j,0] = 1.0 #potential is always 1
            u[i,j,1] = 1.0 #potential is always 1
            
    for a in xrange(2):
        b = 1 - a
        #relax everywhere
        for i in xrange(1,u.shape[0]-1):
            for j in xrange(1, u.shape[1]-1):
                if mask[i,j] == 1:
                    continue
                else:
                    u[i,j,b] = (1-omega) * u[i,j,a] + \
                               omega * ((u[i+1,j,a] + u[i-1,j,b]) * dy2 +
                              (u[i,j+1,a] + u[i,j-1,b]) * dx2) / (2*(dx2+dy2))

                    #~ u[i,j,b] = ((u[i+1,j,a] + u[i-1,j,b]) * dy2 +
                             #~ (u[i,j+1,a] + u[i,j-1,b]) * dx2) / (2*(dx2+dy2))
        
        #~ #reset equipotential
        #~ for i in xrange(strip_left, strip_left+width):
            #~ for j in xrange(h, h+thickness):
                #~ u[i,j,b] = 1.0 #potential is always 1
        
 
       #apply substrate top boundary condition
        j=h
        for i in xrange(1,strip_left):
            u[i,j,b] = ((u[i+1,j,a] + u[i-1,j,a]) * dy2 +
                        2*( u[i,j+1,a] + Esub*u[i,j-1,a] ) * dx2/(Esub+1) ) / (2*(dx2+dy2))
        for i in xrange(strip_left+width, u.shape[0]-1):
            u[i,j,b] = ((u[i+1,j,a] + u[i-1,j,a]) * dy2 +
                        2*( u[i,j+1,a] + Esub*u[i,j-1,a] ) * dx2/(Esub+1) ) / (2*(dx2+dy2))
        
        #apply die boundary conditions on edges
        j = die_bottom
        for i in xrange(die_left, die_right):
            u[i,j,b] = ((u[i+1,j,a] + u[i-1,j,a]) * dy2 +
                        2*(Edie*u[i,j+1,a] + u[i,j-1,a] ) * dx2/(Edie+1) ) / (2*(dx2+dy2))
        j = die_top
        for i in xrange(die_left, die_right):
            u[i,j,b] = ((u[i+1,j,a] + u[i-1,j,a]) * dy2 +
                        2*( u[i,j+1,a] + Edie*u[i,j-1,a] ) * dx2/(Edie+1) ) / (2*(dx2+dy2))
        i = die_left
        for j in xrange(die_bottom, die_top):
            u[i,j,b] = (2*(Edie*u[i+1,j,a] + u[i-1,j,a]) * dy2/(Edie+1) +
                        (u[i,j+1,a] + u[i,j-1,a] ) * dx2 ) / (2*(dx2+dy2))
        i = die_right
        for j in xrange(die_bottom, die_top):
            u[i,j,b] = (2*(u[i+1,j,a] + Edie*u[i-1,j,a]) * dy2/(Edie+1) +
                        (u[i,j+1,a] + u[i,j-1,a] ) * dx2 ) / (2*(dx2+dy2))
        #apply die corner boundary conditions
        i = die_left
        j = die_bottom
        u[i,j,b] = ((0.5*(Edie+1)*u[i+1,j,a] + u[i-1,j,a]) * dy2/(0.5*(Edie+3)) +
                    (0.5*(Edie+1)*u[i,j+1,a] + u[i,j-1,a] ) * dx2/(0.5*(Edie+3)) ) / (dx2+dy2)
        i = die_left
        j = die_top
        u[i,j,b] = ((0.5*(Edie+1)*u[i+1,j,a] + u[i-1,j,a]) * dy2/(0.5*(Edie+3)) +
                    (u[i,j+1,a] + 0.5*(Edie+1)*u[i,j-1,a] ) * dx2/(0.5*(Edie+3)) ) / (dx2+dy2)
        i = die_right
        j = die_bottom
        u[i,j,b] = ((u[i+1,j,a] + 0.5*(Edie+1)*u[i-1,j,a]) * dy2/(0.5*(Edie+3)) +
                    (0.5*(Edie+1)*u[i,j+1,a] + u[i,j-1,a] ) * dx2/(0.5*(Edie+3)) ) / (dx2+dy2)
        i = die_right
        j = die_top
        u[i,j,b] = ((u[i+1,j,a] + 0.5*(Edie+1)*u[i-1,j,a]) * dy2/(0.5*(Edie+3)) +
                    (u[i,j+1,a] + 0.5*(Edie+1)*u[i,j-1,a] ) * dx2/(0.5*(Edie+3)) ) / (dx2+dy2)
                    
    #Evaluate charge on conductor
    for i in xrange(strip_left-2,strip_left+width+2):
        j = h-2
        Q += Esub * (u[i,j-1,b] - u[i,j,b]) / dy
        j = h+thickness+2
        Q += (u[i,j,b] - u[i,j-1,b]) / dy
    for j in xrange(h-2, h+thickness+2):
        if j<h:
            Er = Esub
        elif j>h:
            Er=1.0
        else:
            Er = 0.5 + 0.5*Er
        i = strip_left-2
        Q += Er*(u[i-1,j,b] - u[i,j,b]) / dx
        i = strip_left + width + 2
        Q += Er*(u[i,j,b] - u[i-1,j,b]) / dx
            
    return Q