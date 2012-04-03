import numpy
import time
from matplotlib import pyplot as pp
from tables import openFile
from itertools import cycle, chain, repeat

import pyopencl as cl

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

NX=1000
NY=500

dx = 0.005
dy = 0.005

dx2 = dx*dx
dy2 = dy*dy
dnr_inv = 0.5/(dx2+dy2)

X = dx*(NX-1)
Y = dy*(NY-1)

E_subst = 2.2
E_die = 13.0

h = 26 #130 microns high
width = 72 #width of 360 microns
thickness = 3 #15 micron thick
strip_left = int((NX-width)/2.)

die_bottom = h + thickness + 6
die_top = die_bottom + 100
die_left = int(NX/2) - 300
die_right = die_left + 600 # 3mm wide die block

mask = numpy.zeros((NX,NY), numpy.int32)
mask[strip_left:strip_left+width,h:h+thickness] = 1
#mask[die_left:die_right, die_bottom]=1
#mask[die_left:die_right, die_top]=1
#mask[die_left, die_bottom:die_top]=1
#mask[die_right, die_bottom:die_top]=1
#mask[:, h] = 1

err = numpy.zeros((NY-2,), numpy.float32)

def make_mesh():
    U = numpy.zeros((NX,NY),numpy.float32)
    U[strip_left:strip_left+width, h:h+thickness] = 1.0
    return U

for E_die in [1.0]:
    start = time.time()
    end = start + 3.0
    Qlist = []
    now = int(time.time())
    count = 0
    args = (h, E_subst, strip_left, width, thickness,
               die_left, die_right, die_bottom, die_top,
               E_die, dx, dy)
    this = hash(args)
    
    try:
        raise Exception("don't read in data")
        h5 = openFile("start_%i.h5"%this)
        U = h5.root.data.read()
        h5.close()
    except:
        U = make_mesh()
        
    
    U_buf = cl.Buffer(ctx, mf.READ_WRITE, U.nbytes)
    err_buf = cl.Buffer(ctx, mf.READ_WRITE, err.nbytes)
    mask_buf = cl.Buffer(ctx, mf.READ_ONLY, mask.nbytes)
    
    cl.enqueue_write_buffer(queue, U_buf, U).wait()
    cl.enqueue_write_buffer(queue, mask_buf, mask).wait()
    cl.enqueue_write_buffer(queue, err_buf, err).wait()
    
    omega = 1.9
    
    prg = cl.Program(ctx, """
__kernel void update(__global float *u, __global uint *mask, __global float *err, const uint stidx)
{
    int i = get_global_id(0) + 1;
    int ny = %(NX)d;
    float tmp, newval;
    
    if ( stidx == 1 )
        err[i-1] = 0.0;
    
    for ( int j = 1 + ( ( i + stidx ) %% 2 ); j<( %(NY)d-2 ); j+=2 ) {
        if ( mask[ny*j + i] == 0 ) {
          tmp = u[ny*j + i];
          newval = ((u[ny*(j-1) + i] + u[ny*(j+1) + i])*%(dy2)g +
                               (u[ny*j + i-1] + u[ny*j + i+1])*%(dx2)g)*%(dnr_inv)g;
          err[i-1] = fmax( fabs(newval-tmp), err[i-1] );
          u[ny*j+i] = (1.0 - %(omega)g)*tmp + %(omega)g*newval;
          }
        }
    }
    """%{'NY':NY, 'NX': NX, 'dy2': dy2, 'dx2': dx2,'dnr_inv':dnr_inv, 'omega':omega}
    )
    prg.build()
    
    factor = 0.25*2*NX*NX*NY*NY*(dx2+dy2) / (3.141592654*(NY*NY*dx2 + NX*NX*dy2))
    
    def full_time_step():
        for i in xrange(100):
            evt1 = prg.update(queue, ((NX-2),), None,
                              U_buf, mask_buf, err_buf, numpy.uint32(1))
            cl.enqueue_wait_for_events(queue, [evt1])
            evt2 = prg.update(queue, ((NX-2),), None,
                              U_buf, mask_buf, err_buf, numpy.uint32(2))
            cl.enqueue_wait_for_events(queue, [evt2])
        
            ###need to apply boundary conditions
        
        cl.enqueue_read_buffer(queue, err_buf, err).wait()
        return err.max() * factor
    
    
    start = time.time()
    while True:
        Q = full_time_step()
        last = now
        now = time.time()
        if int(now) != int(last):
            Qlist.append(Q)
            print count, "max error=",Q
            if Q <= 0.001:
                break
        count += 1
    print ">>>Completed after", time.time()-start, "seconds"
        
    cl.enqueue_read_buffer(queue, U_buf, U).wait()
        
#    h5 = openFile("start_%i.h5"%this, 'w')
#    a = h5.createArray("/","data",U)
#    a.attrs.args = args
#    h5.close()
    
    upper = h + thickness + 3
    cap = (E_subst * (U[:,int(h/2)+1] - U[:,int(h/2)])/dy
            + (U[:,upper] - U[:,upper+1])/dy).sum()
    print "Total charge = ", cap
        
    pp.figure()
    pp.imshow(U)
    pp.title("Die E-r = %f"%E_die)
    pp.xlabel("X position")
    pp.ylabel("Y position")
    pp.colorbar()
    
    pp.figure()
    pp.plot(Qlist, 'ro-')
    pp.title("Die E-r = %f"%E_die)
    pp.xlabel("Python func calls")
    pp.ylabel("Maximum error")
    
    pp.figure()
    pp.plot(numpy.diff(U[NX/2,:]),'o-')
    pp.ylabel("Electric field")
    pp.xlabel("Position")
pp.show()
