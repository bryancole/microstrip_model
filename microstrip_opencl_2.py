import numpy
import time
from matplotlib import pyplot as pp
import pyopencl as cl
from collections import namedtuple

Vec = namedtuple("Vec", "x y")


def make_checkerboard(N):
    i = numpy.arange(N)%2
    j = i.reshape(-1,1)
    brd = (i+j)%2
    #inv = 1-brd
    return brd #, inv

def separate(U):
    """split into checkerboard colours
    U - a 2D array with even length dimensions
    """
    a = U[::2,:]
    b = U[1::2,:]
    c = (numpy.arange(U.shape[1])%2).reshape(1,-1)
    red = numpy.where(c, a,b)
    black = numpy.where(c, b,a)
    return red, black
    
def combine(red, black):
    """combine two sets into a single checkerboard arrangement"""
    c = (numpy.arange(red.shape[1])%2).reshape(1,-1)
    a = numpy.where(c, red, black)
    b = numpy.where(c, black, red)
    ret = numpy.empty((a.shape[0]+b.shape[0],a.shape[1]), a.dtype)
    ret[::2,:] = a
    ret[1::2,:] = b
    return ret
    
    
def gen_result()
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    
    tile = Vec(16,16)
    ntiles = Vec(32,16)
    
    im_size = Vec(tile.x*ntiles.x + 2, tile.y*ntiles.y + 2)
    
    delta = Vec(1.,1.)
    
    prg = cl.Program("""
__kernel void update(__global float *u, __global float *v, __global int mask)
{{
    int I = get_global_id(0);
    int J = get_global_id(1);
    int i = get_local_id(0);
    int j = get_local_id(1);
    int n = get_group_id(0);
    int m = get_group_id(1);
    
    int i = get_global_id(0) + 1;
    int ny = %(NY)d;
    float tmp, newval;
}}
""".format())
    prg.build()
    
    
if __name__=="__main__":
    a = numpy.arange(100).reshape(10,10)    
    b = combine(*separate(a))
    
    assert numpy.allclose(a,b)
    
    a = make_checkerboard(a.shape[0])
    
    b,c = separate(a)
    
    assert numpy.alltrue(b)
    assert numpy.alltrue(numpy.logical_not(c))    
    