import pyopencl as cl
import numpy
import time


class Bencher(object):
    def __init__(self):
        plats = cl.get_platforms()
        for plat in plats:
            print "PLATFORMS:", plat
            devs = plat.get_devices()
            print "...DEVICES:", devs
    
    def gpu_bench(self, N):
        ctx = ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags
        
        U = numpy.zeros((N,N), numpy.float32)
        V = numpy.zeros((N,N), numpy.float32)
        
        U[0,::2] = 1.0
        V[0,1::2] = 1.0
        
        U_buf = cl.Buffer(ctx, mf.READ_WRITE, U.nbytes)
        V_buf = cl.Buffer(ctx, mf.READ_WRITE, V.nbytes)
        
        cl.enqueue_write_buffer(queue, U_buf, U).wait()
        cl.enqueue_write_buffer(queue, V_buf, V).wait()
        
        prg = cl.Program(ctx, """
__kernel void update(__global float *u, __global float *v)
{
    int i = get_global_id(0) + 1;
    int j = get_global_id(1) + 1;
    int ny = %(NY)d;
    
    v[ny*j + i] = ((u[ny*(i-1) + j] + u[ny*(i+1) + j]) +
                                   (u[ny*i + j-1] + u[ny*i + j+1]));
    
    }
    """%{'NY':N, 'NX': N}
        )
        prg.build()
        yield 0
        ct = 0
        while True:
            for i in xrange(10):
                prg.update(queue, ((N-2),(N-2)), None, U_buf, V_buf)
                evt = prg.update(queue, ((N-2),(N-2)), None, V_buf, U_buf)
            cl.enqueue_wait_for_events(queue, [evt])
            ct += 1
            yield ct*i
            
    def run_gpu(self, N):
        gen = self.gpu_bench(N)
        gen.next()
        start = time.time()
        while time.time() < (start+1.0):
            ct = gen.next()
        end = time.time()
        rate = ct/(end-start)
        print N, ":", rate, rate*(N*N)/1e6
        
        
if __name__=="__main__":
    b = Bencher()
    for n in 10**(numpy.linspace(1,3,20)):
        N = int(n)
        b.run_gpu(N)
    
    
    