import pyopencl as cl
import numpy
import time
import pyximport

pyximport.install()

from cython_bench import update


class Bencher(object):
    def __init__(self, dev_type='gpu'):
        self.type = dev_type
        plats = cl.get_platforms()
        for plat in plats:
            print "PLATFORMS:", plat
            devs = plat.get_devices()
            print "...DEVICES:", devs
            self.devs = devs
            
    def make_gen(self, N):
        type = self.type
        if type == 'omp':
            gen = self.omp_bench(N)
        elif type == 'cpu':
            self.dev = self.devs[1]
            #print "USING:", self.dev
            gen = self.gpu_bench(N)
        elif type == 'gpu':
            self.dev = self.devs[0]
            #print "USING:", self.dev
            gen = self.gpu_bench(N)
        else:
            raise ValueError("Unknown device type")
        return gen
    
    def gpu_bench(self, N):
        ctx = cl.Context(devices=[self.dev])
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
                                   (u[ny*i + j-1] + u[ny*i + j+1]))*0.25;
    
    }
    """%{'NY':N, 'NX': N}
        )
        prg.build()
        yield 0
        reps = 100 if N<=1000 else 10 
        ct = 0
        while True:
            for i in xrange(reps):
                evt = prg.update(queue, ((N-2),(N-2)), None, U_buf, V_buf)
                evt = prg.update(queue, ((N-2),(N-2)), None, V_buf, U_buf, wait_for=[evt])
                ct += 1
            queue.finish()
            yield ct
            
    def omp_bench(self, N):
        U = numpy.zeros((N,N), numpy.float32)
        V = numpy.zeros((N,N), numpy.float32)
        
        yield 0
        reps = 100 if N<=1000 else 10 
        ct = 0
        while True:
            for i in xrange(reps):
                update(U,V)
                update(V,U)
                ct += 1
            yield ct
            
    def run_gpu(self, N):
        gen = self.gpu_bench(N)
        gen.next()
        start = time.time()
        while time.time() < (start+1.0):
            ct = gen.next()
        end = time.time()
        rate = ct/(end-start)
        return rate, rate*(N*N)/1e6
    
    def run(self, N):
        gen = self.make_gen(N)
        gen.next()
        start = time.time()
        while time.time() < (start+1.0):
            ct = gen.next()
        end = time.time()
        rate = ct/(end-start)
        return rate, rate*(N*N)/1e6
        
        
if __name__=="__main__":
    results={}
    for dev_id in ("omp", "gpu", "cpu"):
        b = Bencher(dev_id)
        out = []
        done = set()
        for n in 10**(numpy.linspace(1,3.3,100)):
            N = int(n)
            if N in done:
                continue
            else:
                done.add(N)
            rate, ops = b.run(N)
            print N, ":", rate, ops
            out.append((N,rate,ops))
        results[dev_id] = out
    from matplotlib import pyplot as pp
    for name in results:
        out = results[name]
        pp.loglog([a[0]**2 for a in out], [a[2] for a in out], 'o-', label=name)
    pp.legend(loc='best')
    pp.xlabel("Problem size, N")
    pp.ylabel("Efficiency / cycles per cell")
    pp.title("OpenCL vs OpemMP benchmark")
    pp.grid()
    pp.show()
    
    
    