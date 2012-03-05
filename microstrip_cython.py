import pyximport

pyximport.install()

import numpy
from laplace import update
import time
from matplotlib import pyplot as pp
from tables import openFile
from itertools import cycle, chain, repeat


NX=1000
NY=500

dx = 0.005
dy = 0.005

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
mask[die_left:die_right, die_bottom]=1
mask[die_left:die_right, die_top]=1
mask[die_left, die_bottom:die_top]=1
mask[die_right, die_bottom:die_top]=1
mask[:, h] = 1

for E_die in [13.0,1.0]:
    end = time.time() + 3.0
    Qlist = []
    now = int(time.time())
    count = 0
    args = (h, E_subst, strip_left, width, thickness,
               die_left, die_right, die_bottom, die_top,
               E_die, dx, dy)
    this = hash(args)
    
    try:
        h5 = openFile("start_%i.h5"%this)
        U = h5.root.data.read()
        h5.close()
    except:
        U = numpy.zeros((NX,NY,2),'d')
    
    omega = repeat(1.8)
    while count < 20000:
        try:
            Q = update(U, mask, h, E_subst, strip_left, width, thickness,
                   die_left, die_right, die_bottom, die_top,
                   E_die, dx, dy, omega.next())
        except ValueError:
            raise
            U = numpy.zeros((NX,NY,2),'d')
        last = now
        now = time.time()
        if int(now) != int(last):
            Qlist.append(Q)
            print count, "Q=",Q
        count += 1
        
    h5 = openFile("start_%i.h5"%this, 'w')
    a = h5.createArray("/","data",U)
    a.attrs.args = args
    h5.close()
        
    pp.figure()
    pp.imshow(U[:,:,0])
    pp.title("Die E-r = %f"%E_die)
    pp.colorbar()
    pp.figure()
    pp.plot(Qlist, 'ro-')
    pp.title("Die E-r = %f"%E_die)
    pp.figure()
    pp.plot(numpy.diff(U[NX/2,:,1]),'o-')
pp.show()



