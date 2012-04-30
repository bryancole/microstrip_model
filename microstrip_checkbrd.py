import pyximport

pyximport.install()

import numpy
from laplace import semi_update
from checkerbrd import separate, combine
import time
from matplotlib import pyplot as pp
from tables import openFile
from itertools import cycle, chain, repeat


NX=1000
NY=500

dx = 0.005
dy = 0.005

dx2 = dx*dx
dy2 = dy*dy

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

U = numpy.zeros((NX,NY),'d')
OP = numpy.zeros((NX,NY), numpy.int)

U[strip_left:strip_left+width, h:h+thickness] = 1.0
OP[strip_left:strip_left+width, h:h+thickness] = -1
OP[0,:] = -1
OP[-1,:] = -1
OP[:,0] = -1
OP[:,-1] = -1

OP[:strip_left, h] = -2 #horzontal boundary
OP[strip_left+width:, h] = -2

OP[die_left:die_right, die_bottom]=-2
OP[die_left:die_right, die_top]=-2
OP[die_left, die_bottom:die_top]=-3
OP[die_right, die_bottom:die_top]=-3

U1, U2 = separate(U)
OP1, OP2 = separate(OP)

for i in xrange(1000):
    semi_update(U1,U2,OP1,dx2,dy2)
    semi_update(U2,U1,OP2,dx2,dy2)
    if i%100 == 0:
        print i
    
U_out = combine(U1,U2)
pp.figure()
pp.imshow(U_out)
pp.figure()
pp.imshow(U)
pp.show()
