import numpy as np
from math import sqrt
from matplotlib import pyplot
from scipy.stats import binom
from scipy.misc.common import comb
from scipy.stats import norm


def randomWalk1(x, n, p):
    # x is the final position: x = k - (n-k)
    # k steps to right, n-k steps to the left.
    k = float(x+n)/2
    # k has to be a round number. 
    return binom.pmf(k, n, p)

def randomWalk2(x, n, p):
    k = float((x+n)/2)
    # k doesn't have to be round number.
    return comb(n, k)* (p**k) * (1-p)**(n-k) 


### Constants
N = 60
DX = 0.1
xpoints = np.arange(-N, N, DX)

### Random walk.
ypoints = [randomWalk2(x, N, 0.5) for x in xpoints]
# Normalize (Riemann sum).
ypointsSum = sum(ypoints) * DX
ypoints = [y/ypointsSum for y in ypoints]
pyplot.plot(xpoints, ypoints, 'g')

### Compare with a normal distribution.
ypoints = [norm.pdf(x,0,7.7) for x in xpoints]
pyplot.plot(xpoints, ypoints, 'b')

pyplot.show()

