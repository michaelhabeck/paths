import numpy as np
import paths as pth
import matplotlib.pylab as plt

from paths import take_time
from paths.estimators import jarzynski, cumulant, bar

from csb.io import load
from csb.numeric import log_sum_exp

L     = 2**8
ising = pth.IsingModel(L)
beta  = 0.5 * np.log(1 + 2**0.5)      ## critical inverse temperature
x     = ising.sample(None, beta=0.)

with take_time('calculation of energy'):
    ising.energy(x)

with take_time('sampling'):
    y = ising.sample(x, 1e6, beta)

print 'energy before: {0}, after: {1}'.format(ising.energy(x), ising.energy(y))

titles = ('before', 'after')

fig, ax = plt.subplots(1,2,figsize=(10,5))
for i, z in enumerate([x, y]):
    ax[i].set_title(titles[i])
    ax[i].matshow(z.reshape(L,L), cmap=plt.cm.gray)
    ax[i].xaxis.set_visible(False)
    ax[i].yaxis.set_visible(False)
    
