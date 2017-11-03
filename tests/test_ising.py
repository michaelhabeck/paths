import numpy as np
import paths as pth
import matplotlib.pylab as plt

from paths import take_time
from paths.estimators import jarzynski, cumulant, bar

from csb.io import load
from csb.numeric import log_sum_exp

L = 2**8
ising = pth.IsingModel(L)
x = ising.sample(None, beta=0.)

with take_time('energy'):
    ising.energy(x)

## critical inverse temperature

beta = 0.5 * np.log(1 + 2**0.5)

with take_time('sampling'):
    y = ising.sample(x, 1e6, beta)

print 'energy before: {0}, after: {1}'.format(ising.energy(x), ising.energy(y))

fig, ax = plt.subplots(1,2,figsize=(10,5))
for i, z in enumerate([x, y]):
    ax[i].matshow(z.reshape(L,L), cmap=plt.cm.gray)
    ax[i].xaxis.set_visible(False)
    ax[i].yaxis.set_visible(False)
    
