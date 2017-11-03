import numpy as np
import paths as pth
import matplotlib.pylab as plt

from paths import take_time
from paths.potts import PottsHistogram
from paths.estimators import jarzynski, cumulant, bar

from csb.io import load
from csb.numeric import log_sum_exp

Q     = 10
L     = 2**8
hist  = PottsHistogram(L)
potts = pth.PottsModel(L, Q)
beta  = 1 / 0.707                     ## critical inverse temperature
beta *= 1.1
x     = potts.sample(beta=0.)

with take_time('calculation of energy'):
    print potts.energy(x)

with take_time('sampling'):
    y = potts.sample(x, 1e7, beta)

print 'energy before: {0}, after: {1}'.format(potts.energy(x), potts.energy(y))

titles = ('before', 'after')

fig, ax = plt.subplots(1,2,figsize=(10,5))
for i, z in enumerate([x, y]):
    ax[i].set_title(titles[i])
    ax[i].matshow(z.reshape(L,L), cmap=plt.cm.jet)
    ax[i].xaxis.set_visible(False)
    ax[i].yaxis.set_visible(False)
    
x = x * 0
print potts.energy(x), hist.E.min()
