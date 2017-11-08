import os
import numpy as np
import paths as pth

params = np.load('./data/mnistvh_CD25.npz')
               
rbm = pth.RBM(params['a'], params['b'], params['W'])

x = rbm.sample(beta=0.)

print rbm.energy(x), rbm.energy_py(x)
