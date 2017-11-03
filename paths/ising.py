import numpy as np

from .core import Model, Kernel
from ._paths import ising_energy, ising_sample

class IsingModel(Model):

    def __init__(self, L, beta=1.):

        self.L = int(L)
        self.beta = float(beta)
        
    def energy(self, x):
        return self.beta * ising_energy(self.L, x)

    def sample(self, x=None, n=1, beta=None):

        beta = self.beta if beta is None else float(beta)

        if beta == 0.:
            x = 2 * np.random.randint(0,2,self.L**2,dtype='i') - 1
            return np.ascontiguousarray(x)

        else:
            x = x.copy() if x is not None else self.sample(beta=0.)
            ising_sample(float(beta), int(n), self.L, x)
            return x
        
class IsingKernel(Kernel):

    def __init__(self, L, beta, n=1):

        self._stationary = IsingModel(L, beta)
        self.n_transitions = int(n)

    @property
    def stationary(self):
        return self._stationary

    @property
    def beta(self):
        return self.stationary.beta

    @beta.setter
    def beta(self, value):
        self.stationary.beta = float(value)

    def __call__(self, x):
        return self.stationary.sample(x, self.n_transitions, self.beta)
