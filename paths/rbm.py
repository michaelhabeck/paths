import numpy as np

from .core import Model, Kernel
from ._paths import rbm_energy, ising_sample

class RBM(Model):

    @property
    def m(self):
        """
        Number of visible states
        """
        return len(self.a)

    @property
    def n(self):
        """
        Number of hidden states
        """
        return len(self.b)

    def __init__(self, a, b, W, beta=1.):

        self.a = np.ascontiguousarray(a.flatten())
        self.b = np.ascontiguousarray(b.flatten())
        self.W = np.ascontiguousarray(W.flatten())

        self.beta = float(beta)
        
    def energy(self, x):
        return self.beta * rbm_energy(x, self.a, self.b, self.W)

    def energy_py(self, x):

        v, h = x[:self.m], x[self.m:]

        return -self.beta * (np.dot(self.a,v) + np.dot(self.b,h) +
                             np.dot(v, np.dot(self.W.reshape(self.m,self.n),h)))

    def sample(self, x=None, n=1, beta=None):

        beta = self.beta if beta is None else float(beta)

        if beta == 0.:
            x = np.random.randint(0, 2, self.m+self.n, dtype='i')
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
