import numpy as np

from .ising import IsingModel, IsingKernel
from ._paths import potts_energy, potts_sample

class PottsModel(IsingModel):

    def __init__(self, L, Q, beta=1.):

        super(PottsModel, self).__init__(L, beta)

        self.Q = int(Q)

    def energy(self, x):
        return self.beta * potts_energy(self.L, x)

    def sample(self, x=None, n=1, beta=None):

        beta = self.beta if beta is None else float(beta)

        if beta == 0.:
            x = np.random.randint(0,self.Q,self.L**2,dtype='i')
            return np.ascontiguousarray(x)

        else:
            x = x.copy() if x is not None else self.sample(beta=0.)
            m = potts_sample(float(beta), int(n), self.L, self.Q, x)
            return x
        
class PottsKernel(IsingKernel):

    def __init__(self, L, Q, beta, n=1):

        super(PottsKernel, self).__init__(L, beta, n)

        self._stationary = PottsModel(L, Q, beta)

