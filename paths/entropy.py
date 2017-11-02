import numpy as np

from csb.numeric import log_sum_exp

class Entropy(object):
    """
    Microcanonical entropy or log density of states
    """
    def __init__(self, E, s, normed=True):

        self.E = E
        self.s = s
        
        if normed: self.normalize()
        
    def normalize(self):
        self.s -= log_sum_exp(self.s)

    def log_Z(self, beta):
        return log_sum_exp(-beta * self.E + self.s)

class IsingEntropy(Entropy):

    def __init__(self, L, normed=True):

        if not L in (4,5,8,16,32,64,128):
            msg = 'No precomputed microcanonical entropy for L={0} available'
            raise ValueError(msg.format(L))

        self.L = int(L)
        s = np.load('./data/ising-entropies.npz')[str(self.L)]
        E = np.arange(-2*self.L**2, +2*self.L**2+1, 4)
        
        super(IsingEntropy, self).__init__(E, s, normed)
        

for L in (4,5,8,16,32,64,128):

    entropy = IsingEntropy(L)
    print L, entropy.log_Z(1.), entropy.log_Z(1.)/L**2
