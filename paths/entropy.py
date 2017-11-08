import os
import numpy as np

from csb.numeric import log_sum_exp

from .potts import PottsHistogram

class Entropy(object):
    """Entropy
    
    Microcanonical entropy or log density of states
    """
    def __init__(self, E, s, normed=True):
        """
        Parameters:
        ----------
        E : numpy array
          energies at which the microcanonical entropy is known

        s : numpy array
          microcanonical entropies s(E) = log g(E)
        """
        self.E = E
        self.s = s
        
        if normed: self.normalize()
        
    def normalize(self):
        self.s -= log_sum_exp(self.s)

    def log_Z(self, beta):
        """
        Log partition function
        """
        return log_sum_exp(-beta * self.E + self.s)

    def E_mean(self, beta):
        """
        Average energy 
        """
        p  = -beta * self.E + self.s
        p -= log_sum_exp(p)
        p  = np.exp(p)
        
        return np.dot(p, self.E)
    
class IsingEntropy(Entropy):
    """IsingEntropy

    Exact microcanonical entropy of the Ising model due to Beale,
    Phys Rev Lett (1996)
    """
    def __init__(self, L, normed=True):

        if not L in (4,5,8,16,32,64,128):
            msg = 'No precomputed microcanonical entropy for L={0} available'
            raise ValueError(msg.format(L))

        from . import __path__

        path = __path__[0]

        self.L = int(L)

        s = np.load(os.path.join(path, 'data', 'ising-entropies.npz'))[str(self.L)]
        E = np.arange(-2*self.L**2, +2*self.L**2+1, 4)

        i = np.array([1, len(s)-2])
        s = np.delete(s, i)
        E = np.delete(E, i)
        
        super(IsingEntropy, self).__init__(E, s, normed)
        
class PottsEntropy(Entropy):
    """PottsEntropy

    Microcanonical entropies of the ten-state Potts model estimated with
    the Wang-Landau algorithm
    """
    def __init__(self, L, normed=True):

        if not L in (8,16,32):
            msg = 'No precomputed microcanonical entropy for L={0} available'
            raise ValueError(msg.format(L))

        from . import __path__

        path = __path__[0]

        self.L = int(L)

        s = np.load(os.path.join(path, 'data', 'potts-entropies.npz'))[str(self.L)]
        E = PottsHistogram(L).E
        
        super(PottsEntropy, self).__init__(E, s, normed)
        
