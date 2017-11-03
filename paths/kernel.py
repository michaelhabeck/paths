import numpy as np

from scipy import optimize

from .models import Gaussian

class Kernel(object):

    def __call__(self, x):
        raise NotImplementedError

    @property
    def stationary(self):
        pass

class GaussianKernel(Kernel):
    """GaussianKernel

    Gaussian transition kernel with prescribed stationary distribution
    """
    def __init__(self, tau=0., mu=0., sigma=1.):
        """
        Parameters
        ----------
        tau : float in [0., 1.]
          parameter specifying the convergence of the transition kernel

        mu, sigma : float
          mean and standard deviation of the Gaussian stationary distribution
        """
        self._tau   = float(tau)
        self._mu    = float(mu)
        self._sigma = float(sigma)

    def __str__(self):
        return 'GaussianKernel(tau={0:.3e}, mu={1:.2f}, sigma={2:.2f})'.format(
            self.tau, self._mu, self._sigma)

    @property
    def tau(self):
        return self._tau

    @property
    def _mu(self):
        return self._mu_

    @_mu.setter
    def _mu(self, value):
        self._mu_ = float(value)

    def mu(self, y=0):
        return self.tau * y + (1-self.tau) * self._mu

    @property
    def _sigma(self):
        return self._sigma_

    @_sigma.setter
    def _sigma(self, value):
        self._sigma_ = float(value)

    @property
    def sigma(self):
        return np.sqrt(1 - self.tau**2) * self._sigma
        
    @property
    def stationary(self):
        return Gaussian(self._mu, self._sigma)

    def sample(self, n=None, y=0):
        if np.iterable(y): n = len(y)
        return np.random.standard_normal(n) * self.sigma + self.mu(y)

    def compose(self, other):

        tau1, mu1, s1 = self.tau, self._mu, self._sigma
        tau2, mu2, s2 = other.tau, other._mu, other._sigma
        
        tau = tau1 * tau2
        mu  = ((1-tau1) * mu1 + tau1*(1-tau2)*mu2) / (1 - tau)
        s   = ((1-tau1**2) * s1**2 + tau1**2 * (1-tau2**2) * s2**2) / (1-tau**2)

        return GaussianKernel(tau, mu, s**0.5)

    def power(self, n):
        return GaussianKernel(self.tau**n, self._mu, self._sigma)
        
class Bridge(GaussianKernel):

    @property
    def tau(self):
        tau0 = self.initial_kernel.tau
        tau1 = self.target_kernel.tau

        return (1-self.beta) * tau0 + self.beta * tau1

    def __init__(self, beta, initial_kernel, target_kernel):
        """
        Parameters
        ----------
        beta : float
          inverse temperature
          
        initial_kernel, target_kernel : GaussianKernel
          initial and target kernel
        """
        self.initial_kernel = initial_kernel
        self.target_kernel  = target_kernel
        
        self.beta = float(beta)

        super(Bridge, self).__init__(self.tau) 

    @property
    def _mu(self):

        tau0 = self.initial_kernel.tau
        mu0  = self.initial_kernel._mu
        
        tau1 = self.target_kernel.tau
        mu1  = self.target_kernel._mu
        
        return ((1-self.beta) * (1-tau0) * mu0 + \
                    self.beta * (1-tau1) * mu1) / (1-self.tau)

    @_mu.setter
    def _mu(self, value):
        pass
    
    @property
    def _sigma(self):

        tau0   = self.initial_kernel.tau
        sigma0 = self.initial_kernel._sigma
        
        tau1   = self.target_kernel.tau
        sigma1 = self.target_kernel._sigma
        
        return np.sqrt(((1-self.beta) * (1-tau0**2) * sigma0**2 + \
                            self.beta * (1-tau1**2) * sigma1**2) / (1-self.tau**2))

    @_sigma.setter
    def _sigma(self, value):
        pass

class GeometricBridge(Bridge):

    @property
    def _mu(self):

        sigma0 = self.initial_kernel._sigma
        sigma1 = self.target_kernel._sigma
        
        mu0 = self.initial_kernel._mu
        mu1 = self.target_kernel._mu

        return  self._sigma**2 * ((1-self.beta) * mu0 / sigma0**2 + \
                                      self.beta * mu1 / sigma1**2)

    @_mu.setter
    def _mu(self, value):
        pass
    
    @property
    def _sigma(self):

        sigma0 = self.initial_kernel._sigma
        sigma1 = self.target_kernel._sigma
        
        return 1 / np.sqrt((1-self.beta) / sigma0**2 + self.beta / sigma1**2)

    @_sigma.setter
    def _sigma(self, value):
        pass

class Scheduler(object):

    def __init__(self, start, end, bridge_constructor=Bridge):

        self.start = start
        self.end   = end

        self._bridge = bridge_constructor

    def schedule(self, incr):
        incr = incr**2
        return np.append(0, np.add.accumulate(incr)/incr.sum())

    def __call__(self, x):

        bridge = [self._bridge(beta, self.start, self.end)
                  for beta in self.schedule(x)]
        prob   = [T.stationary for T in bridge]

        kl = np.array([p.kl(q) for p, q in zip(prob,prob[1:])])

        return np.sum((kl-kl.mean())**2)
        
    def find_schedule(self, length_bridge):

        x = np.ones(int(length_bridge))
        y = optimize.fmin_powell(self, x)

        return self.schedule(y)

