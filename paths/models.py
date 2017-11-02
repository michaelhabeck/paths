import numpy as np

class Model(object):

    def sample(self, n=None):
        raise NotImplementedError

    def sample(self, x):
        raise NotImplementedError

class Gaussian(Model):

    @property
    def log_Z(self):
        return 0.5 * np.log(2*np.pi*self.sigma**2)

    def __init__(self, mu=0., sigma=1.):

        super(Gaussian, self).__init__()

        self.mu    = float(mu)
        self.sigma = float(sigma)

    def sample(self, n=None):        
        return np.random.standard_normal(n) * self.sigma + self.mu

    def energy(self, x):
        return 0.5 * (self.mu - x)**2 / self.sigma**2

    def __str__(self):
        return 'Gaussian(mu={0:.2f}, sigma={1:.2f})'.format(self.mu, self.sigma)

    def kl(self, other):
        return 0.5 * ((self.sigma**2 + (self.mu - other.mu)**2) / other.sigma**2 - 1 + \
                      np.log(other.sigma**2/self.sigma**2))


