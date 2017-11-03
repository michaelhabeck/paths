import time
import contextlib

def format_time(t):

    units = [(1.,'s'),(1e-3,'ms'),(1e-6,'us'),(1e-9,'ns')]
    for scale, unit in units:
        if t > scale or t==0: break
        
    return '{0:.1f} {1}'.format(t/scale, unit)

@contextlib.contextmanager
def take_time(desc):
    t0 = time.clock()
    yield
    dt = time.clock() - t0
    print '{0} took {1}'.format(desc, format_time(dt))


class Model(object):
    """Model

    Probabilistic model
    """
    def sample(self, x=None, n=None):
        raise NotImplementedError

    def energy(self, x):
        """
        Minus log probability
        """
        raise NotImplementedError

class Kernel(object):
    """Kernel
    
    Markov transition kernel
    """
    def __call__(self, x):
        """
        Transition starting from given state 'x'
        """
        raise NotImplementedError

    @property
    def stationary(self):
        """
        Stationary distribution (instance of Model)
        """
        pass

