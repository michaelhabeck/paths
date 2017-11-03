from .models import Gaussian
from .kernel import GaussianKernel, Bridge, GeometricBridge, Scheduler
from .simulate import make_bridge, simulate
from .ising import IsingModel, IsingKernel
from .entropy import Entropy, IsingEntropy

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


