import numpy as np

from .gaussian import Bridge

def make_bridge(start, end, schedule, n=1, constructor=Bridge):
    """
    Construct a 'bridge', i.e. a sequence of transition kernels

    Parameters
    ----------
    start, end : TransitionKernels
      transition kernels whose stationary distributions are the
      initial and final ensemble

    schedule : iterable
      inverse temperature schedule

    n : integer > 0
      power to which the intermediate transition kernels will be
      raised

    constructor :
      constructor for the bridge
    """
    bridge = [constructor(beta, start, end) for beta in schedule]
    if n > 1:
        bridge = [T.power(n) for T in bridge]

    return bridge

def generate_paths(bridge, n_paths=1, store_paths=False):
    """
    Run a nonequilibrium simulation by stepping through a sequence
    of Markov perturbations
    
    Parameters
    ----------
    bridge : iterable
      sequence of transition kernels

    n_paths : integer
      number of paths that will be simulated

    store_paths : boolean
      flag that specifies if the full paths will be return or only
      the final states
    """
    X = [bridge[0].stationary.sample(n=int(n_paths))]

    for T in bridge[1:]:
        x = T(X[-1])
        X.append(x)
        
    return np.array(X)

def simulate(bridge, n_paths=1):
    """
    Generate multiple paths from the bridge and compute the work
    Returns log weights (work) and final states (weighted samples
    from the target ensemble).
    """
    X = generate_paths(bridge, n_paths)
    p = [T.stationary for T in bridge]
    W = np.sum([p[k+1].energy(X[k]) - p[k].energy(X[k])
                for k in range(len(bridge)-1)],0)

    return W, X[-1]

