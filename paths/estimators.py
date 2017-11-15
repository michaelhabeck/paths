"""
Various evidence estimators for forward / backward MCMC simulations
"""
import numpy as np

from .entropy import Entropy

from csb.numeric import log_sum_exp

def jarzynski(w):
    """
    Estimator based on Jarzynski equality. This is the estimator used
    in standard annealed importance sampling.
    """
    return np.log(len(w))- log_sum_exp(-w)

def cumulant(w_f, w_r=None):
    """
    Cumulant estimators developed by Hummer.
    """
    if w_r is None:
        return np.mean(w_f) - 0.5 * np.var(w_f)
    else:
        return 0.5 * (np.mean(w_f) - np.mean(w_r)) - (np.var(w_f) - np.var(w_r)) / 12.

def bar(w_f, w_r, tol=1e-4):
    """
    Bennett's acceptance ratio
    """
    dF = 0.5 * (jarzynski(w_f) - jarzynski(-w_r))
    
    while 1:
        
        lhs  = log_sum_exp(-np.log(1 + np.exp(+w_f - dF)))
        rhs  = log_sum_exp(-np.log(1 + np.exp(-w_r + dF)))

        incr = rhs - lhs 
        
        dF  += incr

        if abs(incr) < tol: break

    return dF

def histogram(w_f, w_r, n_iter=1e5, tol=1e-10, alpha=0., return_histogram=False):
    """
    Histogram estimator analogous to histogram methods used
    in DOS estimation
    """
    w = np.append(w_f,w_r)

    N = np.array([len(w_f), len(w_r)])
    q = np.multiply.outer(np.array([0.,1.])-alpha, w)
    p = np.zeros(len(w)) - np.log(len(w))

    L = []

    for _ in xrange(int(n_iter)):

        f = -log_sum_exp((-q + p).T, 0)
        
        ## store log likelihood and report on progress

        L.append(-np.dot(N,f) - p.sum())

        ## update log histogram and normalize
        
        p  = -log_sum_exp((-q.T + f + np.log(N)).T, 0)
        p -=  log_sum_exp(p)

        if len(L) > 1 and abs((L[-2]-L[-1]) / (L[-2]+L[-1])) < tol:
            break

    p = Entropy(w, p)

    if return_histogram:
        return p.log_Z(-alpha)-p.log_Z(1-alpha), p
    else:
        return p.log_Z(-alpha)-p.log_Z(1-alpha)
