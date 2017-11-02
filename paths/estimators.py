"""
Various evidence estimators for forward / backward MCMC simulations
"""
import numpy as np

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
    Bennet acceptance ratio
    """
    dF = 0.5 * (jarzynski(w_f) - jarzynski(-w_r))
    
    while 1:
        
        lhs  = log_sum_exp(-np.log(1 + np.exp(+w_f - dF)))
        rhs  = log_sum_exp(-np.log(1 + np.exp(-w_r + dF)))

        incr = rhs - lhs 
        
        dF  += incr

        if abs(incr) < tol: break

    return dF

def histogram(w_f, w_r, alpha=0.):

    w = np.append(w_f,w_r)
    wham = WHAM(len(w), 2)
    wham.H += 1./len(w)
    wham.N[:] = len(w_f)

    q = np.multiply.outer(np.array([0.,1.])-alpha, w)
    wham.run(q, niter=int(1e5), tol=1e-10, verbose=0)

    p = DOS(w, wham.s)

    return p.log_Z(-alpha)-p.log_Z(1-alpha), p

