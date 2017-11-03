import paths as pth
import numpy as np
import matplotlib.pylab as plt

def test_composition():

    T = pth.GaussianKernel(0.9, np.random.standard_normal() * 10, np.random.gamma(1))

    print T
    print T.compose(T.compose(T.compose(T)))
    print T.power(4)
    print T.stationary

def test_simulation(n=1e2, tau=0.995):

    T = pth.GaussianKernel(tau)
    X = [np.random.random(int(n)) * 100]

    for i in xrange(1000):
        X.append(T(X[-1]))

    X = np.array(X)

    ## plot convergence to stationary distribution
    
    fig, ax = plt.subplots(1,2,figsize=(8,4))
    fig.suptitle(r'$\tau={0:.3e}$'.format(tau), y=1.)
    
    [ax[0].plot(x,lw=2,color='k',alpha=0.1) for x in X.T]

    ax[0].plot(X.mean(1), lw=3, color='r')
    ax[0].axhline(T.stationary.mu,ls='--',lw=3, label=r'$\mu$',color='k')
    ax[0].set_xlabel('iteration')
    ax[0].set_ylabel('mean')
    ax[0].legend()
    
    ax[1].plot(X.std(1), color='r')
    ax[1].axhline(T.stationary.sigma,ls='--',lw=3, label=r'$\sigma$',color='k')
    ax[1].set_xlabel('iteration')
    ax[1].set_ylabel('standard deviation')
    ax[1].legend()

    fig.tight_layout()

test_composition()

## probabing different relaxation rates

test_simulation(tau=1-5e-2)
test_simulation(tau=1-5e-3)
test_simulation(tau=1-5e-4)

