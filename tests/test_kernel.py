import paths as pth
import numpy as np
import matplotlib.pylab as plt

def test_composition():

    T = pth.GaussianKernel(0.9, np.random.standard_normal() * 10, np.random.gamma(1))

    print T
    print T.compose(T.compose(T.compose(T)))
    print T.power(4)
    print T.stationary

def test_simulation(n=1e3, tau=0.995):

    T = pth.GaussianKernel(tau)
    X = [np.random.random(int(n)) * 100]

    for i in xrange(1000):
        X.append(T.sample(y=X[-1]))

    X = np.array(X)

    fig, ax = plt.subplots(1,2,figsize=(8,4))
    ax[0].plot(X.mean(1))
    ax[0].axhline(T._mu,ls='--',lw=3)
    ax[1].plot(X.std(1))
    ax[1].axhline(T._sigma,ls='--',lw=3)

test_composition()
test_simulation()

