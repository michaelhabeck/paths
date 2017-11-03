"""
Compute forward / reverse simulations of the Ising model
and compare with exact value.
"""
import numpy as np
import paths as pth
import seaborn as sns
import matplotlib.pylab as plt

from paths.estimators import jarzynski, cumulant, bar

from csb.numeric import log_sum_exp

sns.set(style='ticks', palette='Paired', context='notebook', font_scale=1.75)

L        = (16, 32)[0]  ## size of Ising model: L**2 spins
n_paths  = 1000         ## number of paths that will be simulated
n_beta   = 20           ## length of inverse temperature schedule
n_relax  = 1e3          ## number of spin flips used in transition kernel

entropy  = pth.IsingEntropy(L)
beta     = np.linspace(0., 1., n_beta)
bridge   = [pth.IsingKernel(L, b, n_relax) for b in beta]
energy   = pth.IsingModel(L,beta=1.).energy

## forward sampling, forward work

X_f = [bridge[0].stationary.sample() for _ in xrange(n_paths)]
E_f = [map(energy, X_f)]

for T in bridge[1:]:
    X_f  = [T(x) for x in X_f]
    E_f += [map(energy, X_f)]
    
E_f = np.array(E_f)
W_f = np.dot(beta[1:]-beta[:-1], E_f[:-1])

## compute importance weights of final states

p = np.exp(-W_f-log_sum_exp(-W_f))
p/= p.sum()

## select initial states from reverse simulation accoring to
## importance weights of final states from forward simulation

X_r = [X_f[i] for i in np.random.multinomial(1,p,size=n_paths).argmax(1)]
E_r = [map(energy, X_r)]

## backward simulation using reverse bridge (detailed balance)

for T in bridge[::-1][1:]:
    X_r  = [T(x) for x in X_r]
    E_r += [map(bridge[-1].stationary.energy, X_r)]
    
E_r = np.array(E_r)
W_r = np.dot(beta[::-1][1:]-beta[::-1][:-1], E_r[:-1])

## report estimated evidence and compare to exact value

est = (entropy.log_Z(1.),
       -jarzynski(+W_f),
       -cumulant(W_f), 
       +jarzynski(+W_r),
       -cumulant(W_f, W_r),
       -bar(W_f, -W_r))

out = 'log_Z={0:.2f}, JE_f={1:.2f}, Cum_f={2:.2f}, JE_r={3:.2f}, Cum={4:.2f}, BAR={5:.2f}'

print out.format(*est)

## plot work distributions and hysteresis

_, bins = np.histogram(np.append(W_f, -W_r), bins=100)

kw_hist = dict(alpha=0.3, normed=True, histtype='stepfilled', bins=bins)

fig, ax = plt.subplots(1,2,figsize=(8,4))

ax[0].hist(+W_f, color='b', label=r'$W_f$', **kw_hist)
ax[0].hist(-W_r, color='g', label=r'$W_r$', **kw_hist)
ax[0].axvline(-entropy.log_Z(1.), ls='--', color='k', label=r'$-log\,Z$')
ax[0].legend()

ax[1].plot(beta, E_f.mean(1), color='b')
ax[1].plot(beta[::-1], E_r.mean(1), color='g')

fig.tight_layout()
