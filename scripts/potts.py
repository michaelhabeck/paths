"""
Compute forward / reverse simulations of the Potts model
and compare evidence estimates with value from Wang-Landau.
"""
import numpy as np
import paths as pth
import seaborn as sns
import matplotlib.pylab as plt

from paths.potts import PottsHistogram
from paths.estimators import jarzynski, cumulant, bar

from csb.numeric import log_sum_exp

sns.set(style='ticks', palette='Set2', context='notebook', font_scale=1.75)

Q        = 10           ## number of colors
L        = (16, 32)[0]  ## size of Potts model: L**2 spins
n_paths  = 1000         ## number of paths that will be simulated
n_beta   = 100          ## length of inverse temperature schedule
n_relax  = 1e4          ## number of color flips used in transition kernel

entropy  = pth.PottsEntropy(L)
beta     = np.linspace(0., 2., n_beta)
hist     = PottsHistogram(L)
bridge   = [pth.PottsKernel(L, Q, b, n_relax) for b in beta]
energy   = pth.PottsModel(L, Q, beta=1).energy
E_mean   = np.array(map(entropy.E_mean, beta))

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

## select initial states from reverse simulation according to
## importance weights of final states from forward simulation

X_r = [X_f[i] for i in np.random.multinomial(1,p,size=n_paths).argmax(1)]
E_r = [map(energy, X_r)]

## backward simulation using reverse bridge (detailed balance!)

for T in bridge[::-1][1:]:
    X_r  = [T(x) for x in X_r]
    E_r += [map(energy, X_r)]
    
E_r = np.array(E_r)
W_r = np.dot(beta[::-1][1:]-beta[::-1][:-1], E_r[:-1])

## report estimated evidence and compare to exact value

est = (entropy.log_Z(beta.max()),
       -jarzynski(+W_f),
       -cumulant(W_f), 
       +jarzynski(+W_r),
       -cumulant(W_f, W_r),
       -bar(W_f, -W_r))

out = 'log(Z)={0:.2f}, JE_f={1:.2f}, Cum_f={2:.2f}, JE_r={3:.2f}, Cum={4:.2f}, BAR={5:.2f}'

print out.format(*est)

## plot work distributions and hysteresis

_, bins = np.histogram(np.append(W_f, -W_r), bins=50)

kw_hist = dict(alpha=0.3, normed=True, histtype='stepfilled', bins=bins)
kw_plot = dict(alpha=0.5, lw=3)

fig, ax = plt.subplots(1,2,figsize=(10,5))

ax[0].hist(+W_f, color='b', label=r'$p_f$', **kw_hist)
ax[0].hist(-W_r, color='g', label=r'$p_r$', **kw_hist)
ax[0].axvline(-entropy.log_Z(beta.max()), ls='--', color='k', label=r'$-log\,Z$')
ax[0].set_xlabel(r'work $W=-\log w$')
ax[0].set_ylabel(r'work distributions $p(W)$')
ax[0].legend()

label = r'$\langle E \rangle_{0}$'

ax[1].axvline(1/0.707, color='r', ls='--', label=r'$\beta_{crit}$', **kw_plot)
ax[1].plot(beta, E_mean, color='k', ls='--', label=label.format(r'\beta'), **kw_plot)
ax[1].plot(beta, E_f.mean(1), color='b', label=label.format('f'), **kw_plot)
ax[1].plot(beta[::-1], E_r.mean(1), color='g', label=label.format('r'), **kw_plot)
ax[1].set_xlabel(r'inverse temperature $\beta$')
ax[1].set_ylabel(r'average energy $\langle E \rangle$')
ax[1].legend()

fig.tight_layout()

