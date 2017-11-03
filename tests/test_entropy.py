"""
Plot precomputed entropy density of the Ising model
"""
import numpy as np
import paths as pth
import seaborn as sns
import matplotlib.pylab as plt

sns.set(style='ticks', palette='Paired', context='notebook', font_scale=1.75)

fig, ax = plt.subplots(1,1,figsize=(10,10))

for L in (4,8,16,32,64,128):

    entropy = pth.IsingEntropy(L)
    print L, entropy.log_Z(1.), entropy.log_Z(1.)/L**2

    ax.plot(entropy.E.astype('d')/L**2, entropy.s/L**2, label='L={}'.format(L))

ax.set_xlabel(r'energy density $\epsilon = E/L^2$')
ax.set_ylabel(r'entropy density $s(\epsilon) = \log g(L^2 \epsilon) / L^2$')
ax.legend()

