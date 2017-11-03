import numpy as np
import paths as pth
import seaborn as sns
import matplotlib.pylab as plt

from csb.numeric import log_sum_exp
from paths.estimators import bar, jarzynski

sns.set(style='ticks', palette='Paired', context='notebook', font_scale=1.75)

colors    = sns.color_palette()
colors[4] = 'k'

## specify initial and target distribution

start     = pth.GaussianKernel(0.9, 20., 10.)
end       = pth.GaussianKernel(0.995, 0., 1.)
log_Z     = end.stationary.log_Z - start.stationary.log_Z

## load precomputed optimized inverse temperature schedules

schedules = np.load('../tests/data/schedules.npz')

estimators = []

for K in sorted(map(int, schedules.keys())):

    schedule = schedules[str(K)]
    bridge   = pth.make_bridge(start, end, schedule, 5, pth.GeometricBridge)
    W_f      = +pth.simulate(bridge, 1e4)[0]
    W_r      = -pth.simulate(bridge[::-1], 1e4)[0]

    estimators.append((len(schedule), log_Z, -jarzynski(W_f), -W_f.mean(), -W_r.mean()))

    print estimators[-1]
        
estimators = np.transpose(estimators)

fig, ax = plt.subplots(1,1,figsize=(6,6))
    
ax.plot(estimators[0], estimators[3], label=r'$-\langle W \rangle_f$', lw=3)
ax.plot(estimators[0], estimators[4], label=r'$-\langle W \rangle_r$', lw=3)
ax.plot(estimators[0], estimators[2], label=r'$\log \langle e^{-W} \rangle_f$', lw=4)
ax.plot(estimators[0], estimators[1], color=colors[4], label=r'$\log Z$', ls='--', lw=2)
ax.set_xlabel(r'$K$')
ax.set_ylabel(r'work $W$')
ax.set_xlim(0, 200.)
ax.set_ylim(-13, 1.)
ax.set_xticks(np.arange(0,201,50))
ax.legend()

fig.tight_layout()
