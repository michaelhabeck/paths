import numpy as np
import paths as pth
import seaborn as sns
import matplotlib.pylab as plt

from csb.numeric import log_sum_exp
from paths.estimators import bar, jarzynski

sns.set(style='ticks', palette='Paired', context='notebook', font_scale=1.75)

start     = pth.GaussianKernel(0.9, 20., 10.)
end       = pth.GaussianKernel(0.995, 0., 1.)
log_Z     = end.stationary.log_Z - start.stationary.log_Z

constructor = pth.GeometricBridge
scheduler   = pth.Scheduler(start, end, constructor)
schedule    = scheduler.find_schedule(10)

bridge    = pth.make_bridge(start, end, schedule, 5, constructor=constructor)
p         = [T.stationary for T in bridge]
q         = [p[0]]

for T in bridge[1:]:

    mu, sigma, tau = T._mu, T._sigma, T.tau

    mu  = (1-tau) * mu + tau * q[-1].mu
    var = (1-tau**2) * sigma**2 + tau**2 * q[-1].sigma**2

    q.append(pth.Gaussian(mu,var**0.5))

n_paths = 1e6

W_f, x =  pth.simulate(bridge, n_paths)
W_r    = -pth.simulate(bridge[::-1], n_paths)[0]

p_mu   = np.array([pp.mu for pp in p])
p_std  = np.array([pp.sigma for pp in p])

q_mu   = np.array([pp.mu for pp in q])
q_std  = np.array([pp.sigma for pp in q])

print log_Z, -jarzynski(W_f), jarzynski(-W_r), -bar(W_f, W_r)

_, bins = np.histogram(np.append(W_f,W_r),bins=500)
w = 0.5 * (bins[1:] + bins[:-1])
x = np.arange(len(schedule))

p_f, _ = np.histogram(W_f, bins=bins, normed=True)
p_r, _ = np.histogram(W_r, bins=bins, normed=True)

colors = sns.color_palette()
colors[4] = 'k'
xy = (0.8, 0.85)

fig, ax = subplots(1,2,figsize=(12,6))
ax = ax.flat
ax[0].annotate("A", xy=xy, fontsize=24, xycoords='axes fraction')
ax[0].fill_between(w, w*0, p_f, color=colors[0], alpha=0.5)
ax[0].plot(w, p_f, lw=3, color=colors[1], alpha=0.8, label=r'$p_f(W)$')
ax[0].fill_between(w, w*0, p_r, color=colors[2], alpha=0.5)
ax[0].plot(w, p_r, lw=3, color=colors[3], alpha=0.8, label=r'$p_r(W)$')
ax[0].axvline(-log_Z,ls='--',lw=4,color=colors[4], alpha=0.7, label=r'$-\log Z$')
ax[0].set_xticks(np.arange(0,21,5))
ax[0].set_xlim(-2.5, 20)
ax[0].set_ylim(0., 0.425)
ax[0].set_xlabel(r'work $W$')
ax[0].set_ylabel(r'work distribution $p(W)$')
ax[0].legend(loc=7)

ax[1].annotate("B", xy=xy, fontsize=24, xycoords='axes fraction')
ax[1].plot(x, p_mu, ls='-',lw=3, color=colors[1], label=r'$p_k$')
ax[1].plot(x, q_mu, ls='-',lw=3, color=colors[3], label=r'$q_k$')
ax[1].fill_between(x, p_mu - p_std, p_mu + p_std, color=colors[0], alpha=0.3)
ax[1].fill_between(x, q_mu - q_std, q_mu + q_std, color=colors[2], alpha=0.3)
ax[1].plot(x, p_mu-p_std, ls='--',lw=2, color=colors[1], alpha=0.7)
ax[1].plot(x, p_mu+p_std, ls='--',lw=2, color=colors[1], alpha=0.7)
ax[1].plot(x, q_mu-q_std, ls='--',lw=2, color=colors[3], alpha=0.7)
ax[1].plot(x, q_mu+q_std, ls='--',lw=2, color=colors[3], alpha=0.7)
ax[1].set_xlabel(r'index $k$')
ax[1].set_ylabel(r'parameter $x$')
ax[1].set_xlim(0,len(schedule)-1)    
ax[1].legend(loc=7)

fig.tight_layout()

schedules = np.load('./data/schedules.npz')

estimators = []

for K in sorted(map(int, schedules.keys())):

    schedule = schedules[str(K)]
    bridge   = pth.make_bridge(start, end, schedule, 5, constructor)
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
