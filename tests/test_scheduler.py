import numpy as np
import paths as pth
import matplotlib.pylab as plt

start = pth.GaussianKernel(0.9, 20., 10.)
end   = pth.GaussianKernel(0.995, 0., 1.)

log_Z = end.stationary.log_Z - start.stationary.log_Z

scheduler = pth.Scheduler(start, end, pth.Bridge)
schedule  = scheduler.find_schedule(10)

scheduler = pth.Scheduler(start, end, pth.GeometricBridge)
schedule2 = scheduler.find_schedule(10)

bridge  = pth.make_bridge(start, end, schedule, constructor=pth.Bridge)
bridge2 = pth.make_bridge(start, end, schedule2, constructor=pth.GeometricBridge)

prob  = [T.stationary for T in bridge]
prob2 = [T.stationary for T in bridge2]

kl  = [q.kl(p) for p, q in zip(prob[1:],prob)]
kl2 = [q.kl(p) for p, q in zip(prob2[1:],prob2)]

print 'first bridge', np.mean(kl), np.std(kl)
print 'second bridge', np.mean(kl2), np.std(kl)

colors = ('b', 'g')
probs  = (prob, prob2)

fig, ax = plt.subplots(1,1,figsize=(10,10))

for i, P in enumerate(probs):

    mu  = np.array([p.mu for p in P])
    sig = np.array([p.sigma for p in P])
    c   = colors[i]
    
    ax.fill_between(schedule, mu-sig, mu+sig, color=c, alpha=0.3)
    ax.plot(schedule, mu, lw=3, color=c)
    ax.plot(schedule, mu-sig, lw=2, ls='--', color=c)
    ax.plot(schedule, mu+sig, lw=2, ls='--', color=c)

ax.semilogx()
