import numpy as np
import matplotlib.pyplot as plt
from pylab import rc
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import scipy.stats as stats

# cyclical variability parameters
P = 0.05043723 
P_sig = 3.3e-6
T0 = 0
T0_sig = 6.1e-3
N_cycles = 243

# sampling parameter distributions
samples_P = np.random.normal(P, P_sig, size=1000000)
samples_T = np.random.normal(T0, T0_sig, size=1000000)

# propagating forward in time sampled parameters
sim = samples_T + samples_P*N_cycles

# getting distribution parameters of the simulation
sim_mean = np.mean(sim)
sim_std = np.std(sim, ddof=1)

# setting up plotting environment
rc('axes', linewidth=2)
fig, ax = plt.subplots(1, 1, figsize=(20,10))

# plotting histogram of the simulation
ax.hist(sim/P, bins=50, color='black', edgecolor='black')

# plotting modeled distribution
x = np.linspace(sim_mean - 5*sim_std, sim_mean + 5*sim_std, 100)
pdf = stats.norm.pdf(x, sim_mean, sim_std) 
ylim = max(plt.ylim())
pdf = (pdf/max(pdf))*ylim
ax.plot(x/P, pdf, c='b')

# marking regions beyond which cycles can be mistaken
ax.plot([N_cycles + 0.5, N_cycles + 0.5],[0,ylim], c='r')
ax.plot([N_cycles - 0.5, N_cycles - 0.5],[0,ylim], c='r')

# plot parameters
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='y',which='both', width=2,direction="in",pad=10)
ax.tick_params(axis='y',which='major', length=14,direction="in")
ax.tick_params(axis='x',which='both', width=2,direction='in')
ax.tick_params(axis='x',which='major', length=14,direction='in',pad=10)
ax.axes.get_yaxis().set_visible(False)
ax.axes.get_yaxis().set_ticks([])
ax.tick_params(axis='x',which='minor', length=8)
ax.tick_params(axis='y',which='minor', length=8,direction='in')
ax.tick_params(labelsize=22)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.set_xlabel('Cycles', fontsize=22)
plt.savefig('CycleSpread.pdf')