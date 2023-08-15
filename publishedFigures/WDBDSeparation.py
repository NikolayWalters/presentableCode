import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pylab import rc
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib import transforms
from pylab import *
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times']
from adjustText import adjust_text

# separation, mass and label arrays arrays
x = np.array([0.38,0.431,0.686,1.98,0.593,0.569,
             0.699,0.448,2.04,0.496])
y = np.array([0.05,0.049,0.0665,0.0593,0.053,0.053,
             0.063,0.0668,0.0649,0.095])
labels = ['SDSS J1411+2009','SDSS J1205-0242',
         'WD 1032+011', 'ZTF J0038+2030',
         'WD 0137-349', 'NLTT 5306',
         'SDSS J1557+0916', 'EPIC212235321',
         'GD 1400', 'SDSS J1231+0041']

# Global plot parameters
plt.rc('xtick',c='k')
lw=1.2
rc('axes', linewidth=2)
plt.rc('ytick',c='k')
fig, ax = plt.subplots(1, 1, figsize=(12,10))

# plotting brown dwarfs
scatter = ax.scatter(x, y, c='b')

# plotting the single known planet
ax.scatter([4.387], [0.011], c='r')

# creating label array
texts = []
for i, txt in enumerate(labels):
    texts.append(ax.text(x[i], y[i], txt, fontsize=12))
    
# individual planet annotation
ax.annotate('WD 1856+534', (4.387, 0.011), fontsize=12, xytext=(-30,5), textcoords='offset points')

# annotating brown dwarfs using adjust_text to avoid label overlap
adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k'))

# Set plot parameters
ax.xaxis.set_major_locator(MultipleLocator(8))
ax.xaxis.set_minor_locator(MultipleLocator(2))
ax.tick_params(axis='y',which='both', width=2,direction="in",pad=10)
ax.tick_params(axis='y',which='major', length=14,direction="in")
ax.tick_params(axis='x',which='both', width=2,direction='in')
ax.tick_params(axis='x',which='major', length=14,direction='in',pad=10)
ax.yaxis.set_major_locator(MultipleLocator(0.01))
ax.yaxis.set_minor_locator(MultipleLocator(0.0025))
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.25))
ax.tick_params(axis='x',which='minor', length=8)
ax.tick_params(axis='y',which='minor', length=8,direction='in')
ax.tick_params(labelsize=22)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.set_xlabel('Orbital sepraration (R$_\odot$)', fontsize=22)
ax.set_ylabel('Companion mass (M$_\odot$)', fontsize=22)
ax.set_xlim(0.0128, 5)
ax.set_ylim(0.0,0.1)

# crop and save the plot
bboxx = transforms.Bbox([[0.3, 0.45], [11., 8.95]])
plt.savefig('BDWDSeparation.pdf', bbox_inches=bboxx)