"""
A script to plot 2 histograms with colour depending on magnetic DA
white dwarf crystallization status. The histograms show log magnetic
field strength distribution.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import transforms
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from pylab import rc
from pylab import *
rcParams['font.sans-serif'] = ['Times']
rc('axes', linewidth=2)

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('AllMagneticDA.csv')

# Extract the 'crystallized' column
crystallized_1 = data[data['isCryst'] == 1]
crystallized_0 = data[data['isCryst'] == 0]
# Create a figure and axis object
fig, ax = plt.subplots(figsize=(20,10))
bins = np.linspace(min(np.log10(data['B'])), max(np.log10(data['B'])), 21)

ax.hist(np.log10(crystallized_1['B']), bins=bins, density=True, histtype='step', edgecolor='black', color='r', label='20%+ crystallized', fill=True)

ax.hist(np.log10(crystallized_0['B']), bins=bins, density=True, histtype='step', edgecolor='black', fill=False, hatch='///',label='Less than 20% crystallized')



# Add labels and a legend
ax.set_xlabel('log$_1$$_0$B (MG)',fontsize=36)
ax.set_ylabel('Relative Frequency',fontsize=36)
plt.legend(prop={'size': 35})
plt.xticks(fontsize=38)
plt.yticks(fontsize=38)
plt.tick_params(axis='y', length=14, width=2, direction='in')
plt.tick_params(axis='x', length=14, width=2, direction='out')
plt.minorticks_on()
plt.tick_params(axis='x',which='minor', length=8, width=2, direction='out')
plt.tick_params(axis='y',which='minor', length=0, width=2, direction='in')
plt.ylim(0,2.999999)
bboxx = transforms.Bbox([[1., -0.1], [18.2, 8.84]])
plt.savefig('ThesisMagneticDACristDist.pdf', bbox_inches=bboxx)