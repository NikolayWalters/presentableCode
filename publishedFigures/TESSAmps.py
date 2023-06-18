"""
Plot showing estimated TESS semi-amplitudes
for different sectord
"""


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pylab import rc
import numpy as np

rc('axes', linewidth=2)
plt.figure(figsize=(20, 10))

x = [1750.599568993838,1942.5369130833083,1969.661272981842,
 1996.4705435535009,2022.6838904073516,2680.349936342566,
 2730.86022603806,2838.9257048053105,2924.428049924587]
y = [0.00660187919,0.00606687145,0.00566619593,
 0.00623091178,0.00767144050,0.00631751061,
 0.00561225022,0.00623003709,0.00549583767]
y_err = [0.0003146561982,0.0002977904329,0.0002728579721,
 0.0003832003734,0.0005224671393,0.0003341725220,
 0.0003554759400,0.0002960421190,0.0003025942468]

# Create the scatter plot with y errors
sns.scatterplot(x=x, y=100*np.array(y))
plt.errorbar(x, 100*np.array(y), yerr=100*np.array(y_err),capsize=6, c='k', marker='o',
             ecolor='k', ls='', ms=16,elinewidth=3, label='TESS')

# Mean value line
plt.plot([1700,3000], [100*0.006210326048888889,100*0.006210326048888889], c='k', ls='--', label='Mean')


# Add the trendline
sns.regplot(x=x, y=100*np.array(y), scatter=False,ci=95,color='blue', scatter_kws={'color': 'black'},label='Best fit')

# Plot aesthetics
plt.legend(prop={'size': 35})
plt.xlim(1700,3000)
plt.xticks(fontsize=38)
plt.yticks(fontsize=38)
plt.tick_params(axis='both', length=14, width=2, direction='in')
plt.minorticks_on()
plt.tick_params(axis='x',which='minor', length=8, width=2, direction='in')
plt.tick_params(axis='y',which='minor', length=0, width=2, direction='in')

plt.ylabel('Light Curve Semi-amplitude (%)',fontsize=36)
plt.xlabel('BJD - 2457000 (days)',fontsize=36)

plt.tight_layout()
plt.savefig('TESSAmpsThesis.pdf')