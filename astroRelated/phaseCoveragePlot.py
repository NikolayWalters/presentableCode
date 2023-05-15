"""
A script to generate phase coverage plot for HST GD 356 observations
"""

import glob
from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np
from pylab import rc

# read in spectra
filepath = '/home/nwal/Desktop/hstData/red/'
files = sorted(glob.glob(filepath+'*.fits'))
period = 1/12.453166316631663 # days
startTimes = []
endTimes = []

# extract start and end obs dates
for file in files:
    with fits.open(file) as hdu:
        start = hdu[1].header['EXPSTART']
        end = hdu[1].header['EXPEND']
        startTimes.append(start)
        endTimes.append(end)
        
# calculate phases and remove a phase onset
# so that the first obs starts at phase = 0
phaseOffset = (startTimes[0]/period)
timeOffset = period*phaseOffset
startTimes = np.array(startTimes) - timeOffset
endTimes = np.array(endTimes) - timeOffset
startPhases = (startTimes/period)
endPhases = (endTimes/period)

# account for phase overflow, i.e. when ending phase < start phase
adjStartPhases = []
adjEndPhases = []
for sta, end in zip(startPhases,endPhases):
    if sta%1 > end%1:
        adjStartPhases.append(sta%1)
        adjEndPhases.append(1)
        adjStartPhases.append(0)        
        adjEndPhases.append(end%1)
    else:
        adjStartPhases.append(sta%1)
        adjEndPhases.append(end%1)

# generate orbits and insert an extra orbit for phase overflow
orbits = np.arange(1,7)
orbits = np.insert(orbits,1,2)

# plot results
rc('axes', linewidth=4)
plt.figure(figsize=(20,10))
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 4
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 4
plt.tick_params(axis='both', direction='in')
plt.tick_params(axis='both', labelsize=24)
plt.xlabel('Phase', fontsize=24)
plt.ylabel('Orbit', fontsize=24)
plt.xlim(0,1)
plt.title('FUV', fontsize=24)
for count,el in enumerate(orbits):
    plt.plot([adjStartPhases[count],adjEndPhases[count]], [el,el], c='k', lw=4)
for el in np.arange(1,10)/10:
    plt.plot([el,el],[1,6], linestyle='dotted', lw=2, c='k')
plt.savefig('FUVPhaseCoverage.pdf')