"""
Script aimed at analysing the source of TESS variability. Does several things:
1) Plots LS periodogram
2) Folds and re-bins the light curve on the most powerful LS peak
3) Downloads postage stamp and estimates periodograms for each pixel
4) Uses Tess localise to show the location of the most powerful period source 
"""

import numpy as np
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt
from pylab import rc
from pylab import *
import TESS_Localize as tl
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times']
from astropy import units as u
import lightkurve as lk
import re
fontsize = 12 # plot label fontsize 
targetName = 'WD 1009-184' # target name

def foldMeth(timeArray, period):
    """
    Method to transform from time to phase for a specified period
    """
    phases = []
    for el in timeArray:
        passed = int(el/period)
        leftover = el - (passed*period)
        phase = leftover/period
        phases.append(phase)
    return np.array(phases)

# downlaod light curves
search_result = lk.search_lightcurvefile(targetName, exptime=120)
collection = search_result.download_all()
data = collection.PDCSAP_FLUX.stitch()
data = data.remove_nans()
flux = data.flux.value
time = data.time.value

# create periodogram
freq = np.linspace(0.001, 25, 10000)
power = LombScargle(time, flux).power(freq)
bestPeriod = 1/freq[np.argmax(power)] # keep the most significant period

# plot periodogram
plt.figure(figsize=(20,10))
plt.plot(freq,power, c='k')
plt.xlabel('Frequency [1/d]', fontsize=fontsize)
plt.ylabel('LS Power', fontsize=fontsize)
plt.title('Lomb-Scargle Periodogram', fontsize=fontsize)
plt.show()

# plot phase folded light curve with re-binning
plt.figure(figsize=(20,10))
phases = foldMeth(time, bestPeriod)
sortIndi = np.argsort(phases)
phases = phases[sortIndi]
flux = flux[sortIndi]
numPoints = 200 # number of points to bin
flux = np.nanmean(np.pad(flux.astype(float), (0, numPoints - flux.size%numPoints),
         mode='constant', constant_values=np.NaN).reshape(-1, numPoints), axis=1)
phases = np.nanmean(np.pad(phases.astype(float), (0, numPoints - phases.size%numPoints),
         mode='constant', constant_values=np.NaN).reshape(-1, numPoints), axis=1)
plt.plot(phases, flux, marker='o',mfc='none', c='k',ls='')
plt.plot(phases+1, flux, marker='o',mfc='none', c='k',ls='')
plt.title('Folded on '+ str(np.round(bestPeriod,decimals=2)) +' days', fontsize=fontsize)
plt.xlabel('Phase', fontsize=fontsize)
plt.ylabel('Normalised Flux', fontsize=fontsize)
plt.show()

# get the first observed sector number
string = search_result[0].mission[0]
pattern = r'\d+'
match = re.search(pattern, string)
sector = int(match.group())

# download target pixel file (postage stamp)
tpf = lk.search_targetpixelfile(targetName, sector = sector).download()

# instead of LS can switch to BLS as periodogram method
#method = {'method':'bls'}
#tpf.plot_pixels(periodogram=True, **method)

# plot periodograms for each pixel
tpf.plot_pixels(periodogram=True, minimum_period=0.01, maximum_period=1) # periods in days
plt.show()

# run Tess Localize on the most significant period to find its location
bestFreq = [1/bestPeriod]
low = tl.Localize(targetpixelfile=tpf, frequencies=bestFreq, frequnit=(1/u.day),
                  principal_components='auto')
low.plot()
plt.show()