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
fontsize = 12
targetName = 'WD 1009-184'

def foldMeth(timeArray, period):
    phases = []
    for el in timeArray:
        passed = int(el/period)
        leftover = el - (passed*period)
        phase = leftover/period
        phases.append(phase)
    return np.array(phases)

search_result = lk.search_lightcurvefile(targetName, exptime=120)
collection = search_result.download_all()
data = collection.PDCSAP_FLUX.stitch()
data = data.remove_nans()
flux = data.flux.value
time = data.time.value
freq = np.linspace(0.001, 25, 10000)
power = LombScargle(time, flux).power(freq)
bestPeriod = 1/freq[np.argmax(power)]
plt.figure(figsize=(20,10))
plt.plot(freq,power, c='k')
plt.xlabel('Frequency [1/d]', fontsize=fontsize)
plt.ylabel('LS Power', fontsize=fontsize)
plt.title('Lomb-Scargle Periodogram', fontsize=fontsize)
plt.show()
plt.figure(figsize=(20,10))
phases = foldMeth(time, bestPeriod)
sortIndi = np.argsort(phases)
phases = phases[sortIndi]
flux = flux[sortIndi]
flux = np.nanmean(np.pad(flux.astype(float), (0, 200 - flux.size%200), mode='constant', constant_values=np.NaN).reshape(-1, 200), axis=1)
phases = np.nanmean(np.pad(phases.astype(float), (0, 200 - phases.size%200), mode='constant', constant_values=np.NaN).reshape(-1, 200), axis=1)
plt.plot(phases, flux, marker='o',mfc='none', c='k',ls='')
plt.plot(phases+1, flux, marker='o',mfc='none', c='k',ls='')
plt.title('Folded on '+ str(np.round(bestPeriod,decimals=2)) +' days', fontsize=fontsize)
plt.xlabel('Phase', fontsize=fontsize)
plt.ylabel('Normalised Flux', fontsize=fontsize)
plt.show()


string = search_result[0].mission[0]
pattern = r'\d+'
match = re.search(pattern, string)
sector = int(match.group())
tpf = lk.search_targetpixelfile(targetName, sector = sector).download ()
#method = {'method':'bls'}
#tpf.plot_pixels(periodogram=True, **method)
tpf.plot_pixels(periodogram=True, minimum_period=0.01, maximum_period=1)
plt.show()
bestFreq = [1/bestPeriod]

low = tl.Localize(targetpixelfile=tpf, frequencies=bestFreq, frequnit=(1/u.day),
                  principal_components='auto')
low.plot()
plt.show()