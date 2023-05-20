"""
Script to retrieve PDCSAP 120s data and perform simple periodicity analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from astropy.timeseries import LombScargle

def foldMeth(timeArray, period):
    """
    method to convert time to relative phase for a specified period
    """
    phases = []
    for el in timeArray:
        passed = int(el/period)
        leftover = el - (passed*period)
        phase = leftover/period
        phases.append(phase)
    return np.array(phases)

# read in coordinates to check
dahe_df = pd.read_csv('DAHe.csv')

# main loop
for el in dahe_df['radec']:
    # find object
    search_result = lk.search_lightcurvefile(el, author='SPOC',exptime=120)
    # check how many sectors are present
    if len(search_result)>1:
        collection = search_result.download_all()
        data = collection.PDCSAP_FLUX.stitch()
        data = data.remove_nans()
    elif len(search_result) == 1:
        data = search_result[0].download()
        data = data.PDCSAP_FLUX.remove_nans()
    else:
        # if none continue the main loop
        print('No suitable data')
        continue

    # get flux and time arrays
    flux = data.flux.value
    time = data.time.value

    # calculate LS
    freq = np.linspace(0.01, 150, 10000)
    power = LombScargle(time, flux).power(freq)
    plt.figure(figsize=(20,10))
    plt.plot(freq,power)
    plt.xlabel('Frequency, 1/d')
    plt.title(el)
    plt.show()
    bestPeriod = 24/freq[np.argmax(power)]

    # phase fold data on the best period
    plt.figure(figsize=(20,10))
    phases = foldMeth(time, bestPeriod)
    sortIndi = np.argsort(phases)
    phases = phases[sortIndi]
    flux = flux[sortIndi]
    flux = np.nanmean(np.pad(flux.astype(float), (0, 200 - flux.size%200), mode='constant', constant_values=np.NaN).reshape(-1, 200), axis=1)
    phases = np.nanmean(np.pad(phases.astype(float), (0, 200 - phases.size%200), mode='constant', constant_values=np.NaN).reshape(-1, 200), axis=1)
    plt.plot(phases, flux, marker='o',mfc='none', c='k',ls='')
    plt.title('Folded on '+ str(np.round(bestPeriod,decimals=2)) +' days')
    plt.xlabel('Phase')
    print('==============================================')
    print('==============================================')
    print('==============================================')
    print('==============================================')
    print('==============================================')
    
