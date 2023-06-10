"""
Thesis figure showing a comparison between optical and 
Gaia BP/RP mean spectra for GD 356 and a physically similar 
DA white dwarf
"""

from gaiaxpy import calibrate
import os
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from astropy import units as u
from astropy_healpix import HEALPix
from gaiaxpy import convert
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib import transforms
from pylab import *
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times']
rc('axes', linewidth=2)

# read in two BP/RP spectra and calibrate it
f = '/home/nwal/Desktop/Random-Astro-master/GaiaBPRP/example.csv'
df = pd.read_csv(f)
calibrated_spectra, sampling = calibrate(df)
f = '/home/nwal/Desktop/Random-Astro-master/XP_CONTINUOUS-Gaia DR3 395234439752169344.csv'
df3 = pd.read_csv(f)
calibrated_spectra2, sampling2 = calibrate(df3)

# define a cubic for continiuum normalisation
def cubic(x,a,b,c,d):
    return a*x*x*x+b*x*x+c*x+d

# fit continiuum
params, _ = curve_fit(cubic, sampling,calibrated_spectra['flux'][0])#,bounds=bounds,p0=initial_guess)
a,b,c,d = params
best_fit_curve=cubic(sampling,a,b,c,d)

# plot optical spectrum of GD 356
fig, ax1 = plt.subplots(1, 1, figsize=(12,10))
df = pd.read_csv('WHTCoAddedSpec.csv')
ax1.plot(df['wav'],df['flux'],c='k', label='GD 356 WHT')
ax1.plot(sampling*10,calibrated_spectra['flux'][0]/best_fit_curve,c='b', label='GD 356 Gaia')

# read in second optical spectrum, fit and plot it
df2 = pd.read_csv('spect.GJ 1004.BRL.csv')
wavelength = df2['wavelength']
flux = df2['flux']
params, _ = curve_fit(cubic, wavelength,flux)
a,b,c,d = params
best_fit_curve=cubic(wavelength,a,b,c,d)
# includes a small offset for clarity
ax1.plot(wavelength[17:-25],(flux/best_fit_curve)[17:-25]+0.5,c='g',label='LHS 1038 Bergeron et al. (2001)')

# plot BP/RP spectrum
params, _ = curve_fit(cubic, sampling,calibrated_spectra2['flux'][0])
a,b,c,d = params
best_fit_curve=cubic(sampling2,a,b,c,d)
# includes a small offset for clarity
ax1.plot(sampling2*10,(calibrated_spectra2['flux'][0]/best_fit_curve)+0.5,c='r', label='LHS 1038 Gaia')

# plot parameters
ax1.set_xlim(4000,7999.999)
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.tick_params(axis='y',which='both', width=2,direction="in",pad=10)
ax1.tick_params(axis='y',which='major', length=14,direction="in")
ax1.tick_params(axis='x',which='both', width=2,direction='in')
ax1.tick_params(axis='x',which='major', length=14,direction='in',pad=10)
ax1.yaxis.set_major_locator(MultipleLocator(0.5))
ax1.yaxis.set_minor_locator(MultipleLocator(0.125))
ax1.tick_params(axis='x',which='minor', length=8)
ax1.tick_params(axis='y',which='minor', length=8,direction='in')
ax1.tick_params(labelsize=22)
ax1.xaxis.set_ticks_position('both')
ax1.yaxis.set_ticks_position('both')
ax1.set_xlabel('Wavelength (Å)', fontsize=22)
ax1.set_ylabel('Relative Flux', fontsize=22)
ax1.legend(prop={'size': 20}, loc='lower left', frameon=True)
bboxx = transforms.Bbox([[0.45, 0.4], [10.9, 8.9]])
#plt.savefig('GaiaSpectraThesis.pdf', bbox_inches=bboxx)