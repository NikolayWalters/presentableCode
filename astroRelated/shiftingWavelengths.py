"""
Script to apply wavelength shifts to spectra based on the RV model
that also estimates SNR and uses it as a weight
"""

import numpy as np
from astropy.io import fits
from specutils import Spectrum1D
import pandas as pd

# Get spectra flux, wavelength and obs date
fluxes = []
wavels = []
dates2 = []
snrArray = []
with open(path+'files') as file:
    for line in file:
        line = line.strip('\n')
        line = path+line
        wvl, flx = pyasl.read1dFitsSpec(line)
        wvl = wvl*10 # NB XShooter is in nm so convert to Ang
        spectrum = Spectrum1D(spectral_axis=wvl* u.AA, flux=flx* u.Jy)
        der = snr_derived(spectrum) # estimate SNR for weighted co-add
        snrArray.append(der)
        flx = flx/np.median(flx)
        fluxes.append(flx)
        wavels.append(np.log10(wvl))
        hdul = fits.open(line)
        dates2.append(hdul[0].header['MJD-OBS'])
        hdul.close()
wavelsAng = np.array(wavels)

# calculating wavelength shifts
df = pd.read_csv('NLTTShifts.dat')
barycentricCor = df['bary'].values
modelVel = df['model'].values
model = modelVel - barycentricCor # barrycentric correction
shift = modelVel*5892.5/(299792458/1000) # for Na I line

# adjusting wavelengths with the shift
adjWvl = []
count = 0
for el in wavels:
    newW = 10**el-shift[count]
    adjWvl.append(newW)
    count = count + 1
count = 0

# write out adjusted spectra
for el in adjWvl:
    nameFile = str(count)+'ShiftedNLTT_Na.fits'
    pyasl.write1dFitsSpec(nameFile, fluxes[count]*snrArray[count].value, wvl=el, clobber=True)
    count = count +1