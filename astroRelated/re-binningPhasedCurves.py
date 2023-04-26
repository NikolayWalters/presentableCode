"""
Script showing two ways to re-bin phased light curves.
First method simply re-bins N phase consecutive points.
Second method re-bins with a uniform bin distribution,
i.e. 1 point per equally sized bin
"""

import numpy as np

def foldMeth(timeArray, period):
    """
    Method to convert time to relative phase based on the supplied period
    """
    phases = []
    for el in timeArray:
        passed = int(el/period)
        leftover = el - (passed*period)
        phase = leftover/period
        phases.append(phase)
    return np.array(phases)

# First method
phases = foldMeth(allAllT, 0.0802984187660111)
sortIndi = np.argsort(phases)
phases = phases[sortIndi]
flux = allAllFlux[sortIndi]
points = 400 # number of points to be binned together
flux = np.nanmean(np.pad(flux.astype(float), (0, points - flux.size%points), mode='constant', constant_values=np.NaN).reshape(-1, points), axis=1)
phases = np.nanmean(np.pad(phases.astype(float), (0, points - phases.size%points), mode='constant', constant_values=np.NaN).reshape(-1, points), axis=1)

# Second method
phases = foldMeth(allAllT, 0.0802984187660111)
sortIndi = np.argsort(phases)
phases = phases[sortIndi]
flux = allAllFlux[sortIndi]
N_bins = 50
bin_width = 1/N_bins
rebinned_flux = []
for i in range(N_bins):
    bin_start = i*bin_width
    bin_end = bin_start+bin_width
    flux_in_bin = flux[(phases>=bin_start)&(phases<bin_end)]
    bin_flux = np.mean(flux_in_bin)
    rebinned_flux.append(bin_flux)
phases = np.arange(0,1,bin_width)
flux = rebinned_flux