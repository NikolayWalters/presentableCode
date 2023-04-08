"""
Determines T0 ephemeride by finding a minimum flux region based on a re-binning of 150 points
"""

import numpy as np
 
# Load the data from the file
data = np.loadtxt('lp705_s30.dat')

# Extract the time, flux, and flux error from the data
time = data[:, 0]
flux = data[:, 1]

# Define the period of the variability
period = 0.0504373

# Calculate the phase of each data point (NB the first time value corresponds to phase = 0)
phases = ((time - time[0]) / period) % 1

# Sort to be phase-consecutive
sortIndi = np.argsort(phases)
phases = phases[sortIndi]
flux = flux[sortIndi]

# Re-bin to find minimum average flux value
n_bins = 150
flux = np.nanmean(np.pad(flux.astype(float), (0, n_bins - flux.size%n_bins), mode='constant', constant_values=np.NaN).reshape(-1, n_bins), axis=1)
phases = np.nanmean(np.pad(phases.astype(float), (0, n_bins - phases.size%n_bins), mode='constant', constant_values=np.NaN).reshape(-1, n_bins), axis=1)

# Minimum flux value and corresponding phase
min_flux_idx = np.argmin(flux)
min_phase = phases[min_flux_idx]

# Corresponding minimum time
min_light_T = time[0] + min_phase * period
print("T0 = %.5f (d)" %min_light_T)

# Optional figure of phase folded LC
#import matplotlib.pyplot as plt
#plt.figure(figsize=(20,10))
#plt.plot(phases, flux, marker='o',mfc='none', c='k',ls='')
#plt.plot([min_phase,min_phase], [min(flux),max(flux)])

