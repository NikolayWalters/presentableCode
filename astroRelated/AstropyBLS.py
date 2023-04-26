"""
Simple Astropy BLS implementation
"""

import numpy as np
import astropy.units as u
from astropy.timeseries import BoxLeastSquares
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# loading and plotting data
first = pd.read_csv('sdss1252gr_1.dat', delimiter="\s+", names=['time','flux','err'])
second = pd.read_csv('sdss1252gr_2.dat', delimiter="\s+", names=['time','flux','err'])
third = pd.read_csv('sdss1252gr_4.dat', delimiter="\s+", names=['time','flux','err'])
combined = pd.concat([first,second,third])
sns.scatterplot(x=combined['time'], y=combined['flux'])

# creating BLS model
model = BoxLeastSquares(combined['time'] * u.day, combined['flux'])
periods = np.linspace(0.001, 0.02, 100000) # period array
periodogram = model.power(periods, 0.001/2) # step needs to be smaller than the smallest period

# plot periodogram
plt.figure(figsize=(20,10))
plt.plot(1/periodogram.period, periodogram.power) # NB plotting as frequency
plt.xlabel('Frequency (1/d)', fontsize=26)
plt.legend(fontsize=26)
#plt.savefig('SDSS1252BLS.pdf')