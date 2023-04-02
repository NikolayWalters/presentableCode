"""
This is a random forest classifier test script that performs binary classification
of DA and non-DA white dwarf spectral classes based on SDSS optical spectra. It
conists of a combined random forest model. One model is trained on the DA/non-DA sample.
The second is trained on DA/A-type sample, since A-types proved difficult to distinguish.
The resulting model is then pickled
"""

from astropy.io import fits
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import optimize
from scipy.interpolate import CubicSpline
from scipy.integrate import simps
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter
import scipy.signal as signal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import classification_report

def fni(array, value):
    """
    A simple function that identifies the closest index in a list corresponding to
    a given value

    Example usage:
    >>> list1 = [5, 10, 15]
    >>> fni(list1, 9)
    Output: 1
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def prepSpectra(line):
    """
    Thius function accesses .fits files, retrieves spectral wavelength and flux arrays
    then crops the spectra to be between 3838.8-8914.6 AA and applies a Gaussian filter
    to identify any flux outliers in the data that are then replaced by the smoothed
    Gaussian filter value. The resulting spectrum is then standardised.
    """
    line = line.strip('\n')
    filepath = path + line
    hdul = fits.open(filepath)
    typeEx = hdul[0].data
    dataF = typeEx[0,:]
    typeWave = hdul[0].data
    dataWave = typeWave[2,:]
    hdul.close()
    start = fni(dataWave, 3838.839)
    finish = fni(dataWave, 8914.597)
    tempData = dataF[start:finish]
    GaussData = gaussian_filter(tempData, sigma=1)
    GaussMax = np.max(GaussData)
    GaussMin = np.min(GaussData)
    for n, i in enumerate(tempData):
        if i > GaussMax:
            tempData[n] = GaussMax
        if i < GaussMin:
            tempData[n] = GaussMin
    tempData = (np.array(tempData) - np.mean(tempData))/np.std(tempData)
    return tempData, dataWave


# DA Class training sample
path = '/home/nwal/Desktop/Jin/DAWD_fits/'
count = 0
names = []
with open(path+'DATar') as file:
    for line in file:
        names.append(line)
        tempData, dataWave = prepSpectra(line)
        if count == 0:
            minWave = np.min(dataWave)
            maxWave = np.max(dataWave)
            dataStackWD = tempData
        else:
            dataStackWD = np.vstack((dataStackWD, tempData))
            if np.min(dataWave) > minWave:
                minWave = np.min(dataWave)
            if np.max(dataWave) < maxWave:
                maxWave = np.max(dataWave)
        count = 1

WDTars = np.ones(np.shape(dataStackWD)[0])

# non-DA training sample
path = '/home/nwal/Desktop/Jin/others_fits/'
count = 0
names2 = []
with open(path+'AAA') as file:
    for line in file:
        names2.append(line)
        tempData, dataWave = prepSpectra(line)
        if count == 0:
            dataStackN = tempData
            minWave = np.min(dataWave)
            maxWave = np.max(dataWave)
        else:
            dataStackN = np.vstack((dataStackN, tempData))
            if np.min(dataWave) > minWave:
                minWave = np.min(dataWave)
            if np.max(dataWave) < maxWave:
                maxWave = np.max(dataWave)
        count = 1
NoTars = np.zeros(np.shape(dataStackN)[0])
namesTr = names + names2

# stacking training data
allData = np.vstack((dataStackWD, dataStackN))
allLabels = np.concatenate((WDTars, NoTars))
names = namesTr

# training first general random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(allData, allLabels)

# A type stars for training
path = '/home/nwal/Desktop/fits/'
count = 0
names2 = []
with open(path+'ATypeNames') as file:
    for line in file:
        names2.append(line)
        tempData, dataWave = prepSpectra(line)
        if count == 0:
            dataStackN = tempData
            minWave = np.min(dataWave)
            maxWave = np.max(dataWave)
        else:
            dataStackN = np.vstack((dataStackN, tempData))
            if np.min(dataWave) > minWave:
                minWave = np.min(dataWave)
            if np.max(dataWave) < maxWave:
                maxWave = np.max(dataWave)
        count = 1
NoTars = np.zeros(np.shape(dataStackN)[0])
namesTr = names + names2

# stacking training data for the second classifier
allData = np.vstack((dataStackWD, dataStackN))
allLabels = np.concatenate((WDTars, NoTars))
names = namesTr

# training second classifier
clf2 = RandomForestClassifier(n_estimators=50, random_state=42)
clf2.fit(allData, allLabels)

# combine forests
clf.estimators_ += clf2.estimators_
clf.n_estimators = len(clf.estimators_)

# serialization
pickle.dump(clf_comb, open( "clf_comb.p", "wb" ) )