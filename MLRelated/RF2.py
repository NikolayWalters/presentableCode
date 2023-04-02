"""
This is a random forest classifier test script that performs binary classification
of DA and non-DA white dwarf spectral classes based on SDSS optical spectra
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
    return tempData


# DA Class training sample
path = '/home/nwal/Desktop/Jin/DAWD_fits/' #<------------------ path to DAWDs with DATar filename containing spec names on line 25
count = 0
names = []
with open(path+'DATar') as file:
    for line in file:
        names.append(line)
        tempData = prepSpectra(line)
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
path = '/home/nwal/Desktop/Jin/others_fits/' #<------------------ path to ~680 non-DAs that you sent with the DAs with AAA filename containing spec names on line 66
count = 0
names2 = []
with open(path+'AAA') as file:
    for line in file:
        names2.append(line)
        tempData = prepSpectra(line)
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

# stacking the training set
allData = np.vstack((dataStackWD, dataStackN))
allLabels = np.concatenate((WDTars, NoTars))
names = namesTr

# training RF classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(allData, allLabels)

# unseen spectra classification
arrayDigits = np.linspace(1000, 50000, 50)    
jinfiles = arrayDigits    
globalMisclass = []
for k in range(len(jinfiles)):
    path = '/home/nwal/Desktop/fits/' #<------------------ path to 50k specs with smr1000, smr2000 etc being files containing batches of filenames (1000 in each)
    count = 0
    names2 = []
    pathFile = '/home/nwal/Desktop/Jin/smr'+str(int(jinfiles[k]))
    with open(pathFile) as file:
        for line in file:
            names2.append(line)
            tempData = prepSpectra(line)
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
    names = names2
    y_pred2 = clf.predict(dataStackN)
    misclas = []
    label = []
    j = 0
    for j in range(len(NoTars)):
        if NoTars[j] != y_pred2[j]:
            misclas.append(names[j])
            #label.append(NoTars[j])
            #pXMesh = np.linspace(3837.956, 8914.597, 3659) 
            #plt.title(names[j] + " is "+str(NoTars[j]))
            #plt.plot(pXMesh, dataStackN[j][:], 'k-')
            #plt.show()
            if len(globalMisclass) == 0:
                globalMisclass = misclas
            else:
                globalMisclass = np.concatenate((globalMisclass, misclas))
    out = '\n'.join('{} {}'.format(x,int(y)) for x,y in zip(misclas,label))
print(globalMisclass)