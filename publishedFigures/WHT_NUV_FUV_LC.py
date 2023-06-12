from astropy.io import fits
import os
from sklearn import preprocessing
import shutil
from astropy.timeseries import LombScargle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import optimize
from scipy.interpolate import CubicSpline
from scipy.integrate import simps
from lmfit import Parameters, minimize, report_fit, Model,fit_report
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter
import scipy.signal as signal
from astropy.time import Time
import pandas as pd
from lmfit import Model
from PyAstronomy.pyasl import foldAt
from scipy.optimize import curve_fit
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from scipy.ndimage.filters import uniform_filter
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import lightkurve as lk
from pylab import rc
from matplotlib import gridspec
from matplotlib import transforms
from matplotlib import colors
from scipy.stats import gaussian_kde
from pylab import *
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times']


def foldMeth(timeArray, period):
    phases = []
    for el in timeArray:
        passed = int(el/period)
        leftover = el - (passed*period)
        phase = leftover/period
        phases.append(phase)
    return np.array(phases)

def sineFun(x,amp,phase,off):
    return amp * np.sin(6.283185307*x + phase)+off

indata = np.loadtxt("GD356Uband.dat", usecols=(0,1))
time = indata[:,0]
flux = indata[:,1]
plt.plot(time,flux,'kx')
timeWHTU = time
fluxWHTU = flux

indata = np.loadtxt("HSTPhotFUVAnother.dat", usecols=(0,1))
time = indata[:,0] +2457000
flux = indata[:,1]
plt.plot(time,flux,'kx')
plt.show()
phasesFold = foldMeth(time,0.08029842621026712)
sortInd = np.argsort(phasesFold)
phasesFoldSorted = phasesFold[sortInd]
FluxSorted = flux[sortInd]
plt.plot(phasesFoldSorted, FluxSorted, 'kx') #needs 5 points binned
timeHSTFUV = time
fluxHSTFUV = flux


indata = np.loadtxt("HSTPhotNUV.dat", usecols=(0,1))
time = indata[:,0]
flux = indata[:,1]
plt.plot(time,flux,'kx')
plt.show()
phasesFold = foldMeth(time,0.08029842621026712)
sortInd = np.argsort(phasesFold)
phasesFoldSorted = phasesFold[sortInd]
FluxSorted = flux[sortInd]
plt.plot(phasesFoldSorted, FluxSorted, 'kx') #needs 5 points binned as well
timeHSTNUV = time
fluxHSTNUV = flux






from pylab import rc
rc('axes', linewidth=2)
fig, axs = plt.subplots(3, 1,
                        gridspec_kw={'hspace': 0., 'wspace': 0}, figsize=(20,30))
timeWHTUAdj = timeWHTU-2457000.- 1296.0255697684468
WHTPhases = foldMeth(timeWHTUAdj,  0.08029820145279275)
sortIndi = np.argsort(WHTPhases)
WHTPhases = WHTPhases[sortIndi]
fluxWHTUS = fluxWHTU[sortIndi]

meanPoints = 5
fluxWHTUS = np.nanmean(np.pad(fluxWHTUS.astype(float), (0, meanPoints - fluxWHTUS.size%meanPoints), mode='constant', constant_values=np.NaN).reshape(-1, meanPoints), axis=1)
WHTPhasesMean = np.nanmean(np.pad(WHTPhases.astype(float), (0, meanPoints - WHTPhases.size%meanPoints), mode='constant', constant_values=np.NaN).reshape(-1, meanPoints), axis=1)
axs[0].plot(WHTPhasesMean, fluxWHTUS-1, c='g',marker='o', ls = '')
axs[0].plot(WHTPhasesMean+1, fluxWHTUS-1, c='g', marker='o', ls = '')
axs[0].plot(WHTPhasesMean, sineFun(WHTPhasesMean,0.01569, 1.828,0), c='k',lw=4)
axs[0].plot(WHTPhasesMean+1, sineFun(WHTPhasesMean,0.01569, 1.828,0), c='k',lw=4)


timeHSTNUVAdj = timeHSTNUV-2457000- 1296.0255697684468
PhaseHSTNUV = foldMeth(timeHSTNUVAdj,  0.08029820145279275)
sortIndi = np.argsort(PhaseHSTNUV)
PhaseHSTNUV = PhaseHSTNUV[sortIndi]
fluxHSTNUVAdj = fluxHSTNUV[sortIndi]

meanPoints = 5
fluxHSTNUVAdj = np.nanmean(np.pad(fluxHSTNUVAdj.astype(float), (0, meanPoints - fluxHSTNUVAdj.size%meanPoints), mode='constant', constant_values=np.NaN).reshape(-1, meanPoints), axis=1)
PhaseHSTNUV = np.nanmean(np.pad(PhaseHSTNUV.astype(float), (0, meanPoints - PhaseHSTNUV.size%meanPoints), mode='constant', constant_values=np.NaN).reshape(-1, meanPoints), axis=1)
axs[1].plot(PhaseHSTNUV, fluxHSTNUVAdj, c='b',marker='o', ls = '')
axs[1].plot(PhaseHSTNUV+1, fluxHSTNUVAdj, c='b', marker='o', ls = '')
axs[1].plot(PhaseHSTNUV, sineFun(PhaseHSTNUV,0.02803, 1.566,0), c='k',lw=4)
axs[1].plot(PhaseHSTNUV+1, sineFun(PhaseHSTNUV,0.02803, 1.566,0), c='k',lw=4)



timeHSTFUVAdj = np.array(timeHSTFUV)-2457000 - 1296.0255697684468
phaseHSTFUV = foldMeth(timeHSTFUVAdj,  0.08029820145279275)
sortIndi = np.argsort(phaseHSTFUV)
phaseHSTFUV = phaseHSTFUV[sortIndi]
fluxHSTFUVAdj = fluxHSTFUV[sortIndi]
meanPoints = 5
fluxHSTFUVAdj = np.nanmean(np.pad(fluxHSTFUVAdj.astype(float), (0, meanPoints - fluxHSTFUVAdj.size%meanPoints), mode='constant', constant_values=np.NaN).reshape(-1, meanPoints), axis=1)
PhaseHSTFUV = np.nanmean(np.pad(phaseHSTFUV.astype(float), (0, meanPoints - phaseHSTFUV.size%meanPoints), mode='constant', constant_values=np.NaN).reshape(-1, meanPoints), axis=1)
axs[2].plot(PhaseHSTFUV, fluxHSTFUVAdj, c='magenta',marker='o', ls = '')
axs[2].plot(PhaseHSTFUV+1, fluxHSTFUVAdj, c='magenta', marker='o', ls = '')
axs[2].plot(PhaseHSTFUV, sineFun(PhaseHSTFUV,0.09397, 1.552,0), c='k',lw=4)
axs[2].plot(PhaseHSTFUV+1, sineFun(PhaseHSTFUV,0.09397, 1.552,0), c='k',lw=4)



xlimits = axs[0].set_xlim()
axs[0].set_xlim(xlimits)
axs[2].set_xlabel("Phase", fontsize=36)
axs[0].set_ylabel('Relative u-band Flux', fontsize=36)
axs[1].set_ylabel('Relative NUV Flux', fontsize=36)
axs[2].set_ylabel('Relative FUV Flux', fontsize=36)
axs[0].set_ylim(-0.045,0.045)
axs[0].set_xlim(-0.05,2.05)
axs[1].set_xlim(-0.05,2.05)
axs[2].set_xlim(-0.05,2.05)
axs[1].set_ylim(-0.07,0.07)
axs[2].set_ylim(-0.35,0.35)
axs[0].xaxis.set_minor_locator(AutoMinorLocator())
axs[1].xaxis.set_minor_locator(AutoMinorLocator())
axs[2].yaxis.set_major_locator(MultipleLocator(0.1))
axs[0].xaxis.set_minor_locator(MultipleLocator(0.125))
axs[0].yaxis.set_minor_locator(MultipleLocator(0.005))
axs[1].xaxis.set_minor_locator(MultipleLocator(0.125))
axs[1].yaxis.set_minor_locator(MultipleLocator(0.25))
axs[2].xaxis.set_minor_locator(MultipleLocator(0.125))
axs[2].yaxis.set_minor_locator(MultipleLocator(0.05))
axs[0].tick_params(which='minor', length=8,direction="in")
axs[1].tick_params(which='minor', length=8,direction="in")
axs[2].tick_params(which='minor', length=8,direction="in")
axs[0].tick_params(which='both', width=2,direction='in',pad=10)
axs[1].tick_params(which='both', width=2,direction='in',pad=10)
axs[2].tick_params(which='both', width=2,direction='in',pad=10)
axs[0].tick_params(which='major', length=14,direction='in')
axs[1].tick_params(which='major', length=14,direction='in')
axs[2].tick_params(which='major', length=14,direction='in')
axs[0].set_xticklabels([])
axs[1].set_xticklabels([])
axs[0].tick_params(labelsize=38)
axs[1].tick_params(labelsize=38)
axs[2].tick_params(labelsize=38)
axs[0].xaxis.set_ticks_position('both')
axs[1].xaxis.set_ticks_position('both')
axs[2].xaxis.set_ticks_position('both')
axs[0].yaxis.set_ticks_position('both')
axs[1].yaxis.set_ticks_position('both')
axs[2].yaxis.set_ticks_position('both')
bboxx = transforms.Bbox([[0.18, 2.64], [18.22, 26.44]])
plt.savefig('GD356ThesisLCs.pdf',bbox_inches=bboxx) 
plt.show()