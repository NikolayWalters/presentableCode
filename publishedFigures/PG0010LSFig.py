import numpy as np
import matplotlib.pyplot as plt
from PyAstronomy import pyasl
from matplotlib import gridspec
from pylab import rc
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib import transforms
from pylab import *
from lmfit import Parameters, minimize, report_fit, Model,fit_report
from gatspy import periodic
import pandas as pd
import string
import matplotlib.ticker as mticker

data1 = pd.read_csv('pg10ztf/PG10G.csv', delimiter=',')
data1 = data1[data1["catflags"] < 1]
timeZTF1 = data1['mjd']
fluxZTF1 = data1['mag']
errZTF1 = data1['magerr']
tZTFG = timeZTF1
magsZTFG = fluxZTF1
dyZTFG = errZTF1
filtZTFG = np.take(list('G'), np.arange(len(dyZTFG)), mode='wrap')
data1 = pd.read_csv('pg10ztf/PG10R.csv', delimiter=',')
data1 = data1[data1["catflags"] < 1]
timeZTF1 = data1['mjd']
fluxZTF1 = data1['mag']
errZTF1 = data1['magerr']
tZTFR = timeZTF1
magsZTFR = fluxZTF1
dyZTFR = errZTF1
filtZTFR = np.take(list('R'), np.arange(len(dyZTFR)), mode='wrap')
data1 = pd.read_csv('pg10ztf/PG10I.csv', delimiter=',')
data1 = data1[data1["catflags"] < 1]
timeZTF1 = data1['mjd']
fluxZTF1 = data1['mag']
errZTF1 = data1['magerr']
tZTFI = timeZTF1
magsZTFI = fluxZTF1
dyZTFI = errZTF1
filtZTFI = np.take(list('I'), np.arange(len(dyZTFI)), mode='wrap')


data1 = pd.read_csv('/home/nwal/Desktop/PG0010/Gaia EDR3 2859951106737135488.csv', delimiter=',')
timeZTF1 = data1['mjd']
fluxZTF1 = data1['flux_W1']
errZTF1 = data1['flux_error_W1']
tWISE1 = timeZTF1
magsWISE1 = fluxZTF1
dyWISE1 = errZTF1
filtWISE1 = np.take(list('WISE1'), np.arange(len(dyWISE1)), mode='wrap')

data1 = pd.read_csv('/home/nwal/Desktop/PG0010/Gaia EDR3 2859951106737135488.csv', delimiter=',')
nan_value = float("NaN") 
data1.replace("", nan_value, inplace=True)
data1.dropna(subset = ["flux_W2"], inplace=True)
timeZTF1 = data1['mjd']
fluxZTF1 = data1['flux_W2']
errZTF1 = data1['flux_error_W2']
tWISE2 = timeZTF1
magsWISE2 = fluxZTF1
dyWISE2 = errZTF1
filtWISE2 = np.take(list('WISE2'), np.arange(len(dyWISE2)), mode='wrap')


search_result = lk.search_lightcurvefile('WD 0010+280')
sec3O = search_result[0].download()
sec3 = sec3O.PDCSAP_FLUX.remove_nans()
fluxTESS = sec3.flux
timeTESS = sec3.time
errTESS = sec3.flux_err
filtTESS = np.take(list('T'), np.arange(len(timeTESS)), mode='wrap')


timeF = np.concatenate((tZTFG,tZTFR,tZTFI,timeTESS.mjd,tWISE1,tWISE2))
magsF = np.concatenate((magsZTFG,magsZTFR,magsZTFI,fluxTESS.to_value(),magsWISE1,magsWISE2))
dyF = np.concatenate((dyZTFG,dyZTFR,dyZTFI,errTESS.to_value(),dyWISE1,dyWISE2))
filF = np.concatenate((filtZTFG,filtZTFR,filtZTFI,filtTESS,filtWISE1,filtWISE2))
freq = np.linspace(0.001, 25, 10000)
periods = 1/freq
LS_multi = periodic.LombScargleMultiband(Nterms_base=1, Nterms_band=0)
LS_multi.fit(timeF, magsF, dyF, filF)
P_multi = LS_multi.periodogram(periods)


keepSafeTime = timeF
keepSafeMag = magsF
keepSafeErr = dyF
keepSafePower = P_multi
keepSafeFreq = freq

plt.plot(1/periods, P_multi, lw=1, color='k')

freq = np.linspace(0.001, 25, 10000)
powerPl = LombScargle(tWISE1,magsWISE1).power(freq)
print(freq[np.argmax(powerPl)])
probabilities = [0.1, 0.05, 0.01]
ls = LombScargle(tWISE1, magsWISE1)
print('wise1 ', ls.false_alarm_level(probabilities, method='bootstrap')  )

powerWISE1 = powerPl
freq = np.linspace(0.001, 25, 10000)
powerPl = LombScargle(tWISE2,magsWISE2).power(freq)
print(freq[np.argmax(powerPl)])
ls = LombScargle(tWISE2, magsWISE2)
print('wise2 ', ls.false_alarm_level(probabilities, method='bootstrap')  )

powerWISE2 = powerPl

freq = np.linspace(0.001, 25, 10000)
powerTess = LombScargle(timeTESS.mjd,fluxTESS).power(freq)
ls = LombScargle(timeTESS.mjd,fluxTESS)
print('TESS ', ls.false_alarm_level(probabilities, method='bootstrap'))

data1 = pd.read_csv('pg10ztf/PG10G.csv', delimiter=',')
data1 = data1[data1["catflags"] < 1]
timeZTF1 = data1['mjd']
fluxZTF1 = data1['mag']
fluxZTF1Old = fluxZTF1
zeroMag = data1['magzp']
timeZTF1 = timeZTF1
fluxZTF1 = 3.631*(10**(-fluxZTF1/2.5))
fluxZTF1 = (fluxZTF1/np.mean(fluxZTF1))-1
freq = np.linspace(0.001, 25, 10000)
powerPl = LombScargle(timeZTF1,fluxZTF1).power(freq)
ls = LombScargle(timeZTF1,fluxZTF1)
print('ZTFG ', ls.false_alarm_level(probabilities, method='bootstrap'))

powerZTFG = powerPl

data1 = pd.read_csv('pg10ztf/PG10R.csv', delimiter=',')
data1 = data1[data1["catflags"] < 1]
timeZTF1 = data1['mjd']
fluxZTF1 = data1['mag']
fluxZTF1Old = fluxZTF1
zeroMag = data1['magzp']
timeZTF1 = timeZTF1
fluxZTF1 = 3.631*(10**(-fluxZTF1/2.5))
fluxZTF1 = (fluxZTF1/np.mean(fluxZTF1))-1
freq = np.linspace(0.001, 25, 10000)
powerPl = LombScargle(timeZTF1,fluxZTF1).power(freq)
ls = LombScargle(timeZTF1,fluxZTF1)
print('ZTFr ', ls.false_alarm_level(probabilities, method='bootstrap'))

powerZTFR = powerPl
data1 = pd.read_csv('pg10ztf/PG10I.csv', delimiter=',')
data1 = data1[data1["catflags"] < 1]
timeZTF1 = data1['mjd']
fluxZTF1 = data1['mag']
fluxZTF1Old = fluxZTF1
zeroMag = data1['magzp']
timeZTF1 = timeZTF1
fluxZTF1 = 3.631*(10**(-fluxZTF1/2.5))
fluxZTF1 = (fluxZTF1/np.mean(fluxZTF1))-1
freq = np.linspace(0.001, 25, 10000)
powerPl = LombScargle(timeZTF1,fluxZTF1).power(freq)
ls = LombScargle(timeZTF1,fluxZTF1)
print('ZTFi ', ls.false_alarm_level(probabilities, method='bootstrap'))

powerZTFI = powerPl


fig, axs = plt.subplots(7,1,figsize=(12,10))
# periodogram plot

plt.rcParams["axes.axisbelow"] = False

plt.subplots_adjust(hspace = 0)

axs = axs.flat

plt.rc('xtick',c='k')
rc('axes', linewidth=2)
plt.rc('ytick',c='k')
colour = 'gray'
fontsize = 22
axs[0].axes.yaxis.set_visible(False)
axs[1].axes.yaxis.set_visible(False)
axs[2].axes.yaxis.set_visible(False)
axs[3].set_yticklabels([])
axs[4].axes.yaxis.set_visible(False)
axs[5].axes.yaxis.set_visible(False)
axs[6].axes.yaxis.set_visible(False)
axs[0].xaxis.set_minor_locator(AutoMinorLocator())
axs[0].tick_params(axis='x',which='both',width=2,direction='in')
axs[0].tick_params(axis='x',which='major',length=14,direction='in',pad=10)
axs[0].tick_params(axis='x',which='minor',length=8)
axs[0].tick_params(labelsize=fontsize)
axs[0].xaxis.set_ticks_position('both')
axs[0].yaxis.set_ticks_position('both')
axs[0].set_xticklabels([])
axs[1].xaxis.set_minor_locator(AutoMinorLocator())
axs[1].tick_params(axis='x',which='both',width=2,direction='in')
axs[1].tick_params(axis='x',which='major',length=14,direction='in',pad=10)
axs[1].tick_params(axis='x',which='minor',length=8)
axs[1].tick_params(labelsize=fontsize)
axs[1].xaxis.set_ticks_position('both')
axs[1].yaxis.set_ticks_position('both')
axs[1].set_xticklabels([])
axs[2].xaxis.set_minor_locator(AutoMinorLocator())
axs[2].tick_params(axis='x',which='both',width=2,direction='in')
axs[2].tick_params(axis='x',which='major',length=14,direction='in',pad=10)
axs[2].tick_params(axis='x',which='minor',length=8)
axs[2].tick_params(labelsize=fontsize)
axs[2].xaxis.set_ticks_position('both')
axs[2].yaxis.set_ticks_position('both')
axs[2].set_xticklabels([])
axs[3].xaxis.set_minor_locator(AutoMinorLocator())
axs[3].tick_params(axis='x',which='both',width=2,direction='in')
axs[3].tick_params(axis='x',which='major',length=14,direction='in',pad=10)
axs[3].tick_params(axis='x',which='minor',length=8)
axs[3].tick_params(labelsize=fontsize)
axs[3].xaxis.set_ticks_position('both')
axs[3].yaxis.set_ticks_position('both')
axs[3].set_xticklabels([])
axs[4].xaxis.set_minor_locator(AutoMinorLocator())
axs[4].tick_params(axis='x',which='both',width=2,direction='in')
axs[4].tick_params(axis='x',which='major',length=14,direction='in',pad=10)
axs[4].tick_params(axis='x',which='minor',length=8)
axs[4].tick_params(labelsize=fontsize)
axs[4].xaxis.set_ticks_position('both')
axs[4].yaxis.set_ticks_position('both')
axs[4].set_xticklabels([])
axs[5].xaxis.set_minor_locator(AutoMinorLocator())
axs[5].tick_params(axis='x',which='both',width=2,direction='in')
axs[5].tick_params(axis='x',which='major',length=14,direction='in',pad=10)
axs[5].tick_params(axis='x',which='minor',length=8)
axs[5].tick_params(labelsize=fontsize)
axs[5].xaxis.set_ticks_position('both')
axs[5].yaxis.set_ticks_position('both')
axs[5].set_xticklabels([])
axs[6].xaxis.set_minor_locator(AutoMinorLocator())
axs[6].tick_params(axis='x',which='both',width=2,direction='in')
axs[6].tick_params(axis='x',which='major',length=14,direction='in',pad=10)
axs[6].tick_params(axis='x',which='minor',length=8)
axs[6].tick_params(labelsize=fontsize)
axs[6].xaxis.set_ticks_position('both')
axs[6].yaxis.set_ticks_position('both')
axs[0].set_ylim(0,0.0024772)
axs[1].set_ylim(0,0.09699059) 
axs[2].set_ylim(0,0.09733297)
axs[3].set_ylim(0,0.40304024)
axs[4].set_ylim(0,0.09741804)
axs[5].set_ylim(0,0.25263025)
axs[6].set_ylim(0,0.016588646899857087)


names = ['TESS', 'g', 'r', 'i','W1', 'W2', 'Multiband']
for n, ax in enumerate(axs):
    ax.text(0.98, 0.6, names[n], transform=ax.transAxes, size=20, ha='right')
axs[0].plot(freq, powerTess,lw=0.6, c=colour)
axs[0].set_xlim(0,25)
axs[1].plot(freq, powerZTFG,lw=0.6, c=colour)
axs[1].set_xlim(0,25)
axs[2].plot(freq, powerZTFR,lw=0.6, c=colour)
axs[2].set_xlim(0,25)
axs[3].plot(freq, powerZTFI,lw=0.6, c=colour)
axs[3].set_xlim(0,25)
axs[4].plot(freq, powerWISE1, lw=0.6, c=colour)
axs[4].set_xlim(0,25)
axs[5].plot(freq, powerWISE2,lw=0.6, c=colour)
axs[5].set_xlim(0,25)
axs[6].plot(1/periods, P_multi,lw=0.6, c=colour)
axs[6].set_xlim(0,25)
axs[6].set_xlabel('Frequency (1/d)', fontsize=fontsize)
axs[3].set_ylabel('Lomb Scargle Power', fontsize=fontsize)
axs[0].axes.get_yaxis().set_ticks([])
axs[1].axes.get_yaxis().set_ticks([])
axs[2].axes.get_yaxis().set_ticks([])
axs[3].axes.get_yaxis().set_ticks([])
axs[4].axes.get_yaxis().set_ticks([])
axs[5].axes.get_yaxis().set_ticks([])
axs[6].axes.get_yaxis().set_ticks([])


bboxx = transforms.Bbox([[1.1, 0.42], [11, 8.85]])
plt.savefig('PG0010PhotFigurePaper.pdf', bbox_inches=bboxx)