import numpy as np
import matplotlib.pyplot as plt
from PyAstronomy import pyasl
from matplotlib import gridspec
from pylab import rc
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, ScalarFormatter)
from matplotlib import transforms
from pylab import *
from lmfit import Parameters, minimize, report_fit, Model,fit_report
from astropy import units as u


plt.rc('xtick',c='k')
rc('axes', linewidth=2)
plt.rc('ytick',c='k')
fig, ax1 = plt.subplots(1, 2, figsize=(20,10))
plt.subplots_adjust(wspace=0.04)
flux=[4.34,2.56,2.25,1.56,1.08,0.83,0.46,0.30,0.19,0.145,0.088,0.089]
wavel=[0.23,0.36,0.47,0.62,0.75,0.89,1.25,1.63,2.20,3.55,4.49,5.73]
#ax1[0].plot(wavel, flux, 'bo')
ax1[0].set_yscale('log')
ax1[0].set_xscale('log')
ax1[0].set_xlim(0.1, 20)

ax1[0].plot([-99,-99],[-99,-99], 'k--')


ax1[0].set_ylim(0.010001,15)
#sdss
ax1[0].errorbar(0.23, 4.34, yerr=0.43, marker='', capsize=6,elinewidth=3,markeredgewidth=3, c='b')
ax1[0].errorbar(0.36, 2.55, yerr=0.13, marker='', capsize=6,elinewidth=3,markeredgewidth=3, c='b')
ax1[0].errorbar(0.47, 2.25, yerr=0.11, marker='', capsize=6,elinewidth=3,markeredgewidth=3, c='b')
ax1[0].errorbar(0.62, 1.56, yerr=0.08, marker='', capsize=6,elinewidth=3,markeredgewidth=3, c='b')
ax1[0].errorbar(0.75, 1.08, yerr=0.05, marker='', capsize=6,elinewidth=3,markeredgewidth=3, c='b')
ax1[0].errorbar(0.89, 0.83, yerr=0.04, marker='', capsize=6,elinewidth=3,markeredgewidth=3, c='b')

#ukirt
ax1[0].errorbar(1.25, 0.46, yerr=0.02, marker='', capsize=6,elinewidth=3,markeredgewidth=3, c='b')
ax1[0].errorbar(1.63, 0.3, yerr=0.02, marker='', capsize=6,elinewidth=3,markeredgewidth=3, c='b')
ax1[0].errorbar(2.20, 0.19, yerr=0.01, marker='', capsize=6,elinewidth=3,markeredgewidth=3, c='b')

#irac
ax1[0].errorbar(3.55, 0.145,   yerr=0.008,marker='', capsize=6,elinewidth=3,markeredgewidth=3, c='b')
ax1[0].errorbar(4.49, 0.088,   yerr=0.006,marker='', capsize=6,elinewidth=3,markeredgewidth=3, c='b')
ax1[0].errorbar(5.73, 0.089,   yerr=0.021,marker='', capsize=6,elinewidth=3,markeredgewidth=3, c='b')

upper_error =  [0,0]
lower_error=  [0.005,0.005]

asymmetric_error = np.array(list(zip(lower_error, upper_error))).T
ax1[0].errorbar([7.87,50], [0.036,10], yerr=asymmetric_error, linestyle='', marker='_', capsize=6,elinewidth=3,markeredgewidth=3, c='b')
ax1[0].plot(7.87, 0.036-lower_error[0]-0.0035, marker=7, ms=14,c='b')

wavelengthBD = [1.63, 2.2, 3.55, 4.49, 5.73, 7.87]
l1400=  np.array([15.15,14.88,13.92,13.94,13.58,13.42])
m = l1400 + 5.6 - 2.7
f0 = np.array([1020.0,640.0,280.9,179.7,115.0,64.1])*1e3
fbd   = np.array([61.4611,49.4516,52.5474,33.0026,29.4237,19.0046])/1000
wavelengthBD = [1.63, 2.2, 3.55, 4.49]
fwdbd = np.array([347.286,208.418,116.287,73.4927])/1000
ax1[0].plot(wavelengthBD, fwdbd, 'ro', ms=10)
ax1[0].plot([-99,-99],[-99,-99], 'ro', label='L5 brown dwarf', ms=15)
fwdbd2 = np.array([359.578,224.242,146.764,103.525])/1000
ax1[0].plot(wavelengthBD, fwdbd2, c='orange', marker='o', ls='',ms=10)
ax1[0].plot([-99,-99],[-99,-99], c='orange', marker='o', ls='',ms=15, label='L5 irradiated')

data = np.loadtxt('/home/nwal/Desktop/PG0010/IRTF/da27000_800.dat')

wave = data[:,0]
fnu = data[:,1]
fnu = fnu/max(fnu)
fnu = fnu*12.05

data = np.loadtxt('/home/nwal/Desktop/PG0010/IRTF/Default DatasetS.csv', delimiter=',')
wvl = data[:,0]
flux = data[:,1]

wave = np.concatenate((wave[:-10], wvl))
flux = np.concatenate((fnu[:-10], flux))
ax1[0].plot(wave, flux, 'k--')

flux=[4.34,2.56,2.25,1.56,1.08,0.83,0.46,0.30,0.19]
lambd=[0.23,0.36,0.47,0.62,0.75,0.89,1.25,1.63,2.20]

data = np.loadtxt('/home/nwal/Desktop/PG0010/IRTF/irtf.dat', delimiter=" ")
wvl = data[:,0]
flux = data[:,1]

ax1[1].plot(wvl, flux*1000, c='gray', marker='.', ls='', ms=8)
ax1[1].plot([-99,-99],[-99,-99], c='gray', marker='.', ls='', ms=30, label='SpeX')
data = np.loadtxt('/home/nwal/Desktop/PG0010/IRTF/da27000_800.dat')

wave = data[:,0]
fnu = data[:,1]
fnu = fnu/max(fnu)
fnu = fnu*12.05

ax1[1].plot(wave, fnu,'k--')
ax1[1].set_xlim(0.6,2.45)
ax1[1].set_ylim(0,2)

data = np.loadtxt("/home/nwal/Desktop/PG0010/NIRSPEC/shortened/tar1.dat", delimiter=",")
wave = data[:,0]
fnu = np.array(data[:,1])

data = np.loadtxt("/home/nwal/Desktop/PG0010/NIRSPEC/shortened/tar2.dat", delimiter=",")
fnu = fnu + np.array(data[:,1])

data = np.loadtxt("/home/nwal/Desktop/PG0010/NIRSPEC/shortened/tar3.dat", delimiter=",")
fnu = fnu + np.array(data[:,1])

data = np.loadtxt("/home/nwal/Desktop/PG0010/NIRSPEC/shortened/tar4.dat", delimiter=",")
fnu = fnu + np.array(data[:,1])

data = np.loadtxt("/home/nwal/Desktop/PG0010/NIRSPEC/shortened/tar5.dat", delimiter=",")
fnu = fnu + np.array(data[:,1])

data = np.loadtxt("/home/nwal/Desktop/PG0010/NIRSPEC/shortened/tar6.dat", delimiter=",")
fnu = fnu + np.array(data[:,1])

data = np.loadtxt("/home/nwal/Desktop/PG0010/NIRSPEC/shortened/tar7.dat", delimiter=",")
fnu = fnu + np.array(data[:,1])

data = np.loadtxt("/home/nwal/Desktop/PG0010/NIRSPEC/shortened/tar8.dat", delimiter=",")
fnu = fnu + np.array(data[:,1])
fnu = fnu / 8

flam = (fnu * u.Jy).to(u.photon / u.cm**2 / u.s / u.Hz,equivalencies=u.spectral_density(wave * u.micron)) 
ax1[1].set_ylim(0,0.9999)

flux=[4.34,2.56,2.25,1.56,1.08,0.83,0.46,0.30,0.19]
ax1[1].errorbar(0.89, 0.83, yerr=0.83*0.05,marker='', capsize=6,elinewidth=3,markeredgewidth=3, c='b', zorder=99)

ax1[1].errorbar(1.25, 0.46, yerr=0.46*0.05,marker='', capsize=6,elinewidth=3,markeredgewidth=3, c='b', zorder=99)

ax1[1].errorbar(1.63, 0.3, yerr=0.3*0.05,marker='', capsize=6,elinewidth=3,markeredgewidth=3, c='b', zorder=99)

ax1[1].errorbar(2.20, 0.19, yerr=0.19*0.05, marker='', capsize=6,elinewidth=3,markeredgewidth=3, c='b', zorder=99)
fontsize= 20

ax1[0].tick_params(axis='y',which='both', width=2,direction="in",pad=10)
ax1[0].tick_params(axis='y',which='major', length=14,direction="in")
ax1[0].tick_params(axis='x',which='both', width=2,direction='in')
ax1[0].tick_params(axis='x',which='major', length=14,direction='in',pad=10)
ax1[0].tick_params(axis='x',which='minor', length=8)
ax1[0].tick_params(axis='y',which='minor', length=8,direction='in')
ax1[0].tick_params(labelsize=fontsize)
ax1[0].xaxis.set_ticks_position('both')
ax1[0].yaxis.set_ticks_position('both')

ax1[0].set_xlabel('λ (μm)', fontsize=fontsize)
ax1[0].set_ylabel('F$_\\nu$ (mJy)', fontsize=fontsize)
ax1[0].xaxis.set_major_formatter(ScalarFormatter())
ax1[0].yaxis.set_major_formatter(ScalarFormatter())

ax1[1].tick_params(axis='y',which='both', width=2,direction="in",pad=10)
ax1[1].tick_params(axis='y',which='major', length=14,direction="in")
ax1[1].tick_params(axis='x',which='both', width=2,direction='in')
ax1[1].tick_params(axis='x',which='major', length=14,direction='in',pad=10)
ax1[1].tick_params(axis='x',which='minor', length=8)
ax1[1].tick_params(axis='y',which='minor', length=8,direction='in')
ax1[1].tick_params(labelsize=fontsize)
ax1[1].xaxis.set_ticks_position('both')
ax1[1].yaxis.set_ticks_position('both')

ax1[1].set_xlabel('λ (μm)', fontsize=fontsize)
ax1[1].xaxis.set_major_formatter(ScalarFormatter())
ax1[1].yaxis.tick_right()
ax1[1].yaxis.set_ticks_position('both')
ax1[0].legend(prop={'size': fontsize}, loc='upper right', frameon=False)

ax1[1].legend(prop={'size': fontsize}, loc='upper right', frameon=False)

bboxx = transforms.Bbox([[1.38, 0.3], [18.6, 8.84]])
plt.savefig('PG0010SEDFigure.pdf', bbox_inches=bboxx)