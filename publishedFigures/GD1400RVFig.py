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

rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times']

def sineModelFix(x,a,c,d):
    """
    Sine function with a fixed frequency
    """
    return a*np.sin(2*np.pi*x+c)+d

plt.figure(figsize=(12,10))
gs = gridspec.GridSpec(2, 1,height_ratios=[5., 1], wspace=0.04,hspace=0) 
ax1 = plt.subplot(gs[0,0])

ISAACPhase = np.load('keepSafe/ISAACPhasesUpdated.npy')
ISAACRV = np.load('keepSafe/ISAACRVUpdated.npy')
ISAACError = np.load('keepSafe/ISAACErrorUpdated.npy')

UVESPhase = np.load('keepSafe/UVESPhasesUpdated.npy')+0.5
UVESRV = np.load('keepSafe/UVESRV.npy')
UVESError = np.load('keepSafe/UVESError.npy')



plt.rc('xtick',c='k')
plt.rc('ytick',c='k')
ms = 6
width = 1
mod = Model(sineModelFix)
pars = mod.make_params(a=200,b=1,c=0,d=100)
result = mod.fit(ISAACRV,pars,x=ISAACPhase)
ax2 = plt.subplot(gs[1,0])
TDummy = np.linspace(0,1,1000)
ypointsModel = sineModelFix(TDummy,-result.params['a'].value,0,result.params['d'].value)
ax1.plot(TDummy,ypointsModel, c='r', ls='-',lw=2)
offset = (-0.05518+np.pi)/(2*np.pi)
ax1.errorbar(ISAACPhase-1+offset, ISAACRV, yerr=ISAACError, ls='',marker='o', c='r',capsize=0,ms=ms)
ax1.errorbar(ISAACPhase+offset, ISAACRV, yerr=ISAACError, ls='',marker='o', c='r',capsize=0,ms=ms)
ax1.errorbar(ISAACPhase+1+offset, ISAACRV, yerr=ISAACError, ls='',marker='o', c='r',capsize=0,ms=ms)
ax2.plot([0,1],[0,0], 'k--')
capsize = 6
elinewidth = 2.
capthick = 2.
residualsBD = sineModelFix(ISAACPhase+offset,-result.params['a'].value,0,result.params['d'].value)
ax2.errorbar(ISAACPhase+offset, ISAACRV-residualsBD, yerr=ISAACError, ls='', marker='', c='r',capsize=capsize,elinewidth=elinewidth,capthick=capthick)
residualsBD = sineModelFix(ISAACPhase+offset+1,-result.params['a'].value,0,result.params['d'].value)
ax2.errorbar(ISAACPhase+offset+1, ISAACRV-residualsBD, yerr=ISAACError, ls='', marker='', c='r',capsize=capsize,elinewidth=elinewidth,capthick=capthick)
residualsBD = sineModelFix(ISAACPhase+offset-1,-result.params['a'].value,0,result.params['d'].value)
ax2.errorbar(ISAACPhase+offset-1, ISAACRV-residualsBD, yerr=ISAACError, ls='', marker='', c='r',capsize=capsize,elinewidth=elinewidth,capthick=capthick)


fontsize=22
ax2.set_xlabel('Phase', fontsize=fontsize)
ax1.set_ylabel('Radial Velocity (km s$^{-1}$)', fontsize=fontsize)
mod = Model(sineModelFix)
pars = mod.make_params(a=25,b=2,c=4.5,d=50)
result = mod.fit(UVESRV,pars,x=UVESPhase)
TDummy = np.linspace(0,1,1000)
ypointsModel = sineModelFix(TDummy,result.params['a'].value,0,result.params['d'].value)
offset = (3.07+np.pi)/(2*np.pi)
ax1.errorbar(UVESPhase-2+offset, UVESRV, yerr=UVESError, ls='', marker='o', c='b',capsize=0,ms=ms)
ax1.errorbar(UVESPhase-1+offset, UVESRV, yerr=UVESError, ls='', marker='o', c='b',capsize=0,ms=ms)
ax1.errorbar(UVESPhase+offset, UVESRV, yerr=UVESError, ls='', marker='o', c='b',capsize=10,ms=ms)
ax1.plot(TDummy,ypointsModel, c='b', ls='-',lw=2)

ax1.set_xlim(0,1)
ylim1=199
ylim2=21.73
ax1.set_ylim(-ylim1-(ylim1*0.15)+15.97, ylim1+(ylim1*0.125)+15.97)

ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.tick_params(axis='y',which='both', width=width,direction="in",pad=10)
ax1.tick_params(axis='y',which='major', length=14,direction="in")
ax1.tick_params(axis='x',which='both', width=width,direction='in')
ax1.tick_params(axis='x',which='major', length=14,direction='in',pad=10)
ax1.xaxis.set_minor_locator(MultipleLocator(0.05))
ax1.xaxis.set_major_locator(MultipleLocator(0.2))
ax1.yaxis.set_major_locator(MultipleLocator(100))
ax1.yaxis.set_minor_locator(MultipleLocator(25))
ax1.tick_params(axis='x',which='minor', length=8)
ax1.tick_params(axis='y',which='minor', length=8,direction='in')
ax1.tick_params(labelsize=fontsize)



ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.tick_params(axis='y',which='both', width=width,direction="in",pad=10)
ax2.tick_params(axis='y',which='major', length=14,direction="in")
ax2.tick_params(axis='x',which='both', width=width,direction='in')
ax2.tick_params(axis='x',which='major', length=14,direction='in',pad=10)
ax2.xaxis.set_minor_locator(MultipleLocator(0.05))
ax2.xaxis.set_major_locator(MultipleLocator(0.2))
ax2.yaxis.set_major_locator(MultipleLocator(10))
ax2.yaxis.set_minor_locator(MultipleLocator(5))
ax2.tick_params(axis='x',which='minor', length=8)
ax2.tick_params(axis='y',which='minor', length=8,direction='in')
ax2.tick_params(labelsize=fontsize)
ax1.set_xticklabels([])


ax1.xaxis.set_ticks_position('both')
ax2.xaxis.set_ticks_position('both')


residualsBD = sineModelFix(UVESPhase+offset,result.params['a'].value,0,result.params['d'].value)
ax2.errorbar(UVESPhase+offset, UVESRV-residualsBD, yerr=UVESError, ls='', marker='', c='b',capsize=capsize,elinewidth=elinewidth,capthick=capthick)
residualsBD = sineModelFix(UVESPhase+offset-2,result.params['a'].value,0,result.params['d'].value)
ax2.errorbar(UVESPhase+offset-2, UVESRV-residualsBD, yerr=UVESError, ls='', marker='', c='b',capsize=capsize, elinewidth=elinewidth,capthick=capthick)
residualsBD = sineModelFix(UVESPhase+offset-1,result.params['a'].value,0,result.params['d'].value)
ax2.errorbar(UVESPhase+offset-1, UVESRV-residualsBD, yerr=UVESError, ls='', marker='', c='b',capsize=capsize,elinewidth=elinewidth,capthick=capthick)
ax2.set_xlim(0,1)
ax1.yaxis.set_ticks_position('both')
ax2.yaxis.set_ticks_position('both')
bboxx = transforms.Bbox([[0.1, 0.35], [11.05, 8.85]])
plt.savefig('GD1400RVFigurePaper.pdf', bbox_inches=bboxx)