"""
Script for the mass loss figure from the Detection of a substellar wind using a white dwarf host paper
"""

import numpy as np
import matplotlib.pyplot as plt
from PyAstronomy import pyasl
from matplotlib import gridspec
from pylab import rc
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib import transforms
from pylab import *
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times']
plt.rc('xtick',c='k')
rc('axes', linewidth=2)
plt.rc('ytick',c='k')
fig, ax1 = plt.subplots(1, 2,gridspec_kw={'hspace': 0.15, 'wspace': 0.05},figsize=(20,10))
ax1[0].set_yscale('log')
ax1[1].set_yscale('log')
ax1[1].set_xscale('log')
xlabels=['C', 'Na', 'Mg', 'Al', 'Si', 'Ca', 'Fe']
xaxis = np.linspace(0,6,7)

# accretion rates for each element
ax1[0].plot([2,3,4,5,6], [6.93391736563295e-16,7.55584147497912e-16,1.27937786762841e-15,7.75238572340336e-16,8.66991450389853e-16],c='deeppink',marker='x',ls='-', ms=12) #0354
ax1[0].plot([2,3,4,5,6], [1.83660222656361e-16,2.19023988049728e-16,8.07543783465942e-16,1.16456629860081e-16,2.0405331735495e-16], 'ro', ls='-',ms=12)#0419
ax1[0].plot([0,4], [3.568460929959988e-18 , 2.674908054611724e-18],'b*' ,ls='-',ms=12)#2257
ax1[0].plot([0,4], [2.6944805525723e-18,3.13147074126614e-18], 'g^', ls='-',ms=12)#0710
ax1[0].plot([1], [3.296563385397655e-15], marker='s', c='purple', ls='', ms=12)
ax1[0].plot([0,4,6], [2.3883159495181517e-17,1.0966390114430455e-17,1.0158268145766127e-17], marker='p',c='orange',ls='-', ms=12)
ax1[0].plot([2, 3, 4, 5, 6], [6.8283117963786484e-15, 3.553182543118292e-15, 7.549418752091741e-15,
                          7.585252725485695e-15,1.1726529404795977e-14], 'cP', ls='-', ms=12)
# NLTT upper limit point
x = np.array([5])
y = np.array([1.917388939919167e-18])
y_el = np.array([1e-18])
y_eu = np.array([0])
mask = (y != y_el)
ax1[0].errorbar(x, y, yerr=[y_el, y_eu],
            c='purple', capsize=6.25, elinewidth=1, marker='s',
             linestyle='',uplims=mask, ms=12)

# label points
ax1[1].plot([0], [100], 'ro',ls='', label='LHS 1660', ms=20)#0419
ax1[1].plot([0],[100],'b*' ,label='PG 2257+162', ms=20)#2257
ax1[1].plot([0],[100], marker='p', c='orange', label='BPM 6502', ls='', ms=20)
ax1[1].plot([0], [100], 'cP', label='LTT 560', ls='', ms=20)
ax1[1].plot([0], [100],c='deeppink',marker='x',ls='',label='Rubin 80', ms=20) #0354
ax1[1].plot([0],[100], 'g^', label='GD 448',ms=20)#0710
ax1[1].plot([0],[100],marker='s', c='purple', label='NLTT 5306', ls='', ms=20)
ax1[1].plot(0,100, c='saddlebrown',marker='v',label='Case 1',ls='', ms=20)
ax1[1].plot(0, 100, c='yellowgreen', marker='d', ls='', label='PG 1026+002', ms=20)

# adjusting labels and ticks
ax1[0].set_ylim(5e-19,2.5e-14)
ax1[0].set_xticks(xaxis)
ax1[0].set_xticklabels(xlabels)
ax1[0].set_xlabel('Element', fontsize=26)
ax1[0].set_ylabel('$\,\dot{M}_1\,(M_\odot$yr$^{-1}$)', fontsize=26)
ax1[0].tick_params(labelsize=14)
ax1[0].tick_params(axis='y',which='both', width=2,direction="in",pad=10)
ax1[0].tick_params(axis='y',which='major', length=14,direction="in")
ax1[0].tick_params(axis='x',which='both', width=2,direction='in')
ax1[0].tick_params(axis='x',which='major', length=14,direction='in',pad=10)
ax1[0].tick_params(axis='x',which='minor', length=8)
ax1[0].tick_params(axis='y',which='minor', length=8,direction='in')
ax1[0].tick_params(labelsize=26)
ax1[0].xaxis.set_ticks_position('both')
ax1[0].yaxis.set_ticks_position('both')


# second sub-plot with mass loss estimates 
ax1[1].plot(0.006647, 2.5e-15, c='saddlebrown',marker='v',ls='', ms=12)
ax1[1].plot(0.002278, 5.7e-16, c='yellowgreen', marker='d', ls='', ms=12)
ax1[1].plot(0.001158, 1.4e-14, 'ro',ls='', ms=12)
ax1[1].plot(0.001229, 1.9e-16, 'b*' ,ls='', ms=12)
ax1[1].plot(0.001284, 4.1e-16, marker='p', c='orange', ls='', ms=12)
ax1[1].plot(0.00036, 1.6e-13, 'cP', ls='', ms=12)
ax1[1].plot(0.0003, 1.8e-14, c='deeppink',marker='x',ls='', ms=12)
ax1[1].plot(0.000187, 1.4e-16, 'g^', ls='',ms=12)
ax1[1].set_xlabel('Rossby number', fontsize=26)
ax1[1].set_ylabel('$\,\dot{M}_{2,\,BH}\,(M_\odot$yr$^{-1}$)', fontsize=26)
ax1[1].tick_params(labelsize=14)
ax1[1].legend(prop={'size': 20}, loc='upper right', frameon=True)

# setting up ticks and other params for second sub-plot
ax1[1].tick_params(axis='y',which='both', width=2,direction="in",pad=10)
ax1[1].tick_params(axis='y',which='major', length=14,direction="in")
ax1[1].tick_params(axis='x',which='both', width=2,direction='in')
ax1[1].tick_params(axis='x',which='major', length=14,direction='in',pad=10)
ax1[1].tick_params(axis='x',which='minor', length=8)
ax1[1].tick_params(axis='y',which='minor', length=8,direction='in')
ax1[1].tick_params(labelsize=26)
ax1[1].xaxis.set_ticks_position('both')
ax1[1].yaxis.set_ticks_position('both')
ax1[1].set_xlim(0.0001001,0.01)
ax1[1].set_ylim(5e-17,0.3e-12)
ax1[1].yaxis.tick_right()
ax1[1].yaxis.set_label_position("right")

bboxx = transforms.Bbox([[0.86, 0.18], [19.7, 8.84]])
plt.savefig('MassLossGlobal.pdf', bbox_inches=bboxx)