"""
This script estimates mass loss rates for a brown dwarf companion
in a white dwarf binary based on photospheric pollution. Mass
loss rates are estimated for Bondi Hoyle and Eddington/gravitational
formalisms.
"""

import numpy as np
from astropy import units as u
from astropy.coordinates import Distance
from scipy import interpolate
import scipy.constants
import periodictable

# physical parameters and constants
CaH_ratio = -7.8 # from model fit
GravConstant = scipy.constants.G*u.m*u.m*u.m/u.kg/u.second/u.second
A_Ca = periodictable.Ca.mass
A_H = periodictable.H.mass
P_orb = 1.906320*u.h
M_WD = 0.403 *u.M_sun
M_BD = 0.054*u.M_sun

# grid interpolation
teff = 17550
logg = 7.58
xlogg = [7.5, 8] # grid parameters
yteff = [17500,17750]
qcvzModel = [[-15.749,-16.526],[-15.726,-16.513]] # grid options
tauModel = [[-1.483,-2.311], [-1.468,-2.303]]
f = interpolate.interp2d(xlogg, yteff, qcvzModel, kind='linear')
q_cvz = f(logg, teff)
f = interpolate.interp2d(xlogg, yteff, tauModel, kind='linear')
log_tau = f(logg, teff) # log years

# physical parameter calculations
X_Ca = (A_Ca/A_H)*(10**CaH_ratio)
g = (10**logg)/100*u.m/u.s/u.s
R_WD = np.sqrt(GravConstant*M_WD/g)
R_BD = 0.08981*u.R_sun
print("WD radius: {}".format(R_WD.to(u.R_sun)))
v_escape = np.sqrt((2*GravConstant*M_BD)/R_BD)
v_escape = v_escape.to(u.km/u.s)
M_cvz = (10**q_cvz)*M_WD
tau = (10**log_tau)*u.yr
accretionRate = X_Ca*M_cvz/tau
print("Ca accretion rate: {}".format(accretionRate[0]))
print("Ca accretion rate: {}".format(accretionRate[0].to(u.g/u.second)))
orbitalSeparation = np.cbrt((M_BD+M_WD)*GravConstant*P_orb*P_orb/(4*np.pi*np.pi))
orbitalSeparationWD = orbitalSeparation*M_BD/(M_WD+M_BD)
v_orb = 2*np.pi*orbitalSeparationWD/P_orb # Keplerian velocity of the white dwarf
orbitalSeparation = orbitalSeparation - R_BD
print("Orbital separation: {}".format(orbitalSeparation.to(u.R_sun)))

#elemental abundances
atomicMasses = [element.mass for element in periodictable.elements]
with open('solarAbundances') as f:
    solarAbundances = list(map(float, f.readlines()))
atomicMasses = atomicMasses[1:] # drop neutron mass
atomicMasses = atomicMasses[:len(solarAbundances)] #truncate the mass list
zippedList = list(zip(atomicMasses, solarAbundances))
totalMass = sum(mass * abundance for mass, abundance in zippedList)
calciumMassRatio = (6.287e4*A_Ca)/totalMass

# accretion calculations
MDot = accretionRate/calciumMassRatio
print("Ca ratio by mass: {}".format(calciumMassRatio))
print("Total accreted mass: {}".format(MDot[0]))
print("Total accreted mass: {}".format(MDot[0].to(u.g/u.second)))

# bondi hoyle
velocityAcc = np.sqrt(v_escape*v_escape+v_orb*v_orb)
BD_massLoss = (MDot*orbitalSeparation*orbitalSeparation*(velocityAcc**4))/(GravConstant*GravConstant*M_WD*M_WD)
print("Bondi Hoyle accretion: {}".format(BD_massLoss.to(u.g/u.second)[0]))
print("Bondi Hoyle accretion: {}".format(BD_massLoss.to(u.M_sun/u.yr)[0]))

# gravitational
velocityAcc = np.sqrt(v_escape*v_escape+v_orb*v_orb)
BD_massLoss = (2*MDot*velocityAcc*velocityAcc*orbitalSeparation*orbitalSeparation)/(GravConstant*M_WD*R_WD)
print("Gravitational accretion: {}".format(BD_massLoss.to(u.g/u.second)[0]))
print("Gravitational accretion: {}".format(BD_massLoss.to(u.M_sun/u.yr)[0]))