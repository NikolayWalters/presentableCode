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
print('WD radius ', R_WD.to(u.R_sun))
v_escape = np.sqrt((2*GravConstant*M_BD)/R_BD)
v_escape = v_escape.to(u.km/u.s)
M_cvz = (10**q_cvz)*M_WD
tau = (10**log_tau)*u.yr
accretionRate = X_Ca*M_cvz/tau
print('Ca accretion rate ', accretionRate)
print('Ca accretion rate ', accretionRate.to(u.g/u.second)) 
orbitalSeparation = np.cbrt((M_BD+M_WD)*GravConstant*P_orb*P_orb/(4*np.pi*np.pi))
orbitalSeparationWD = orbitalSeparation*M_BD/(M_WD+M_BD)
v_orb = 2*np.pi*orbitalSeparationWD/P_orb # Keplerian velocity of the white dwarf
orbitalSeparation = orbitalSeparation - R_BD
print('Orbital separation: ',orbitalSeparation.to(u.R_sun))

# accretion calculations

#elemental abundances
atomicMasses = [element.mass for element in periodictable.elements]
solarAbundances = [2.884e10,2.288e9,55.47,0.7374,\
17.32,7.079e6,1.95e6,1.413e7,841.1,2.148e6,5.751e4,\
1.021e6,8.410e4,1e6,8373,4.449e5,5237,1.025e5,3692,\
6.287e4,34.20,2422,288.4,1.286e4,9168,8.380e5,2323,\
4.780e4,527,1226,35.97,120.6,6.089,65.79,11.32,55.15,\
6.572,23.64,4.608,11.33,0.7554,2.601,1.9,1.435,1.584,\
3.733,4.815,5.391,4.351,1.169,1.357,3.258]

totalMass = 2.884e10*1.00794+2.288e9*4.002602+55.47*6.941+0.7374*9.012182+\
17.32*10.811+7.079e6*12.0107+1.95e6*14.0067+1.413e7*15.9994+\
841.1*18.9984032+2.148e6*20.1797+5.751e4*22.98976928+1.021e6*24.3050+\
8.410e4*26.9815386+1e6*28.0855+8373*30.973762+4.449e5*32.065+\
5237*35.453+1.025e5*39.948+3692*39.0983+6.287e4*40.078+\
34.20*44.955912+2422*47.867+288.4*50.9415+1.286e4*51.9961+\
9168*54.938045+8.380e5*55.845+2323*58.933195+4.780e4*58.6934+\
527*63.546+1226*65.409+35.97*69.723+120.6*72.64+6.089*74.92160+\
65.79*78.96+11.32*79.904+55.15*83.798+6.572*85.4678+23.64*87.62+\
4.608*88.90585+11.33*91.224+0.7554*92.906+2.601*95.94+1.9*101.07+\
1.435*106.42+1.584*112.411+3.733*118.710+4.815*127.60+5.391*131.293+\
4.351*137.327+1.169*140.116+1.357*195.084+3.258*207.2
calciumMassRatio = (6.287e4*40.078)/totalMass
MDot = accretionRate/calciumMassRatio
#MDot = (M_cvz/tau)*calciumMassRatio
print('Ca ratio by mass: ', calciumMassRatio)
print('Total accreted mass: ', MDot)
print('Total accreted mass: ', MDot.to(u.g/u.second))


# bondi hoyle
velocityAcc = np.sqrt(v_escape*v_escape+v_orb*v_orb)
BD_massLoss = (MDot*orbitalSeparation*orbitalSeparation*(velocityAcc**4))/(GravConstant*GravConstant*M_WD*M_WD)

print('Bondi Hoyle ', BD_massLoss.to(u.g/u.second))

print('Bondi Hoyle ', BD_massLoss.to(u.M_sun/u.yr))
BH_val = 5.1e-17*(u.M_sun/u.yr)
print('Rho_BH', (BH_val/(4*np.pi*velocityAcc*orbitalSeparation*orbitalSeparation)).to(u.M_sun/u.R_sun/u.R_sun/u.R_sun))


# gravitational
velocityAcc = np.sqrt(v_escape*v_escape+v_orb*v_orb)

BD_massLoss = (2*MDot*velocityAcc*velocityAcc*orbitalSeparation*orbitalSeparation)/(GravConstant*M_WD*R_WD)

print('Gravitational ', BD_massLoss.to(u.g/u.second))

print('Gravitational ', BD_massLoss.to(u.M_sun/u.yr))
Edd_val = 2.0e-15*(u.M_sun/u.yr)
print('Rho_Edd', (Edd_val/(4*np.pi*velocityAcc*orbitalSeparation*orbitalSeparation)).to(u.M_sun/u.R_sun/u.R_sun/u.R_sun))