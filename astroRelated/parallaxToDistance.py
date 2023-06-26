"""
A very simple script to convert a list of parallaxes in units of mas
from a .dat file to distances in pc
"""

from astropy import units as u
import pandas as pd

df = pd.read_csv('parallaxes.dat', names=['Parallax'])

# Convert parallaxes
distance = (df['Parallax'].tolist() * u.mas).to(u.parsec, equivalencies=u.parallax())

# print them out
for el in distance:
    print(el.value)