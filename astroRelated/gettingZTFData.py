"""
A simple script to batch download ZTF light curves via HTTP request.
More on API https://irsa.ipac.caltech.edu/docs/program_interface/ztf_api.html
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord

data = pd.read_csv('magneticDA.csv', delimiter=',') # data file with coordinates to batch download
raArr = data1['icrsra'].to_numpy() # coordinates of objects
decArr = data1['icrsdec'].to_numpy()

# coordinates need to be in the form of degrees
# my coordinates are in the form of hourangles
# Example: input ra = 09:48:46.56 dec = 21:50:37.13
# desired output: ra = 147.194 dec = 21.8436472
for i in range(len(decArr)):
    ra, dec = raArr[i], decArr[i]
    coord = ra + " " + dec # merge ra and dec into a single string
    c = SkyCoord(coord, unit=(u.hourangle, u.deg), frame='icrs') # converting coordinates into appropriate form
    ra, dec = c.ra.value, c.dec.value
    search_rad = 0.0014 # search radius in degrees
    band_filter = 'g' # search g band; or r/i
    out_file_format = 'CSV' # other options: tsv/html/ipac_table
    url = 'https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?POS=CIRCLE '
    +str(ra)+' '+str(dec)+' '+str(search_rad)+'&BANDNAME='+band_filter+'&FORMAT='
    +out_file_format+'&BAD_CATFLAGS_MASK=32768' # drops points with bad quality flags
    r = requests.get(url, allow_redirects=True)
    open('ztfMagnetic/'+str(ra)+"_"+str(dec)+'.csv', 'wb').write(r.content) # dump output file