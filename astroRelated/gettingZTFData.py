"""
A simple script to batch download ZTF light curves via HTTP request
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord

