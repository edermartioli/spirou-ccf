# -*- coding: iso-8859-1 -*-
"""
    Created on January 13 2021
    
    Description: This routine plots the template spectrum and the CCF mask to allow visual check
    
    @author: Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    Simple usage example:
    
    python spirou_ccf_mask.py --input=AUMic_template.fits --mask=/Users/eder/spirou/rv_analysis/ccf_mask_3500K.mas --source_rv=-4.50
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

from optparse import OptionParser
import os,sys

import ccf_lib

from scipy import constants
import numpy as np

import astropy.io.fits as fits
import matplotlib.pyplot as plt

parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help="Input template spectrum",type='string',default="")
parser.add_option("-r", "--source_rv", dest="source_rv", help="Source RV in km/s",type='string', default="")

parser.add_option("-1", "--mask1", dest="mask1", help="CCF mask 1",type='string', default="")
parser.add_option("-2", "--mask2", dest="mask2", help="CCF mask 2",type='string', default="")

parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with spirou_ccf_mask.py -h ")
    sys.exit(1)

if options.verbose:
    print('Input template spectrum: ', options.input)
    print('Source RV in km/s: ', options.source_rv)
    print('CCF mask 1: ', options.mask1)
    print('CCF mask 2: ', options.mask2)


source_rv = float(options.source_rv)

MASK_COLS = ['ll_mask_s', 'll_mask_e', 'w_mask']

masktable1 = ccf_lib.read_mask(options.mask1, MASK_COLS)
d_1, ctr_1, w_1 = ccf_lib.get_mask(masktable1, 1.7, 0.0)
w_1 /= np.max(w_1)
bottom_1 = (1.0 - w_1)
ctr_1 *= (1. + source_rv / (constants.c / 1000.))

if options.mask2 != "" :
    masktable2 = ccf_lib.read_mask(options.mask2, MASK_COLS)
    d_2, ctr_2, w_2 = ccf_lib.get_mask(masktable2, 1.7, 0.0)
    w_2 /= np.max(w_2)
    bottom_2 = (1.0 - w_2)
    ctr_2 *= (1. + source_rv / (constants.c / 1000.))

# Load template spectrum
hdu = fits.open(options.input)
wl, flux, eflux = hdu[1].data["wavelength"], hdu[1].data["flux"], hdu[1].data["eflux"]

plt.plot(wl, flux, '-', label="Template spectrum")

plt.bar(ctr_1, w_1, d_1, bottom=bottom_1, color="orange", label="CCF mask={}".format(options.mask1))

if options.mask2 != "" :
    plt.bar(ctr_2, w_2, d_2, bottom=bottom_2, color="green", label="CCF mask={}".format(options.mask2))

plt.legend()

plt.xlabel(r"Wavelength [nm]")
plt.ylabel(r"Flux")

plt.show()
