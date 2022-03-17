# -*- coding: iso-8859-1 -*-
"""
    Created on September 14 2021
    
    Description: This routine runs a series of routines to perform a CCF analysis in the fiber C FP data to obtain the drift correction for the SPIRou data.
    
    @author: Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    Simple usage example:
    
    python ~/spirou-tools/spirou-ccf/spirou_fp_ccf.py --input=*_pp_e2dsff_C.fits -pv
    
    python ~/spirou-tools/spirou-ccf/spirou_fp_ccf.py --input=*e.fits -pv

    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

from optparse import OptionParser
import os,sys
import glob
import numpy as np

import reduc_lib
import ccf_lib
from copy import deepcopy
import matplotlib.pyplot as plt
import ccf2rv
import astropy.io.fits as fits
from scipy import constants

spirou_ccf_dir = os.path.dirname(__file__)
fp_mask = os.path.join(spirou_ccf_dir,'ccf_masks/smart_fp_mask.mas')

parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help="FP e2ds data pattern",type='string',default="")
parser.add_option("-c", "--ccf_mask", dest="ccf_mask", help="Input CCF mask",type='string',default="")
parser.add_option("-w", "--ccf_width", dest="ccf_width", help="CCF half width (km/s)",type='string',default="9")
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h spirou_fp_ccf.py")
    sys.exit(1)

if options.verbose:
    print('FP e2ds data pattern: ', options.input)
    if options.ccf_mask != "":
        print('Input CCF mask: ', options.ccf_mask)
    print('Initial CCF width (km/s): ', options.ccf_width)

# make list of e2ds data files
if options.verbose:
    print("Creating list of e2ds spectrum files...")
inputdata = sorted(glob.glob(options.input))

if len(inputdata) == 0 :
    print("ERROR: the input data is null. Exiting ...")
    exit()

if options.ccf_mask == "" :
    ccf_mask = fp_mask
else :
    ccf_mask = options.ccf_mask

reduc_lib.run_spirou_fp_ccf(inputdata, ccf_mask, ccf_width=options.ccf_width, nsig_clip=4, vel_sampling=1.8, align_spectra=True, save_output=True, plot=options.plot, verbose=options.verbose)
