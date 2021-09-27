# -*- coding: iso-8859-1 -*-
"""
    Created on May 27 2021
    
    Description: This routine is a wrapper to run CCF analysis on FP fiber-C data
    @author: Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    Simple usage example:
    
    python ~/spirou-tools/spirou-ccf/spirou_drift_analysis.py --sci_C_rv=/Volumes/EDERIAP/SLS-ANALYSIS/TOI-1759/DRIFT/TOI-1759_FiberC__smart_fp_mask.mas__tcorr__none__YJHK_fpdrift.rdb --cal_C_rv=/Volumes/EDERIAP/SLS-ANALYSIS/2020-2021_FP_DRITS/CALIBRATION_FiberC__smart_fp_mask.mas__tcorr__none__YJHK_fpdrift.rdb --output=/Volumes/EDERIAP/SLS-ANALYSIS/TOI-1759/DRIFT/TOI-1759_ccf_drift.rdb -vpm
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

from optparse import OptionParser
import os,sys
import subprocess
import glob
import matplotlib.pyplot as plt
from astropy.io import ascii
import numpy as np
import drift_lib

parser = OptionParser()
parser.add_option("-s", "--sci_C_rv", dest="sci_C_rv", help="Science fiber C RV data",type='string',default="")
parser.add_option("-c", "--cal_C_rv", dest="cal_C_rv", help="Calibration fiber C RV data",type='string',default="")
parser.add_option("-o", "--output", dest="output", help="Output RV drift data",type='string',default="")
parser.add_option("-m", action="store_true", dest="subtract_median", help="subtract_median", default=False)
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h spirou_drift_analysis.py")
    sys.exit(1)

if options.verbose:
    print('Science fiber C RV data: ', options.sci_C_rv)
    print('Calibration fiber C RV data: ', options.cal_C_rv)
    print('Output RV drift data: ', options.output)

outvec = drift_lib.calculate_drifts_from_Cfiber_CCFRVs(options.sci_C_rv, options.cal_C_rv, subtract_median=options.subtract_median, output=options.output, plot=options.plot, verbose=options.verbose)
