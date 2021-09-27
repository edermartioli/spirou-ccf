# -*- coding: iso-8859-1 -*-
"""
    Created on May 27 2021
    
    Description: This routine is a wrapper to run CCF analysis on FP fiber-C data
    @author: Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    Simple usage example:
    
    python ~/spirou-tools/spirou-ccf/spirou_drift_pipeline.py --workdir="/Volumes/EDERIAP/SLS-DATA/CALIBRATIONS/" --outputdir="/Volumes/EDERIAP/SLS-DATA/CALIBRATIONS/DRIFTS/" -v
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

from optparse import OptionParser
import os,sys
import subprocess
import glob
import astropy.io.fits as fits

parser = OptionParser()
parser.add_option("-d", "--workdir", dest="workdir", help="Work directory",type='string',default="./")
parser.add_option("-o", "--outputdir", dest="outputdir", help="Output directory",type='string',default="./")
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

SPIROU_CCF_DIR = '/Users/eder/spirou-tools/spirou-ccf/'

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h link_drift_calibration.py")
    sys.exit(1)

if options.verbose:
    print('Work directory: ', options.workdir)
    print('Output directory: ', options.outputdir)

night_dirs = glob.glob("{0}/2*/".format(options.workdir))

#night_dirs = ["/Volumes/EDERIAP/SLS-DATA/CALIBRATIONS/2019-04-19/","/Volumes/EDERIAP/SLS-DATA/CALIBRATIONS/2019-04-20/","/Volumes/EDERIAP/SLS-DATA/CALIBRATIONS/2019-04-27/","/Volumes/EDERIAP/SLS-DATA/CALIBRATIONS/2019-05-01/","/Volumes/EDERIAP/SLS-DATA/CALIBRATIONS/2019-05-15/","/Volumes/EDERIAP/SLS-DATA/CALIBRATIONS/2020-02-16/","/Volumes/EDERIAP/SLS-DATA/CALIBRATIONS/2020-02-17/","/Volumes/EDERIAP/SLS-DATA/CALIBRATIONS/2020-05-10/","/Volumes/EDERIAP/SLS-DATA/CALIBRATIONS/2020-05-12/","/Volumes/EDERIAP/SLS-DATA/CALIBRATIONS/2020-05-15/","/Volumes/EDERIAP/SLS-DATA/CALIBRATIONS/2020-06-06/","/Volumes/EDERIAP/SLS-DATA/CALIBRATIONS/2020-06-30/"]

run_for_real = True

for nightdir in night_dirs :
    night = os.path.basename(os.path.normpath(nightdir))
    
    print("Running CCF analysis on FP (DRIFT) data for NIGHT:",night)
    
    # Set output file name
    output = "{0}/{1}_fiber_C_ccf_rv.rdb".format(options.outputdir,night)
    
    # Check if directory exists, and create a new one if it doesn't exist
    if os.path.exists(nightdir) :
        
        command = "python {0}spirou_ccf_analysis.py --pattern={1}/*_pp_e2dsff_C_ccf_smart_fp_mask_C.fits --bandpass=H --velocity_window=6.5 -c".format(SPIROU_CCF_DIR,nightdir)
        print("Running: ",command)
        if run_for_real :
            os.system(command)
            #subprocess.check_output(command, stderr=subprocess.STDOUT,shell=True)

        command = "cp {0}/fp__fp.mas__tcorr__none__H_fpdrift.rdb {1}".format(nightdir,output)
        print("Running: ",command)
        if run_for_real :
            os.system(command)

    else :
        print("Inexistent nigth path. Skipping ...",nightdir)
        continue

