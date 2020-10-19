# -*- coding: iso-8859-1 -*-
"""
    Created on October 9 2020
    
    Description: This routine is a wrapper to organize data and run the CCF routines in several targets at LAM server
    
    @author: Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    Simple usage example:
    
    python run_ccf_at_lam.py --workdir="/home/external/emartioli/Data/" --targets="HD189733"
    
    nohup python run_ccf_at_lam.py --workdir="/home/external/emartioli/Data/" > log_of_reduction.out 2>&1 &
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

from optparse import OptionParser
import os,sys
import subprocess


parser = OptionParser()
parser.add_option("-d", "--workdir", dest="workdir", help="Work directory",type='string',default="./")
parser.add_option("-t", "--targets", dest="targets", help="List of targets",type='string',default="")
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h run_cc_at_lam.py")
    sys.exit(1)

if options.verbose:
    print('Work directory: ', options.workdir)
    print('List of targets: ', options.targets)

DSR_VERSION = "06131"

LAMDATADIR = "/net/GSP/nas12c/big_spirou/FULL_REDUCTION_CALIBDB_{}/science_targets".format(DSR_VERSION)

spirou_ccf_dir = '/home/external/emartioli/spirou-ccf'
spirou_pol_dir = '/net/GSP/spirou/spirou-polarimetry'

run_for_real = False

if options.targets != "":
    targetlist = options.targets.split(",")
    #print(targetlist)
else :
    #targetlist = ["TOI-1278","TOI-1759","TOI-1452","TOI-233","K2-147","TOI-736","K2-33","TOI-442","TOI-876","TOI-732","GL699"]
    #targetlist =["AUMIC","HD189733","WASP-80","GJ1214","K2-25","GJ3470","GL436","WASP-69","WASP-127"]
    #targetlist = ["TOI-1759","TOI-1452","TOI-233","K2-147","TOI-736","K2-33","TOI-442","TOI-876","TOI-732","GL699","WASP-80","GJ1214","K2-25","GJ3470","GL436","WASP-69","WASP-127"]
    #targetlist=["GJ3470","K2-33","TOI-732","GJ1214","TOI-442"]
    targetlist = ["AUMIC", "TOI-1452", "TOI-1759", "TOI-1278", "WASP-80", "GJ1214", "GL436","WASP-69","WASP-127","K2-25", "HD189733"]


for object in targetlist :
    
    # Set output directory name
    outdir = "{0}/{1}".format(options.workdir,object)
        
    # Check if directory exists, and create a new one if it doesn't exist
    if not os.path.exists(outdir):
        print("Inexistent outdir path. Creating new directory:",outdir)
        os.makedirs(outdir)

    try :
        # Create symbolic links to all *e.fits and *t.fits data
        command = "ln -s {0}/{1}/reduced/*/*e.fits {2}/".format(LAMDATADIR, object, outdir)
        print("Running: ",command)
        if run_for_real :
            subprocess.check_output(command, stderr=subprocess.STDOUT,shell=True)

        command = "ln -s {0}/{1}/reduced/*/*t.fits {2}/".format(LAMDATADIR, object, outdir)
        print("Running: ",command)
        if run_for_real :
            subprocess.check_output(command, stderr=subprocess.STDOUT,shell=True)

        command = "ln -s {0}/{1}/reduced/*/*_pp_e2dsff_C_ccf_smart_fp_mask_C.fits {2}/".format(LAMDATADIR, object, outdir)
        print("Running: ",command)
        if run_for_real :
            subprocess.check_output(command, stderr=subprocess.STDOUT,shell=True)

        command = "python {0}/spirou_ccf_pipeline.py --input={1}/*t.fits -v".format(spirou_ccf_dir, outdir, object)
        print("Running: ",command)
        if run_for_real :
            subprocess.check_output(command, stderr=subprocess.STDOUT,shell=True)

        command = "python {0}/spirou_pol_pipeline.py --epattern={1}/*e.fits -L".format(spirou_pol_dir,outdir,object)
        print("Running: ",command)
        if run_for_real :
            subprocess.check_output(command, stderr=subprocess.STDOUT,shell=True)

    except Exception as e:
        print(e)




