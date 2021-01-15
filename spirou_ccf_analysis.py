# -*- coding: iso-8859-1 -*-
"""
    Created on September 29 2020
    
    Description: This routine performs the CCF analysis for a series of SPIRou CCF data products to obtain optimal radial velocities
    
    @author: Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    Simple usage example:
    
    python ~/spirou-ccf/spirou_ccf_analysis.py --pattern=*.fits --bandpass="HK" --min_snr=20
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

from optparse import OptionParser
import os,sys

import glob
import ccf2rv

parser = OptionParser()
parser.add_option("-i", "--pattern", dest="pattern", help="Input CCF data pattern",type='string',default="")
parser.add_option("-m", "--method", dest="method", help="Method to calculate RVs",type='string',default="all")
parser.add_option("-b", "--bandpass", dest="bandpass", help="Bandpass",type='string',default="YJHK")
parser.add_option("-e", "--exclude_orders", dest="exclude_orders", help="List of orders to exclude in the analysis ",type='string',default="-1")
parser.add_option("-s", "--min_snr", dest="min_snr", help="Minimum SNR",type='string',default="0")
parser.add_option("-w", "--velocity_window", dest="velocity_window", help="Velocity window",type='float',default=10.)
parser.add_option("-a", action="store_true", dest="save_all_subproducts", help="Save all sub-products", default=False)
parser.add_option("-d", action="store_true", dest="correct_drift", help="correct RV drift", default=False)
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with spirou_ccf_analysis.py -h ")
    sys.exit(1)

if options.verbose:
    print('Input CCF data pattern: ', options.pattern)
    print('Method to calculate RVs: ', options.method)
    print('Bandpass: ', options.bandpass)
    print('List of orders to exclude in the analysis: ', options.exclude_orders)
    print('Minimum SNR: ', options.min_snr)
    print('Velocity window (km/s): ', options.velocity_window)

if options.verbose:
    print("Creating list of CCF data files...")
ccf_files = sorted(glob.glob(options.pattern))

#[0,11,12,13,15,16,20,21,22,47,48]
exclude_orders = options.exclude_orders.split(",")

for i in range(len(exclude_orders)) :
    exclude_orders[i] = int(exclude_orders[i])

# detect and organize collection of files based on: object, ccfmask, sanitize, and DRS version
ccf_collections = ccf2rv.create_collections(ccf_files, verbose=options.verbose)

save_plots=False
save_csv_table_of_results = False
save_ccf_cube = False
save_weight_table = False

if options.save_all_subproducts :
    if options.plot :
        save_plots=True
    save_csv_table_of_results = True
    save_ccf_cube = True
    save_weight_table = True

for key in ccf_collections["modes"]:
    list_of_files = ccf_collections[key]

    if options.verbose:
        print("Processing collection {0} containing {1} files".format(key, len(list_of_files)))

    tbl = ccf2rv.get_object_rv(list_of_files, collection_key=key, method=options.method, exclude_orders = exclude_orders, snr_min=float(options.min_snr), bandpass = options.bandpass, velocity_window = options.velocity_window, dvmax_per_order = 3.0, save_rdb_timeseries = True, correct_rv_drift=options.correct_drift, save_csv_table_of_results = save_csv_table_of_results, save_ccf_cube = save_ccf_cube, save_weight_table = save_weight_table, doplot=options.plot, showplots=options.plot, saveplots=save_plots, verbose=options.verbose)
