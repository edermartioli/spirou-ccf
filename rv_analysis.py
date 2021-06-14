# -*- coding: iso-8859-1 -*-
"""
    Created on September 30 2020
    
    Description: This routine runs a basic analysis and do plots of RV data
    
    @author: Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    Simple usage example:
    
    python ~/spirou-tools/spirou-ccf/rv_analysis.py --input=data/TOI-1278/*__HK__*.rdb --output=data/TOI-1278/TOI-1278.rdb -f -c -e --period=14.
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """


from optparse import OptionParser
import os,sys

import glob
import ccf2rv
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats

parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help="Input RV data pattern",type='string',default="")
parser.add_option("-o", "--output", dest="output", help="Output RV file name (end with .rdb)",type='string',default="")
parser.add_option("-r", "--period", dest="period", help="Orbital period (d)",type='string',default="")
parser.add_option("-s", "--binsize", dest="binsize", help="One epoch bin size (d)",type='string',default="1.0")
parser.add_option("-e", action="store_true", dest="combine_epochs", help="Combine points within the same epochs", default=False)
parser.add_option("-c", action="store_true", dest="calib", help="calibrate data sets", default=False)
parser.add_option("-f", action="store_true", dest="fit_orbit", help="fit orbit", default=False)
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with rv_analysis.py -h ")
    sys.exit(1)

if options.verbose:
    print('Input RV data pattern: ', options.input)
    print('Output plot file name: ', options.output)
    print('Orbital period (d): ', options.period)
    print('One epoch bin size (d): ', options.binsize)

if options.verbose:
    print("Creating list of RV data files...")
rv_files = sorted(glob.glob(options.input))

if len(rv_files) == 0:
    print("Could not find input files that match input pattern {}, exiting ... ".format(options.input))
    exit()

fixed_period = False

bjd, rv, rverr = [], [], []
for i in range(len(rv_files)):
    print(i, rv_files[i])
    
    loc_bjd, loc_rv, loc_rverr = ccf2rv.read_rv_time_series(rv_files[i])
    #plt.errorbar(loc_bjd, loc_rv, yerr=loc_rverr, fmt='o', color='r', alpha=0.4)
    if options.combine_epochs :
        loc_bjd, loc_rv, loc_rverr = ccf2rv.combine_rvs_per_epoch(loc_bjd, loc_rv, loc_rverr, one_epoch_size=float(options.binsize), median=False, nsig=5)
        #plt.errorbar(loc_bjd, loc_rv, yerr=loc_rverr, fmt='o', color='k')
    #plt.show()
    bjd.append(loc_bjd)
    rv.append(loc_rv)
    rverr.append(loc_rverr)

bjd = np.array(bjd)
rv = np.array(rv)
rverr = np.array(rverr)

if len(bjd) == 0:
    print("Time array with zero length, exiting ...")
    exit()

mean_rv = np.mean(rv, axis=0)

if options.calib :
    rv_func = lambda t, shift : mean_rv + shift

rvs_calib = []
for i in range(len(rv_files)):
    basename = os.path.basename(rv_files[i])
    
    if options.calib :
        guess = [0.0001]
        pfit, pcov = curve_fit(rv_func, bjd[i], rv[i], p0=guess)
        rv_fit = rv_func(bjd[i], *pfit)
        rv_calib = rv[i] - pfit[0]
        
        if options.plot :
            plt.errorbar(bjd[i], rv_calib, yerr=rverr[i], linestyle="None", fmt='o', alpha = 0.5, label="{0}: RV{1:+.3f} km/s".format(basename,pfit[0]))
    
        sigma_eff = np.median(rverr[i])

        if options.fit_orbit :
            fit = ccf2rv.fitorbit(bjd[i], rv_calib, rverr[i], fixed_period=fixed_period, period=float(options.period), plot=False, verbose=False)
            if options.plot :
                plt.plot(fit["bjd_long_model"], fit["rv_long_model"],':')
            print("{0}: P={1:.2f}d K={2:.3f}+-{3:.3f} km/s rms={4:.1f} m/s mad={5:.1f} m/s sig_eff={6:.1f} m/s File:{7}".format(i,fit["period"],fit["K"],fit["Kerr"],fit["rms_residuals"],fit["mad_residuals"],sigma_eff*1000,basename))
        else :
            rms = np.std(rv_calib)
            mad = stats.median_absolute_deviation(rv_calib)
            print("{0}: rms={1:.1f} m/s mad={2:.1f} m/s sig_eff={3:.1f} m/s File:{4}".format(i,rms*1000,mad*1000,sigma_eff*1000,basename))

    else :
        rv_calib = rv[i]
        if options.plot:
            plt.errorbar(bjd[i], rv[i], yerr=rverr[i], linestyle="None", fmt='o', alpha = 0.5, label="{}".format(basename))
        sigma_eff = np.median(rverr[i])

        if options.fit_orbit :
            fit = ccf2rv.fitorbit(bjd[i], rv[i], rverr[i], fixed_period=fixed_period, period=float(options.period), plot=False, verbose=False)
            if options.plot :
                plt.plot(fit["bjd_long_model"], fit["rv_long_model"],':')
            print("{0}: P={1:.2f}d K={2:.3f}+-{3:.3f} km/s rms={4:.1f} m/s mad={5:.1f} m/s sig_eff={6:.1f} m/s File:{7}".format(i,fit["period"],fit["K"],fit["Kerr"],fit["rms_residuals"],fit["mad_residuals"],sigma_eff*1000,basename))
        else :
            rms = np.std(rv_calib)
            mad = stats.median_absolute_deviation(rv_calib)
            print("{0}: rms={1:.1f} m/s mad={2:.1f} m/s sig_eff={3:.1f} m/s File:{4}".format(i,rms*1000,mad*1000,sigma_eff*1000,basename))

    rvs_calib.append(rv_calib)

rvs_calib = np.array(rvs_calib)

median_rv_calib = np.median(rvs_calib, axis=0)
if len(rv_files) == 1 :
    mad_rv_calib = rverr[0]
else :
    mad_rv_calib = np.median(np.abs(rvs_calib - median_rv_calib), axis=0) / 0.67449

sigma_eff = np.median(mad_rv_calib)

if options.plot :
    plt.errorbar(bjd[0], median_rv_calib, yerr=mad_rv_calib, linestyle="None", fmt='o', color='k', label="Median RV")

if options.fit_orbit :
    fit = ccf2rv.fitorbit(bjd[0], median_rv_calib, mad_rv_calib, fixed_period=fixed_period, period=float(options.period), plot=False, verbose=False)
    if options.plot :
        plt.plot(fit["bjd_long_model"], fit["rv_long_model"],'-', color='k')
    print("Median RVs: P={0:.2f}d K={1:.3f}+-{2:.3f} km/s rms={3:.1f} m/s mad={4:.1f} m/s sig_eff={5:.1f} m/s".format(fit["period"],fit["K"],fit["Kerr"],fit["rms_residuals"],fit["mad_residuals"],sigma_eff*1000))
else:
    rms = np.std(median_rv_calib)
    mad = stats.median_absolute_deviation(median_rv_calib)
    print("Median RVs: rms={0:.1f} m/s mad={1:.1f} m/s sig_eff={2:.1f} m/s".format(rms*1000,mad*1000,sigma_eff*1000))

if options.plot :
    plt.xlabel('BJD')
    plt.ylabel('Velocity [km/s]')
    plt.legend()

if options.output != "":
    ccf2rv.save_rv_time_series(options.output, bjd[0], median_rv_calib, mad_rv_calib, time_in_rjd=True, rv_in_mps=False)
    
    if options.plot :
        plt.savefig((options.output).replace(".rdb",".png"))

if options.plot :
    plt.show()



