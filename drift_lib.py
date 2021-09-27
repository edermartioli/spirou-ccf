# -*- coding: iso-8859-1 -*-
"""
    Created on May 27 2021
    
    Description: utilities for calculation of instrumental drifts in SPIRou RV data
    
    @authors:  Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """
import os,sys
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import ascii
import astropy.io.fits as fits
from copy import deepcopy

from scipy.optimize import minimize


def get_zero_drift_containers(inputdata) :
    
    drifts, drifts_err = np.zeros(len(inputdata)), np.zeros(len(inputdata))
    output = []
    for i in range(len(inputdata)) :
        hdr = fits.getheader(inputdata[i])
        loc = {}
        loc["FILENAME"] = hdr['FILENAME'] # Wavelength sol absolute CCF FP Drift [km/s]
        loc["WFPDRIFT"] = 'None' # Wavelength sol absolute CCF FP Drift [km/s]
        loc["RV_WAVFP"] = 'None' # RV measured from wave sol FP CCF [km/s]
        loc["RV_SIMFP"] = 'None' # RV measured from simultaneous FP CCF [km/s]
        loc["RV_DRIFT"] = drifts[i] # RV drift between wave sol and sim. FP CCF [km/s]
        loc["RV_DRIFTERR"] = drifts_err[i] # RV drift error between wave sol and sim. FP CCF [km/s]
        output.append(loc)
    
    return output


def get_rv_drifts(inputdata, verbose=False) :
    
    drifts, drifts_err = np.zeros(len(inputdata)), np.zeros(len(inputdata))
    output = []

    for i in range(len(inputdata)) :
        hdr0 = fits.getheader(inputdata[i],0)
        hdr1 = fits.getheader(inputdata[i],1)
        hdr = hdr0 + hdr1
        loc = {}

        # Get base file name from header
        loc["FILENAME"] = hdr['FILENAME']
        
        # Wavelength sol absolute CCF FP Drift [km/s]
        if "WFPDRIFT" in hdr.keys() :
            loc["WFPDRIFT"] = hdr['WFPDRIFT']
        else :
            loc["WFPDRIFT"] = 'None'
        
        # RV measured from wave sol FP CCF [km/s]
        if "RV_WAVFP" in hdr.keys() :
            loc["RV_WAVFP"] = hdr['RV_WAVFP']
        else :
            loc["RV_WAVFP"] = 'None'
        
        # RV measured from simultaneous FP CCF [km/s]
        if "RV_SIMFP" in hdr.keys() :
            loc["RV_SIMFP"] = hdr['RV_SIMFP']
        else :
            loc["RV_SIMFP"] = 'None'
        
        # RV drift between wave sol and sim. FP CCF [km/s]
        if "RV_DRIFT" in hdr.keys() :
            loc["RV_DRIFT"] = hdr['RV_DRIFT']
        else :
            loc["RV_DRIFT"] = drifts[i]
        
        # RV drift error between wave sol and sim. FP CCF [km/s]
        if "RV_DRIFTERR" in hdr.keys() :
            loc["RV_DRIFTERR"] = hdr['RV_DRIFTERR']
        else :
            loc["RV_DRIFTERR"] = drifts_err[i]

        output.append(loc)
    
    return output



def calculate_drifts_from_Cfiber_CCFRVs(sci_file, cal_file, subtract_median=False, output="", drifts=None, plot=False, verbose=False) :
    """
        Description: Function to calculate drifts from C-fiber RV files
        
        sci_file: science fiber-C RV file in format .rdb as produced by the tool ccf2rv.get_object_rv
        cal_file: calibration fiber-C RV file in format .rdb as produced by the tool ccf2rv.get_object_rv
        
        """

    sci_C_rvdata = ascii.read(sci_file, data_start=2)
    cal_C_rvdata = ascii.read(cal_file, data_start=2)

    sci_C_jd = np.array(sci_C_rvdata['rjd']) + 2400000.
    cal_C_jd = np.array(cal_C_rvdata['rjd']) + 2400000.

    sci_C_vrad, sci_C_svrad = np.array(sci_C_rvdata['vrad']), np.array(sci_C_rvdata['svrad'])
    cal_C_vrad, cal_C_svrad = np.array(cal_C_rvdata['vrad']), np.array(cal_C_rvdata['svrad'])

    sci_C_wjd = np.array(sci_C_rvdata['WAVETIME'])
    cal_C_wjd = np.array(cal_C_rvdata['WAVETIME'])

    drift, drifterr = np.zeros_like(sci_C_vrad), np.zeros_like(sci_C_vrad)

    for i in range(len(sci_C_vrad)) :
        match = cal_C_wjd == sci_C_wjd[i]
        if len(cal_C_vrad[match]) :
            drift[i] = sci_C_vrad[i] - cal_C_vrad[match][0]
            drifterr[i] = np.sqrt(sci_C_svrad[i]*sci_C_svrad[i] + cal_C_svrad[match][0]*cal_C_svrad[match][0])

    median_drift = 0.

    if subtract_median :
        median_drift = np.nanmedian(drift)
        drift -= median_drift

    if plot :
    
        plt.errorbar(sci_C_jd, drift, yerr=drifterr, fmt='o', alpha=0.7, label="Drift - {:.3f} km/s".format(median_drift))
    
        median_sci_C, median_cal_C = np.nanmedian(sci_C_vrad),np.nanmedian(cal_C_vrad)
    
        plt.errorbar(sci_C_jd, sci_C_vrad-median_sci_C, yerr=sci_C_svrad, fmt='o', alpha=0.7, label="Sci fiber C RV - {:.3f} km/s".format(median_sci_C))
        plt.errorbar(cal_C_jd, cal_C_vrad-median_cal_C, yerr=cal_C_svrad, fmt='o', alpha=0.7, label="Cal fiber C RV - {:.3f} km/s".format(median_cal_C))

        plt.xlabel(r"BJD",fontsize=15)
        plt.ylabel(r"Velocity [km/s]",fontsize=15)
        plt.legend(fontsize=15)
        plt.show()

    RV_WAVFP = sci_C_vrad - drift + median_drift
    filename = np.array(sci_C_rvdata['FILENAME'])

    if output != "" :

        outfile = open(output,"w+")
        outfile.write("rjd\tRV_DRIFT\tRV_DRIFTERR\tRV_WAVFP\tRV_SIMFP\tFILENAME\n")
        outfile.write("---\t--------\t-----------\t--------\t--------\t--------\n")
    
        for i in range(len(sci_C_jd)) :
            rjd = sci_C_jd[i] - 2400000.
            outfile.write("{0:.10f}\t{1:.5f}\t{2:.5f}\t{3:.5f}\t{4:.5f}\t{5}\n".format(rjd, drift[i], drifterr[i], RV_WAVFP[i], sci_C_vrad[i], filename[i]))

        outfile.close()


    if drifts == None :
        drifts = []
        for i in range(len(filename)) :
            loc = {}
            loc["FILENAME"] = filename[i]
            loc["WFPDRIFT"] = 'None' # Wavelength sol absolute CCF FP Drift [km/s]
            loc["RV_WAVFP"] = RV_WAVFP[i] # RV measured from wave sol FP CCF [km/s]
            loc["RV_SIMFP"] = sci_C_vrad[i] # RV measured from simultaneous FP CCF [km/s]
            loc["RV_DRIFT"] = drift[i] # RV drift between wave sol and sim. FP CCF [km/s]
            loc["RV_DRIFTERR"] = drifterr[i] # RV drift error between wave sol and sim. FP CCF [km/s]
            drifts.append(loc)
    else :
        for i in range(len(drifts)) :
            nomatch = True
            for j in range(len(filename)) :
                if drifts[i]["FILENAME"] == filename[j] :
                    drifts[i]["FILENAME"] = filename[j]
                    drifts[i]["WFPDRIFT"] = 'None' # Wavelength sol absolute CCF FP Drift [km/s]
                    drifts[i]["RV_WAVFP"] = RV_WAVFP[j] # RV measured from wave sol FP CCF [km/s]
                    drifts[i]["RV_SIMFP"] = sci_C_vrad[j] # RV measured from simultaneous FP CCF [km/s]
                    drifts[i]["RV_DRIFT"] = drift[j] # RV drift between wave sol and sim. FP CCF [km/s]
                    drifts[i]["RV_DRIFTERR"] = drifterr[j] # RV drift error between wave sol and sim. FP CCF [km/s]
                    nomatch = False
                    break
            if nomatch :
                print("WARNING: file {} has no matching drift in file {}. Skipping ...".format(drifts[i]["FILENAME"],sci_file))

    return drifts

