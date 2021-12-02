# -*- coding: iso-8859-1 -*-
"""
    Created on October 7 2020
    
    Description: This routine runs a series of routines for an optimal CCF analysis for precise
    Radial Velocity measurments of a time series of SPIRou data.
    
    @author: Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    Simple usage example:
    
    python ~/spirou-tools/spirou-ccf/spirou_ccf_pipeline.py --input=2??????t.fits -pvd
    
    python ~/spirou-tools/spirou-ccf/spirou_ccf_pipeline.py --input=2*t.fits --ccf_mask=/Users/eder/spirou-tools/spirou-ccf/ccf_masks/montreal_masks/Gl725B_neg_depth.mas --source_rv=-34.4 --ccf_width=60 -vp
    
    Test using small dataset
    python python ~/spirou-tools/spirou-ccf/spirou_ccf_pipeline.py --input=250528?t.fits --ccf_mask=/Users/eder/spirou-tools/spirou-ccf/ccf_masks/montreal_masks/Gl725B_neg_depth.mas --source_rv=-34.4 --ccf_width=60 -v
    
    Example using e2dsff_C.fits files to calculate the drifts
    python ~/spirou-tools/spirou-ccf/spirou_ccf_pipeline.py --input=2??????t.fits --input_fp_fiberC=2??????o_pp_e2dsff_C.fits -pvd
    
    Example using an existing fiber C drift rdb data file to calculate the drifts
    python ~/spirou-tools/spirou-ccf/spirou_ccf_pipeline.py --input=2??????t.fits --sci_drift_data=TOI-1759_FiberC__smart_fp_mask.mas__False__0.6.132__YJHK_fpdrift.rdb -pvd
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
import spiroulib
from copy import deepcopy
import matplotlib.pyplot as plt
import ccf2rv
import drift_lib

spirou_ccf_dir = os.path.dirname(__file__)
mask_repository = os.path.join(spirou_ccf_dir, 'ccf_masks/apero_masks/')
telluric_mask_repository = os.path.join(spirou_ccf_dir,'ccf_masks/telluric/')
fp_mask = os.path.join(mask_repository,'fp.mas')
h2o_mask = os.path.join(telluric_mask_repository,'trans_h2o_abso_ccf.mas')
tel_mask = os.path.join(telluric_mask_repository,'trans_others_abso_ccf.mas')
drift_repository = os.path.join(spirou_ccf_dir,'drifts/')
cal_fp_rv_file = os.path.join(drift_repository, "CALIBRATION_FiberC__smart_fp_mask.mas__False__0.6.132__YJHK_fpdrift.rdb")
fp_mask = os.path.join(spirou_ccf_dir,'ccf_masks/montreal_masks/smart_fp_mask.mas')

def run_spirou_ccf(inputdata, ccf_mask, drifts, telluric_rv=False, use_efits=False, normalize_ccfs=True, save_output=True, source_rv=0., ccf_width=100, vel_sampling=1.8, run_analysis=True, output_template="", interp_with_gp=False, verbose=False, plot=False) :
    """
        Description: wrapper function to run an optimal CCF analysis of a time series of SPIRou spectra.
        This function run the following steps:
        1. Reduce a time series of SPIRou spectra
        2. Calculate CCF for the template spectrum
        3. Calculate the CCF for each reduced spectrum in the time series (including star, and tellurics)
        4. Run the CCF template matching analysis on the CCF time series data.
        """
    reduced = reduc_lib.reduce_timeseries_of_spectra(inputdata, ccf_mask, rvfile="", use_efits=use_efits, fix_gaps=False, max_gap_size=8.0, nsig_clip=4.0, align_spectra=True, vel_sampling=vel_sampling, min_window_size=float(ccf_width), tel_mask=tel_mask, h2o_mask=h2o_mask, telluric_rv=telluric_rv, ccf_width=ccf_width, source_rv=source_rv, output_template=output_template, interp_with_gp=interp_with_gp, verbose=verbose)
    
    fluxkey, fluxerrkey = reduced["fluxkey"], reduced["fluxerrkey"]
    waveskey, wavekey = reduced["waveskey"], reduced["wavekey"]
    
    array_of_spectra, array_of_espectra = reduced["array_of_spectra"], reduced["array_of_espectra"]
    spectra, template = reduced["spectra"], reduced["template"]
    telluric_template = reduced["telluric_template"]
    espectra, etemplate, efits_ok = reduced["espectra"], reduced["etemplate"], reduced["efits_ok"]
    
    ccf_params, ccfmask = reduced["ccf_params"], reduced["ccfmask"]
    tell_ccf_params, tell_ccfmask = reduced["tell_ccf_params"], reduced["tell_ccfmask"]
    h2o_ccf_params, h2o_ccfmask = reduced["h2o_ccf_params"], reduced["h2o_ccfmask"]
    
    if verbose :
        print("******************************")
        print("STEP: calculating CCFs ...")
        print("******************************")

    order_subset_for_mean_ccf = [2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 28, 29, 30, 31, 32, 33, 34, 35, 36, 40, 41, 42, 43, 44, 45, 46]

    if telluric_rv :
        order_subset_for_mean_tellccf = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,24,25,26,27,28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]

    fluxes_tmpl, waves_sf_tmpl = [], []
    for order in range(49) :
        fluxes_tmpl.append(template[order]['flux'])
        waves_sf_tmpl.append(template[order]['wl'])
    fluxes_tmpl = np.array(fluxes_tmpl, dtype=float)
    waves_sf_tmpl = np.array(waves_sf_tmpl, dtype=float)
    base_header = deepcopy(array_of_spectra["spectra"][0]["header"])

    # run ccf on template
    template_ccf = ccf_lib.run_ccf_eder(reduced["ccf_params"], waves_sf_tmpl, fluxes_tmpl, base_header, reduced["ccfmask"], rv_drifts={}, targetrv=ccf_params["SOURCE_RV"], valid_orders=order_subset_for_mean_ccf, normalize_ccfs=normalize_ccfs, output=False, plot=False, verbose=False)

    # If no source RV is provided then adopt the one measured from the template
    #if source_rv != 0 :
    #    if np.abs(source_rv - template_ccf["header"]['RV_OBJ']) > 2*vel_sampling :
    #        print("WARNING: measure RV on template is different than provided RV")

    ccf_params["SOURCE_RV"] = template_ccf["header"]['RV_OBJ']
    ccf_params["CCF_WIDTH"] = 7 * template_ccf["header"]['CCFMFWHM']
    if telluric_rv :
        tell_ccf_params["CCF_WIDTH"] = ccf_params["CCF_WIDTH"]
        h2o_ccf_params["CCF_WIDTH"] = ccf_params["CCF_WIDTH"]

    if verbose :
        print("Template RV={0:.5f} km/s CCF width={1:.0f} km/s".format(ccf_params["SOURCE_RV"], ccf_params["CCF_WIDTH"]))
    if plot :
        templ_legend = "Template of {}".format(template_ccf["header"]["OBJECT"].replace(" ",""))
        plt.plot(template_ccf['RV_CCF'], template_ccf['MEAN_CCF'], "-", color='green', lw=2, label=templ_legend, zorder=2)

    calib_rv, drift_rv,  = [], []
    tell_rv, h2o_rv = [], []
    mean_fwhm, mean_tell_fwhm, mean_h2o_fwhm = [], [], []
    sci_ccf_file_list = []
    tell_ccf_file_list, h2o_ccf_file_list = [], []

    for i in range(spectra['nspectra']) :
    
        if verbose :
            print("Running CCF on file {0}/{1} -> {2}".format(i,spectra['nspectra']-1,os.path.basename(spectra['filenames'][i])))

        rv_drifts = drifts[i]

        fluxes, waves_sf = [], []
        tellfluxes, waves = [], []
        for order in range(49) :
            fluxes.append(spectra[fluxkey][order][i])
            waves_sf.append(spectra[waveskey][order][i])
            tellfluxes.append(spectra["recons"][order][i])
            waves.append(spectra["waves"][order][i])

        fluxes = np.array(fluxes, dtype=float)
        waves_sf = np.array(waves_sf, dtype=float)
        tellfluxes = np.array(tellfluxes, dtype=float)
        waves = np.array(waves, dtype=float)

        # run main routine to process ccf on science fiber
        header = array_of_spectra["spectra"][i]["header"]

        # run an adpated version of the ccf codes using reduced spectra as input
        sci_ccf = ccf_lib.run_ccf_eder(ccf_params, waves_sf, fluxes, header, ccfmask, rv_drifts=rv_drifts, filename=spectra['filenames'][i], targetrv=ccf_params["SOURCE_RV"], valid_orders=order_subset_for_mean_ccf, normalize_ccfs=normalize_ccfs, output=save_output, plot=False, verbose=False)

        sci_ccf_file_list.append(os.path.abspath(sci_ccf["file_path"]))

        calib_rv.append(sci_ccf["header"]['RV_OBJ'])
        mean_fwhm.append(sci_ccf["header"]['CCFMFWHM'])
        drift_rv.append(sci_ccf["header"]['RV_DRIFT'])
    
        if telluric_rv :
            tell_header = deepcopy(array_of_spectra["spectra"][i]["header"])
            h2o_header = deepcopy(array_of_spectra["spectra"][i]["header"])
            # run a adpated version fo the ccf codes using reduced spectra as input
            tell_ccf = ccf_lib.run_ccf_eder(tell_ccf_params, waves, tellfluxes, tell_header, tell_ccfmask, filename=spectra['filenames'][i],valid_orders=order_subset_for_mean_tellccf, normalize_ccfs=normalize_ccfs, output=save_output, plot=False, verbose=False)

            h2o_ccf = ccf_lib.run_ccf_eder(h2o_ccf_params, waves, tellfluxes, h2o_header, h2o_ccfmask, filename=spectra['filenames'][i],valid_orders=order_subset_for_mean_tellccf, normalize_ccfs=normalize_ccfs, output=save_output, plot=False, verbose=False)

            tell_ccf_file_list.append(os.path.abspath(tell_ccf["file_path"]))
            h2o_ccf_file_list.append(os.path.abspath(h2o_ccf["file_path"]))
            
            tell_rv.append(tell_ccf["header"]['RV_OBJ'])
            h2o_rv.append(h2o_ccf["header"]['RV_OBJ'])
            
            mean_tell_fwhm.append(tell_ccf["header"]['CCFMFWHM'])
            mean_h2o_fwhm.append(h2o_ccf["header"]['CCFMFWHM'])
            
            if plot :
                if i == spectra['nspectra'] - 1 :
                    tellegend, h20legend = "Other tellurics", r"H$_2$O"
                else :
                    tellegend, h20legend = None, None
                plt.plot(tell_ccf['RV_CCF'],tell_ccf['MEAN_CCF'], "--", color='#d62728', label=tellegend)
                plt.plot(h2o_ccf['RV_CCF'],h2o_ccf['MEAN_CCF'], ":", color='#1f77b4', label=h20legend)
        else :
            tell_rv.append(np.nan)
            h2o_rv.append(np.nan)

        if verbose :
            print("Spectrum: {0} DATE={1} Sci_RV={2:.5f} km/s RV_DRIFT={3:.5f} km/s Tell_RV={4:.5f} km/s H2O_RV={5:.5f} km/s".format(os.path.basename(spectra['filenames'][i]), sci_ccf["header"]["DATE"], sci_ccf["header"]['RV_OBJ'], sci_ccf["header"]["RV_DRIFT"], tell_rv[i], h2o_rv[i]))

        if efits_ok :
            # run main routine to process ccf on science fiber
            eheader = array_of_espectra["spectra"][i]["header"]
    
        # run a adpated version of the ccf codes using reduced spectra as input
        #esci_ccf = ccf_lib.run_ccf_eder(ccf_params, espectra["wl_sf"], efluxes, eheader, ccfmask, rv_drifts=rv_drifts, filename=espectra['filenames'][i], targetrv=source_rv, valid_orders=order_subset_for_mean_ccf, normalize_ccfs=normalize_ccfs, output=save_output, plot=False, verbose=False)

        if plot :
            if i == spectra['nspectra'] - 1 :
                scilegend = "{}".format(sci_ccf["header"]["OBJECT"].replace(" ",""))
            else :
                scilegend = None
            #plt.plot(esci_ccf['RV_CCF'],sci_ccf['MEAN_CCF']-esci_ccf['MEAN_CCF'], "--", label="spectrum")
            plt.plot(sci_ccf['RV_CCF'], sci_ccf['MEAN_CCF'], "-", color='#2ca02c', alpha=0.5, label=scilegend, zorder=1)

    mean_fwhm = np.array(mean_fwhm)
    velocity_window = 1.5*np.nanmedian(mean_fwhm)

    if telluric_rv :
        mean_tell_fwhm = np.array(mean_tell_fwhm)
        mean_h2o_fwhm = np.array(mean_h2o_fwhm)

    if plot :
        plt.xlabel('Velocity [km/s]')
        plt.ylabel('CCF')
        plt.legend()
        plt.show()

        calib_rv, median_rv = np.array(calib_rv), np.nanmedian(calib_rv)
        plt.plot(spectra["bjds"], (calib_rv  - median_rv), 'o', color='#2ca02c', label="Sci RV = {0:.4f} km/s".format(median_rv))
        plt.plot(spectra["bjds"], (mean_fwhm  - np.nanmean(mean_fwhm)), '--', color='#2ca02c', label="Sci FWHM = {0:.4f} km/s".format(np.nanmean(mean_fwhm)))
        
        drift_rv = np.array(drift_rv)
        
        mean_drift, sigma_drift = np.nanmedian(drift_rv), np.nanstd(drift_rv)
        plt.plot(spectra["bjds"], drift_rv, '.', color='#ff7f0e', label="Inst. FP drift = {0:.4f}+/-{1:.4f} km/s".format(mean_drift,sigma_drift))

        if telluric_rv :
            tell_rv = np.array(tell_rv)
            zero_telldrift, sigma_telldrift = np.nanmedian(tell_rv), np.nanstd(tell_rv)
            h2o_rv = np.array(h2o_rv)
            zero_h2odrift, sigma_h2odrift = np.nanmedian(h2o_rv), np.nanstd(h2o_rv)
            plt.plot(spectra["bjds"], (tell_rv  - zero_telldrift), '-', color='#d62728', label="Telluric drift = {0:.4f}+/-{1:.4f} km/s".format(zero_telldrift, sigma_telldrift))
            plt.plot(spectra["bjds"], (h2o_rv  - zero_h2odrift), '-', color='#1f77b4', label="H2O drift = {0:.4f}+/-{1:.4f} km/s".format(zero_h2odrift, sigma_h2odrift))
            plt.plot(spectra["bjds"], (mean_tell_fwhm  - np.nanmean(mean_tell_fwhm)), ':', color='#d62728', label="Telluric FWHM = {0:.4f} km/s".format(np.nanmean(mean_tell_fwhm)))
            plt.plot(spectra["bjds"], (mean_h2o_fwhm  - np.nanmean(mean_h2o_fwhm)), ':', color='#1f77b4', label="H2O FWHM = {0:.4f} km/s".format(np.nanmean(mean_h2o_fwhm)))

        plt.xlabel(r"BJD")
        plt.ylabel(r"Velocity [km/s]")
        plt.legend()
        plt.show()

    if run_analysis :
        if verbose :
            print("Running CCF analysis: velocity_window = {0:.3f} km/s".format(velocity_window))
        
        # exclude orders with strong telluric absorption
        exclude_orders = [-1]  # to include all orders
        #exclude_orders = [11,12,22,23,24,25,26,27,37,38,39,40,47,48]

        obj = sci_ccf["header"]["OBJECT"].replace(" ","")
        drs_version = sci_ccf["header"]['VERSION']

        ccf2rv.run_ccf_analysis(sci_ccf_file_list, ccf_mask, obj=obj, drs_version=drs_version, snr_min=10., velocity_window=velocity_window, dvmax_per_order=vel_sampling, sanit=False, correct_rv_drift=True, save_ccf_fitsfile=True, exclude_orders = exclude_orders, plot=plot, verbose=options.verbose)

        if telluric_rv :
            tell_velocity_window = 1.5*np.nanmedian(mean_tell_fwhm)
            ccf2rv.run_ccf_analysis(tell_ccf_file_list, tel_mask, obj=obj, drs_version=drs_version, snr_min=10.,velocity_window=tell_velocity_window, dvmax_per_order=vel_sampling, sanit=False, correct_rv_drift=False, save_ccf_fitsfile=False, plot=False, verbose=options.verbose)
            
            h2o_velocity_window = 1.5*np.nanmedian(mean_h2o_fwhm)
            ccf2rv.run_ccf_analysis(h2o_ccf_file_list, h2o_mask, obj=obj, drs_version=drs_version, snr_min=10.,velocity_window=h2o_velocity_window, dvmax_per_order=vel_sampling, sanit=False, correct_rv_drift=False, save_ccf_fitsfile=False, plot=False, verbose=options.verbose)
#-- end of spirou_ccf routine

parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help="Spectral *t.fits data pattern",type='string',default="*t.fits")
parser.add_option("-c", "--input_fp_fiberC", dest="input_fp_fiberC", help="FP fiber C e2ds data pattern",type='string',default="")
parser.add_option("-m", "--ccf_mask", dest="ccf_mask", help="Input CCF mask",type='string',default="")
parser.add_option("-o", "--output_template", dest="output_template", help="Output template spectrum",type='string',default="")
parser.add_option("-n", "--sci_drift_data", dest="sci_drift_data", help="Input rdb file containing drift data from FP fiber C science frames",type='string',default="")
parser.add_option("-y", "--cal_drift_data", dest="cal_drift_data", help="Input rdb file containing drift data from FP+FP calibrations",type='string',default="")
parser.add_option("-r", "--source_rv", dest="source_rv", help="Input source RV (km/s)",type='float',default=0.)
parser.add_option("-w", "--ccf_width", dest="ccf_width", help="CCF half width (km/s)",type='string',default="150")
parser.add_option("-a", "--vel_sampling", dest="vel_sampling", help="Velocity sampling for the template spectrum (km/s)",type='float',default=1.8)
parser.add_option("-f", "--epoch_max_dt", dest="epoch_max_dt", help="Maximum time difference to stack sequence (days)",type='float',default=0.5)
parser.add_option("-e", action="store_true", dest="use_efits", help="Use e.fits data (saved in the same directory as t.fits)", default=False)
parser.add_option("-s", action="store_true", dest="stack", help="Stack sequence (polar or epoch)", default=False)
parser.add_option("-l", action="store_true", dest="polar_sequence", help="Force stack of a polar sequence", default=False)
parser.add_option("-g", action="store_true", dest="interp_with_gp", help="Interpolate spectra using GP", default=False)
parser.add_option("-t", action="store_true", dest="telluric_ccf", help="Run telluric CCF", default=False)
parser.add_option("-d", action="store_true", dest="correct_drift", help="correct RV drift", default=False)
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h spirou_ccf_pipeline.py")
    sys.exit(1)

if options.verbose:
    print('Spectral t.fits data pattern: ', options.input)
    if options.ccf_mask != "":
        print('Input CCF mask: ', options.ccf_mask)
    if options.output_template != "":
        print('Output template spectrum: ', options.output_template)
    if options.sci_drift_data != "":
        print('Input rdb file containing drift data from FP fiber C science frames: ', options.sci_drift_data)
    if options.cal_drift_data != "":
        print('Input rdb file containing drift data from FP+FP calibrations: ', options.cal_drift_data)
    if options.source_rv != 0 :
        print('Input source RV (km/s): ', options.source_rv)
    print('Initial CCF width (km/s): ', options.ccf_width)
    print('Velocity sampling (km/s): ', options.vel_sampling)
    if options.stack :
        print('Maximum time difference to stack sequence (days): ', options.epoch_max_dt)

# make list of tfits data files
if options.verbose:
    print("Creating list of t.fits spectrum files...")
inputdata = sorted(glob.glob(options.input))
inputfpdata = []

if options.correct_drift :
    # If input pattern for fiber C data is not provided, try to figure it out
    #if options.input_fp_fiberC == "" :
    #    options.input_fp_fiberC = (options.input).replace("t.fits","o_pp_e2dsff_C.fits")
    if options.verbose :
        print('FP fiber C e2ds data pattern: ', options.input_fp_fiberC)
    # make list of e2ds FP data files
    if options.verbose:
        print("Creating list of FP fiber C e2ds spectrum files...")
    inputfpdata = sorted(glob.glob(options.input_fp_fiberC))


if options.stack :
    if options.polar_sequence :
        polar_sets = spiroulib.generate_polar_sets(inputdata, verbose=options.verbose)
        inputdata = spiroulib.stack_polar_sequence(polar_sets, overwrite=True)
        # E. Martioli Sep 20, 2021 -> we need a function spiroulib.stack_e2ds_polar_sequence to stack
        # e2ds files from fiber C data so we can use the drift measure from the stack. It is currently
        # using the drift calculated from the base image in the sequence, which should be ok.
        #if len(inputfpdata) :
        #    fppolar_sets = spiroulib.generate_polar_sets(inputfpdata, verbose=options.verbose)
        #    inputfpdata = spiroulib.stack_e2ds_polar_sequence(fppolar_sets, overwrite=True)
    else :
        epoch_sets = spiroulib.generate_epoch_sets(inputdata, maxdt=options.epoch_max_dt, verbose=options.verbose)
        inputdata = spiroulib.stack_polar_sequence(epoch_sets, overwrite=True)
        #if len(inputfpdata) :
        #    fpepoch_sets = spiroulib.generate_epoch_sets(inputfpdata, maxdt=options.epoch_max_dt, verbose=options.verbose)
        #    inputfpdata = spiroulib.stack_e2ds_polar_sequence(fpepoch_sets, overwrite=True)

# Initialize drift containers with zeros
drifts = drift_lib.get_zero_drift_containers(inputdata)
# Initialize drift containers with values from header (calculated by APERO)
#drifts = drift_lib.get_rv_drifts(inputdata)


if options.correct_drift :
    # Figure out drifts from either an input file or from e2ds fiberC data
    sci_fp_rv_file = options.sci_drift_data
    if options.cal_drift_data != "" :
        # Replace existing file in the drift repository by the input file name
        cal_fp_rv_file = options.cal_drift_data

    if sci_fp_rv_file == "" and len(inputfpdata) == 0:
        # Try to find *e.fits files to measure drift from fiber C data
        inputfpdata = []
        for i in range(len(inputdata)) :
            efilename = inputdata[i].replace("t.fits","e.fits")
            if os.path.exists(efilename) :
                inputfpdata.append(efilename)
            else :
                print("WARNING: failed to find e2ds fiber-C file: {}".format(efilename))
                
        if len(inputfpdata) != len(inputdata) :
            inputfpdata_tmp = []
            e2dsfffilename = inputdata[i].replace("t.fits","o_pp_e2dsff_C.fits")
            if os.path.exists(e2dsfffilename) :
                inputfpdata_tmp.append(e2dsfffilename)
            else :
                print("WARNING: failed to find e2ds fiber-C file: {}".format(efilename))
            if len(inputfpdata_tmp) == len(inputdata) or len(inputfpdata) == 0:
                inputfpdata = inputfpdata_tmp
                
    # If input fp data is different than the input spectral data issue a warning
    if len(inputfpdata) != len(inputdata) :
        print("WARNING: Length of input fp data = {} is different than the length of input spectra = {}.".format(len(inputfpdata),len(inputdata)))

    if len(inputfpdata) :
        sci_fp_rv_file = reduc_lib.run_spirou_fp_ccf(inputfpdata, fp_mask, ccf_width=9.0, nsig_clip=4, vel_sampling=options.vel_sampling, align_spectra=True, save_output=True, plot=options.plot, verbose=options.verbose)
    else :
        print("WARNING: no FP Fiber-C data detected, make sure to have either *e.fits or *o_pp_e2dsff_C.fits files in the same directory as *t.fits data. Skipping drift correction!")

    if sci_fp_rv_file != "" and cal_fp_rv_file != "" :
        if options.verbose :
            print("Science drift data file: ",sci_fp_rv_file)
            print("Calibration drift data file: ",cal_fp_rv_file)
        # Sci and cal drift files exists then we calculate drifts and feed them into the drift containers
        drifts = drift_lib.calculate_drifts_from_Cfiber_CCFRVs(sci_fp_rv_file, cal_fp_rv_file, subtract_median=True, output="", drifts=drifts, plot=options.plot, verbose=options.verbose)

if len(inputdata) <=1 :
    print("ERROR: the input data must be larger than 1 spectrum. If running a polarimetric sequence you may remove the option -s. Exiting ...")
    exit()

if options.ccf_mask == "" :
    teff = spiroulib.get_teff_from_data(inputdata)
    ccf_mask = ccf_lib.select_best_ccf_mask(teff, mask_repository)
    if options.verbose :
        print("Object Temperature: {0} K".format(teff))
        print("Selected mask: {0}".format(ccf_mask))
else :
    ccf_mask = options.ccf_mask

run_spirou_ccf(inputdata, ccf_mask, drifts, telluric_rv=options.telluric_ccf, use_efits=options.use_efits, normalize_ccfs=True, save_output=True, source_rv=options.source_rv, ccf_width=options.ccf_width, vel_sampling=options.vel_sampling, run_analysis=True, output_template=options.output_template, interp_with_gp=options.interp_with_gp, verbose=options.verbose, plot=options.plot)
