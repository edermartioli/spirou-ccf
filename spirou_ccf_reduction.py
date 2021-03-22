# -*- coding: iso-8859-1 -*-
"""
    Created on October 7 2020
    
    Description: This routine calculates the CCF data for a given data set.
    
    @author: Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    Simple usage example:
    
    python ~/spirou-tools/spirou-ccf/spirou_ccf_reduction.py --input=2*stack_t.fits --ccf_mask=/Users/eder/spirou-tools/spirou-ccf/ccf_masks/montreal_masks/Gl846_neg_depth.mas --source_rv=-7.41 --ccf_width=60 -v
    
     python ~/spirou-tools/spirou-ccf/spirou_ccf_reduction.py --input=2*stack_t.fits --ccf_mask=/Users/eder/spirou-tools/spirou-ccf/ccf_masks/montreal_masks/Gl725B_neg_depth.mas --source_rv=-34.4 --ccf_width=60 -vp
    
    Test using small dataset
    python python ~/spirou-tools/spirou-ccf/spirou_ccf_reduction.py --input=250528?t.fits --ccf_mask=/Users/eder/spirou-tools/spirou-ccf/ccf_masks/montreal_masks/Gl725B_neg_depth.mas --source_rv=-34.4 --ccf_width=60 -v
    
    Test using *e.fits files
    python ~/spirou-tools/spirou-ccf/spirou_ccf_reduction.py --input=250528?e.fits --ccf_mask=/Users/eder/spirou-tools/spirou-ccf/ccf_masks/montreal_masks/Gl725B_neg_depth.mas --source_rv=-34.4 --ccf_width=60 -v
    
    Test using telluric mask:
    python ~/spirou-tools/spirou-ccf/spirou_ccf_reduction.py --input=2*e.fits --ccf_mask=/Users/eder/spirou-tools/spirou-ccf/ccf_masks/telluric/trans_h2o_abso_ccf.mas --source_rv=0 --ccf_width=40 -v
    
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

from optparse import OptionParser
import os,sys
import astropy.io.fits as fits
import glob
import numpy as np
from scipy import constants

import reduc_lib
import ccf_lib
import spectrumlib
from copy import deepcopy
import matplotlib.pyplot as plt
import ccf2rv

spirou_ccf_dir = os.path.dirname(__file__) + '/'
mask_repository = spirou_ccf_dir + 'ccf_masks/'
telluric_mask_repository = spirou_ccf_dir + 'ccf_masks/telluric/'
fp_mask = mask_repository + 'fp.mas'
h2o_mask = telluric_mask_repository + 'trans_h2o_abso_ccf.mas'
tel_mask = telluric_mask_repository + 'trans_others_abso_ccf.mas'

def get_rv_drifts(tfits, verbose=False) :
    loc = {}
    
    if tfits.endswith("t.fits") :
        fpfits = tfits.replace("t.fits","o_pp_e2dsff_C_ccf_smart_fp_mask_C.fits")
    elif tfits.endswith("e.fits") :
        fpfits = tfits.replace("e.fits","o_pp_e2dsff_C_ccf_smart_fp_mask_C.fits")

    if fpfits != "" and os.path.exists(fpfits) :
        if verbose :
            print("Getting RV_DRIFT from file:{}".format(fpfits))
        
        hdr = fits.getheader(fpfits,1)
        
        loc["WFPDRIFT"] = "None"
        loc["RV_WAVFP"] = hdr["RV_WAVFP"]
        loc["RV_SIMFP"] = hdr["RV_SIMFP"]
        loc["RV_DRIFT"] = hdr["RV_DRIFT"]
    
    else :
        if tfits.endswith("t.fits") :
            efits = tfits.replace("t.fits","e.fits")
        elif tfits.endswith("e.fits") :
            efits = tfits

        if os.path.exists(efits) :
            print("Measuring RV_DRIFT in spectrum: {}".format(efits))
            try :
                calib_ccf = run_cal_ccf(efits, fp_mask, plot=False, verbose=False)
                hdr = fits.getheader(tfits)
                if "WFPDRIFT" in hdr.keys() :
                    loc["WFPDRIFT"] = hdr["WFPDRIFT"]
                else :
                    loc["WFPDRIFT"] = 'None'

                if "WFPDRIFT" in hdr.keys() :
                    loc["RV_WAVFP"] = hdr["RV_WAVFP"]
                else :
                    loc["RV_WAVFP"] = 'None'

                loc["RV_SIMFP"] = calib_ccf["header"]['RV_OBJ']
                loc["RV_DRIFT"] = calib_ccf["header"]['RV_OBJ']
            except:
                print("WARNING: could not calculate RV drift on file {0}, setting RV drift = 0.".format(efits))
                loc["WFPDRIFT"] ='None'
                loc["RV_WAVFP"] = 'None'
                loc["RV_SIMFP"] = 'None'
                loc["RV_DRIFT"] = 0.
        else :
            print("WARNING: could not find {} nor {}, setting RV drift = 0.".format(fpfits, efits))
            loc["WFPDRIFT"] ='None'
            loc["RV_WAVFP"] = 'None'
            loc["RV_SIMFP"] = 'None'
            loc["RV_DRIFT"] = 0.

    return loc

def run_cal_ccf(efile, fp_mask, save_output=True, plot=False, verbose=False) :
    
    rv_drifts = {}
    rv_drifts["WFPDRIFT"] ='None'
    rv_drifts["RV_WAVFP"] = 'None'
    rv_drifts["RV_SIMFP"] = 'None'
    rv_drifts["RV_DRIFT"] = 0.
    ########################################
    # Then run CCF on the calibration fiber:
    ########################################
    if verbose :
        print("Running CCF on the calibration fiber")
    # Load SPIRou reduced e FITS file -- to obtain Fiber-C FP spectrum
    spectrum_fp = spiroulib.load_spirou_AB_efits_spectrum(efile, nan_pos_filter=False, normalize_blaze=False)
    # load calibration CCF parameters
    ccf_fp_params = ccf_lib.set_ccf_params(fp_mask, science_channel=False)
    # run main routine to process ccf on calibration fiber
    calib_ccf = ccf_lib.run_ccf_new(ccf_fp_params, spectrum_fp, rv_drifts, science_channel=False, output=save_output, plot=plot, merge_headers=True)

    return calib_ccf


def run_ccf_analysis(ccf_files, mask_file, obj="", drs_version="", snr_min=20., dvmax_per_order=3.0, sanit=False, correct_rv_drift=False, save_ccf_fitsfile=False, exclude_orders = [-1], velocity_window = 10., outdir="", plot=False, verbose=False) :
    ##########################
    # Run CCF analysis to get object rvs using an optimized algorithm
    mask_basename = os.path.basename(mask_file)
    
    collection_key = "{}__{}__{}__{}".format(obj, mask_basename, sanit, drs_version)
    
    if verbose :
        print("Running CCF analysis on CCF data for collection: {}".format(collection_key))

    # set bandpass
    bandpass = "YJHK"

    if outdir == "":
        outdir = os.path.dirname(ccf_files[0])

    # form a unique batch name with mask, object and method
    batch_name = '{0}/{1}__{2}'.format(outdir, collection_key, bandpass)

    ccf = ccf2rv.get_object_rv(ccf_files, collection_key=collection_key, method="all", exclude_orders=exclude_orders, snr_min=snr_min, bandpass=bandpass,velocity_window=velocity_window, dvmax_per_order=dvmax_per_order, save_rdb_timeseries=True, correct_rv_drift=correct_rv_drift, save_csv_table_of_results=True, save_ccf_cube=False, save_weight_table=False, doplot=plot, showplots=plot, save_ccf_fitsfile=save_ccf_fitsfile, saveplots=plot, detailed_output=True, verbose=verbose)

    return ccf
    ##########################


def spirou_ccf(inputdata, ccf_mask, normalize_ccfs=True, save_output=True, telluric_rv=False, use_efits=False, align_spectra=True, fix_gaps=False, max_gap_size=8.0, nsig_clip=4.0, vel_sampling=2.0, source_rv=0., ccf_width=100, apply_drift=False, run_analysis=True, gap_diagnostic=False, verbose=False, plot_diagnostic=False, plot=False) :

    # check if there are e.fits files
    if use_efits :
        inputedata, efits_ok = reduc_lib.check_efits(inputdata, verbose)
    else:
        inputedata, efits_ok = [], False

    if verbose :
        print("******************************")
        print("STEP: Loading SPIRou data ...")
        print("******************************")
    # Load data from list of input spectra
    # First load spectra into a container
    array_of_spectra = reduc_lib.load_array_of_spirou_spectra(inputdata, correct_blaze=True, apply_berv=True, verbose=verbose)
    # Then load data into order vectors -- it is more efficient to work the reduction order-by-order
    spectra = reduc_lib.get_spectral_data(array_of_spectra, verbose=verbose)

    if efits_ok :
        array_of_espectra = reduc_lib.load_array_of_spirou_spectra(inputedata, correct_blaze=True, apply_berv=True, verbose=verbose)
        espectra = reduc_lib.get_spectral_data(array_of_espectra, verbose=verbose)
        if gap_diagnostic :
            enan_stats = reduc_lib.get_fraction_of_nans(espectra, label="in orignal data", verbose=True)
    if gap_diagnostic :
        tnan_stats = reduc_lib.get_fraction_of_nans(spectra, label="after telluric removal", verbose=True)

    if plot_diagnostic :
        reforder, refspc = 36,0
        plt.plot(espectra["waves_sf"][reforder][refspc], espectra["fluxes"][reforder][refspc],'b-', lw=0.2, label="original spectrum")
        plt.plot(spectra["waves_sf"][reforder][refspc], spectra["fluxes"][reforder][refspc],'.', color="grey", label="before gap correction")

    #if verbose :
    #    print("******************************")
    #    print("STEP: Detecting gap-free windows ...")
    #    print("******************************")
    # Detect gaps in the data and create a gap function for good windows without gaps
    # Use wide values to avoid too much clipping at this point. This will improve the noise model
    spectra = reduc_lib.get_gapfree_windows(spectra, max_vel_distance=max_gap_size, min_window_size=2*float(ccf_width), fluxkey="fluxes", wavekey="waves_sf", verbose=False)
    if efits_ok :
        espectra = reduc_lib.get_gapfree_windows(espectra, max_vel_distance=max_gap_size, min_window_size=2*float(ccf_width), fluxkey="fluxes", wavekey="waves_sf", verbose=False)

    if fix_gaps :
        spectra = reduc_lib.mask_fluxes_out_of_windows(spectra, fluxkey="fluxes", wavekey="waves_sf", verbose=False)

    if gap_diagnostic:
        gnan_stats = reduc_lib.get_fraction_of_nans(spectra, label="after 1st gap removal", fluxkey="fluxes",fluxerrkey="fluxerrs", verbose=True)

        plt.plot(enan_stats["orders"], enan_stats["orig_dv"], "-", label="From wavelength calibration")
        plt.plot(enan_stats["orders"], enan_stats["dv"], "-", label="in the orignal data")
        plt.plot(tnan_stats["orders"], tnan_stats["dv"], "-", label="after telluric removal")
        plt.plot(gnan_stats["orders"], gnan_stats["dv"], "-", label="after gap < {} km/s removal".format(max_gap_size))
        plt.xlabel("order number")
        plt.ylabel("median sampling [km/s]")
        plt.legend()
        plt.show()

        plt.plot(enan_stats["orders"], enan_stats["fnans"], "-", label="in orignal data")
        plt.plot(tnan_stats["orders"], tnan_stats["fnans"], "--", label="after telluric removal")
        plt.plot(gnan_stats["orders"], gnan_stats["fnans"], ":", label="after gap < {} km/s removal".format(max_gap_size))
        plt.xlabel("order number")
        plt.ylabel("fraction of NaNs (%)")
        plt.legend()
        plt.show()
        exit()

    if align_spectra :
        print("******************************")
        print("STEP: Aligning spectra to a common wavelength grid of {0:.2f} km/s ...".format(vel_sampling))
        print("******************************")
        # Set a common wavelength grid for all input spectra
        spectra = reduc_lib.set_common_wl_grid(spectra, vel_sampling=vel_sampling)
        # Interpolate all spectra to a common wavelength grid
        spectra = reduc_lib.align_spectra(spectra, interp_kind='cubic', verbose=False)
        
        # set keywords to inform following routines to use aligned spectra
        fluxkey, fluxerrkey = "aligned_fluxes", "aligned_fluxerrs"
        waveskey, wavekey = "aligned_waves", "common_wl"
    else :
        fluxkey, fluxerrkey = "fluxes", "fluxerrs"
        waveskey, wavekey =  "waves_sf", "wl_sf"

    if verbose :
        print("******************************")
        print("STEP: Reducing spectra ...")
        print("******************************")

        print("Calculating template of *t.fits spectra ...")
    # First run reduce routine to create template, calibrate all spectra to match template, and then
    # apply a sigma-clip
    template = reduc_lib.reduce_spectra(spectra, nsig_clip=nsig_clip, combine_by_median=True, subtract=False, fluxkey=fluxkey, fluxerrkey=fluxerrkey, wavekey=wavekey, update_spectra=True, plot=False, verbose=verbose)

    if telluric_rv :
        if verbose :
            print("Calculating template of telluric (recon) spectra ...")
        telluric_template = reduc_lib.reduce_spectra(spectra, nsig_clip=6.0, combine_by_median=True, subtract=False, fluxkey="recons", fluxerrkey="fluxerrs", wavekey="wl", update_spectra=True, plot=False, verbose=verbose)

    # Detect gaps again since the above sigma-clip procedure can introduce new gaps
    spectra = reduc_lib.get_gapfree_windows(spectra, max_vel_distance=max_gap_size, min_window_size=2*float(ccf_width), fluxkey=fluxkey, wavekey=waveskey, verbose=False)
    if fix_gaps :
        spectra = reduc_lib.mask_fluxes_out_of_windows(spectra, fluxkey=fluxkey, wavekey=waveskey, verbose=verbose)

    if plot_diagnostic :
        plt.plot(spectra["waves_sf"][reforder][refspc], spectra["fluxes"][reforder][refspc],'.', label="after gap correction")
        plt.plot(spectra[waveskey][reforder][refspc], spectra[fluxkey][reforder][refspc],'-', label="interpolated spectrum")

    if efits_ok :
        if verbose :
            print("Calculating template of *e.fits spectra ...")
        etemplate = reduc_lib.reduce_spectra(espectra, nsig_clip=nsig_clip, combine_by_median=True, subtract=False, verbose=True, plot=False, fluxkey="fluxes", fluxerrkey="fluxerrs", wavekey="wl_sf", update_spectra=True)

    if verbose :
        print("******************************")
        print("STEP: normalizing spectra ...")
        print("******************************")

    if efits_ok :
        espectra, etemplate = reduc_lib.normalize_spectra(espectra, etemplate, plot=plot_diagnostic)
        for order in range(49) :
            order_template = template[order]
            continuum = espectra["continuum_fluxes"][order]
            template[order]["flux"] /= continuum
            template[order]["fluxerr"] /= continuum
            template[order]["fluxerr_model"] /= continuum
    else :
        spectra, template = reduc_lib.normalize_spectra(spectra, template, fluxkey=fluxkey, fluxerrkey=fluxerrkey, plot=plot_diagnostic)

    if verbose :
        print("******************************")
        print("STEP: calculating weigths (1/sig^2) ...")
        print("******************************")

    if efits_ok :
        espectra = reduc_lib.calculate_weights(espectra, etemplate, use_err_model=False, plot=plot_diagnostic)
        spectra["weights"] = espectra["weights"]
    else :
        spectra = reduc_lib.calculate_weights(spectra, template, use_err_model=False, plot=plot_diagnostic)

    # load science CCF parameters
    ccf_params = ccf_lib.set_ccf_params(ccf_mask)

    # update ccf width with input value
    ccf_params["CCF_WIDTH"] = float(ccf_width)

    templ_fluxes, templ_efluxes, templ_tellfluxes = [], [], []
    for order in range(49) :
        order_template = template[order]
        templ_fluxes.append(order_template["flux"])

        if telluric_rv :
            order_telltemplate = telluric_template[order]
            templ_tellfluxes.append(order_telltemplate["flux"])

        if efits_ok :
            order_etemplate = etemplate[order]
            templ_efluxes.append(order_etemplate["flux"])

    templ_fluxes = np.array(templ_fluxes, dtype=float)
    if telluric_rv :
        templ_tellfluxes = np.array(templ_tellfluxes, dtype=float)
    if efits_ok :
        templ_efluxes = np.array(templ_efluxes, dtype=float)

    if verbose :
        print("Applying weights to stellar CCF mask ...")

    if efits_ok :
        ccfmask = ccf_lib.apply_weights_to_ccf_mask(ccf_params, espectra["wl_sf"], templ_efluxes, espectra["weights"], median=True, remove_lines_with_nans=False, verbose=True)
    else :
        ccfmask = ccf_lib.apply_weights_to_ccf_mask(ccf_params, spectra[wavekey], templ_fluxes, spectra["weights"], median=True, remove_lines_with_nans=False, verbose=True)

    if plot_diagnostic :
        inorder = ccfmask["orders"] == reforder
        orderweights = ccfmask["weights"][inorder] / np.nanmax(ccfmask["weights"][inorder])
        medianflux = np.nanmedian(spectra["fluxes"][reforder][refspc])
        orderweights *= medianflux
        bottom = (medianflux - orderweights)
        ctr = ccfmask["centers"][inorder] * (1. + source_rv / (constants.c / 1000.))
        plt.bar(ctr, orderweights, ccfmask["widths"][inorder], bottom=bottom, color="darkgreen", label="CCF mask")

        plt.xlabel("wavelength [nm]")
        plt.ylabel("flux")
        plt.legend()
        plt.show()
        #exit()

    if telluric_rv :
        if verbose :
            print("Applying weights to telluric CCF masks ...")

        #tell_ccf_params = ccf_lib.set_ccf_params(h2o_mask, telluric_masks=[tel_mask])
        tell_ccf_params = ccf_lib.set_ccf_params(tel_mask)
        h2o_ccf_params = ccf_lib.set_ccf_params(h2o_mask)
        
        tell_ccf_params["CCF_WIDTH"] = float(ccf_width)
        h2o_ccf_params["CCF_WIDTH"] = float(ccf_width)

        if efits_ok :
            tell_ccfmask = ccf_lib.apply_weights_to_ccf_mask(tell_ccf_params, spectra["wl"], templ_tellfluxes, espectra["weights"], median=False, verbose=True)
            h2o_ccfmask = ccf_lib.apply_weights_to_ccf_mask(h2o_ccf_params, spectra["wl"], templ_tellfluxes, espectra["weights"], median=False, verbose=True)
        else :
            tell_ccfmask = ccf_lib.apply_weights_to_ccf_mask(tell_ccf_params, spectra["wl"], templ_tellfluxes, np.full_like(spectra["weights"],1.0), median=False, verbose=True)
            h2o_ccfmask = ccf_lib.apply_weights_to_ccf_mask(h2o_ccf_params, spectra["wl"], templ_tellfluxes, np.full_like(spectra["weights"],1.0), median=False, verbose=True)

    if verbose :
        print("******************************")
        print("STEP: calculating CCFs ...")
        print("******************************")

    #order_subset_for_mean_ccf = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 17, 18, 19, 20, 21, 28, 29, 30, 31, 32, 33, 34, 35, 43, 44, 45, 46]
    order_subset_for_mean_ccf = [2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 28, 29, 30, 31, 32, 33, 34, 35, 36, 40, 41, 42, 43, 44, 45, 46]
    
    if telluric_rv :
        order_subset_for_mean_tellccf = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,24,25,26,27,28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]

    calib_rv, drift_rv,  = [], []
    tell_rv, h2o_rv = [], []
    mean_fwhm, mean_tell_fwhm, mean_h2o_fwhm = [], [], []
    sci_ccf_file_list = []
    tell_ccf_file_list, h2o_ccf_file_list = [], []

    for i in range(spectra['nspectra']) :
    
        if verbose :
            print("Running CCF on file {0}/{1} -> {2}".format(i,spectra['nspectra']-1,os.path.basename(spectra['filenames'][i])))
    
        # try to get rv drifts
        if apply_drift :
            rv_drifts = get_rv_drifts(spectra['filenames'][i], verbose=verbose)
        else :
            rv_drifts = {}

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

        # run a adpated version of the ccf codes using reduced spectra as input
        sci_ccf = ccf_lib.run_ccf_eder(ccf_params, waves_sf, fluxes, header, ccfmask, rv_drifts=rv_drifts, filename=spectra['filenames'][i], targetrv=source_rv, valid_orders=order_subset_for_mean_ccf, normalize_ccfs=normalize_ccfs, output=save_output, plot=False, verbose=False)

        sci_ccf_file_list.append(os.path.abspath(sci_ccf["file_path"]))

        calib_rv.append(sci_ccf["header"]['RV_OBJ'])
        mean_fwhm.append(sci_ccf["header"]['CCFMFWHM'])
        if apply_drift :
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
                plt.plot(tell_ccf['RV_CCF'],tell_ccf['MEAN_CCF'], ":", label="tellurics")
                plt.plot(h2o_ccf['RV_CCF'],h2o_ccf['MEAN_CCF'], ":", label="H2O")
        else :
            tell_rv.append(np.nan)
            h2o_rv.append(np.nan)

        if verbose :
            print("Spectrum: {0} DATE={1} SNR={2:.0f} Sci_RV={3:.5f} km/s RV_DRIFT={4:.5f} km/s Tell_RV={5:.5f} km/s H2O_RV={6:.5f} km/s".format(os.path.basename(spectra['filenames'][i]), sci_ccf["header"]["DATE"],sci_ccf["header"]["SPEMSNR"], sci_ccf["header"]['RV_OBJ'], sci_ccf["header"]["RV_DRIFT"], tell_rv[i], h2o_rv[i]))

        if efits_ok :
            # run main routine to process ccf on science fiber
            eheader = array_of_espectra["spectra"][i]["header"]
    
        # run a adpated version fo the ccf codes using reduced spectra as input
        #esci_ccf = ccf_lib.run_ccf_eder(ccf_params, espectra["wl_sf"], efluxes, eheader, ccfmask, rv_drifts=rv_drifts, filename=espectra['filenames'][i], targetrv=source_rv, valid_orders=order_subset_for_mean_ccf, normalize_ccfs=normalize_ccfs, output=save_output, plot=False, verbose=False)

        if plot :
            #plt.plot(esci_ccf['RV_CCF'],sci_ccf['MEAN_CCF']-esci_ccf['MEAN_CCF'], "--", label="spectrum")
            plt.plot(sci_ccf['RV_CCF'],sci_ccf['MEAN_CCF'], "-", label="science")

                
    mean_fwhm = np.array(mean_fwhm)
    if telluric_rv :
        mean_tell_fwhm = np.array(mean_tell_fwhm)
        mean_h2o_fwhm = np.array(mean_h2o_fwhm)
    if plot :
        plt.xlabel('Velocity [km/s]')
        plt.ylabel('CCF')
        #plt.legend()
        plt.show()

        calib_rv, median_rv = np.array(calib_rv), np.nanmedian(calib_rv)
        plt.plot(spectra["bjds"], (calib_rv  - median_rv), 'o', label="Sci RV = {0:.4f} km/s".format(median_rv))
        plt.plot(spectra["bjds"], (mean_fwhm  - np.nanmean(mean_fwhm)), '--', label="Sci FWHM = {0:.4f} km/s".format(np.nanmean(mean_fwhm)))
        
        if apply_drift :
            drift_rv = np.array(drift_rv)
            zero_drift, sigma_drift = np.nanmedian(drift_rv), np.nanstd(drift_rv)
            plt.plot(spectra["bjds"], (drift_rv  - zero_drift), '.', label="Inst. FP drift = {0:.4f}+/-{1:.4f} km/s".format(zero_drift,sigma_drift))

        if telluric_rv :
            tell_rv = np.array(tell_rv)
            zero_telldrift, sigma_telldrift = np.nanmedian(tell_rv), np.nanstd(tell_rv)
            h2o_rv = np.array(h2o_rv)
            zero_h2odrift, sigma_h2odrift = np.nanmedian(h2o_rv), np.nanstd(h2o_rv)
            plt.plot(spectra["bjds"], (tell_rv  - zero_telldrift), '-', label="Telluric drift = {0:.4f}+/-{1:.4f} km/s".format(zero_telldrift, sigma_telldrift))
            plt.plot(spectra["bjds"], (h2o_rv  - zero_h2odrift), '-', label="H2O drift = {0:.4f}+/-{1:.4f} km/s".format(zero_h2odrift, sigma_h2odrift))
            plt.plot(spectra["bjds"], (mean_tell_fwhm  - np.nanmean(mean_tell_fwhm)), ':', label="Telluric FWHM = {0:.4f} km/s".format(np.nanmean(mean_tell_fwhm)))
            plt.plot(spectra["bjds"], (mean_h2o_fwhm  - np.nanmean(mean_h2o_fwhm)), ':', label="H2O FWHM = {0:.4f} km/s".format(np.nanmean(mean_h2o_fwhm)))

        plt.xlabel('BJD')
        plt.ylabel('Velocity [km/s]')
        plt.legend()
        plt.show()

    if run_analysis :
        # exclude orders with strong telluric absorption
        #exclude_orders = [-1]  # to include all orders
        #exclude_orders = [11,12,22,23,24,25,26,27,37,38,39,40,47,48]
        exclude_orders = [0,1,10,11,12,13,14,21,22,23,24,25,26,27,37,38,39,40,47,48]

        obj = sci_ccf["header"]["OBJECT"].replace(" ","")
        drs_version = sci_ccf["header"]['VERSION']

        run_ccf_analysis(sci_ccf_file_list, ccf_mask, obj=obj, drs_version=drs_version, snr_min=10., velocity_window=1.5*np.nanmedian(mean_fwhm), dvmax_per_order=vel_sampling, sanit=False, correct_rv_drift=apply_drift, save_ccf_fitsfile=False, exclude_orders = exclude_orders, plot=False, verbose=options.verbose)

        if telluric_rv :
            run_ccf_analysis(tell_ccf_file_list, tel_mask, obj=obj, drs_version=drs_version, snr_min=10.,velocity_window=1.5*np.nanmedian(mean_tell_fwhm), dvmax_per_order=vel_sampling, sanit=False, correct_rv_drift=False, save_ccf_fitsfile=False, plot=False, verbose=options.verbose)
            run_ccf_analysis(h2o_ccf_file_list, h2o_mask, obj=obj, drs_version=drs_version, snr_min=10.,velocity_window=1.5*np.nanmedian(mean_h2o_fwhm), dvmax_per_order=vel_sampling, sanit=False, correct_rv_drift=False, save_ccf_fitsfile=False, plot=False, verbose=options.verbose)

#-- end of spirou_ccf routine


parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help="Spectral *t.fits data pattern",type='string',default="*t.fits")
parser.add_option("-m", "--ccf_mask", dest="ccf_mask", help="Input CCF mask",type='string',default="")
parser.add_option("-s", "--source_rv", dest="source_rv", help="Input source RV (km/s)",type='float',default=0.)
parser.add_option("-w", "--ccf_width", dest="ccf_width", help="CCF half width (km/s)",type='string',default="100")
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h spirou_ccf_reduction.py")
    sys.exit(1)

if options.verbose:
    print('Spectral t.fits data pattern: ', options.input)
    if options.ccf_mask != "":
        print('Input CCF mask: ', options.ccf_mask)
    print('Input source RV (km/s): ', options.source_rv)
    print('CCF width (km/s): ', options.ccf_width)

# make list of tfits data files
if options.verbose:
    print("Creating list of t.fits spectrum files...")
inputdata = sorted(glob.glob(options.input))

#spirou_ccf(inputdata, ccf_mask=options.ccf_mask, telluric_rv=True, use_efits=True, align_spectra=True, fix_gaps=False, max_gap_size=8.0, nsig_clip=4.0, vel_sampling=2.0, source_rv=options.source_rv, ccf_width=options.ccf_width, apply_drift=True, run_analysis=True, gap_diagnostic=False, verbose=options.verbose, plot=options.plot)

spirou_ccf(inputdata, ccf_mask=options.ccf_mask, telluric_rv=True, use_efits=True, align_spectra=True, fix_gaps=False, max_gap_size=5.0, nsig_clip=4.0, vel_sampling=2.0, source_rv=options.source_rv, ccf_width=options.ccf_width, apply_drift=True, run_analysis=True, verbose=options.verbose, plot=options.plot)
